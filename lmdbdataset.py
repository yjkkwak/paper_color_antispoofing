import torch
import os
import lmdb
import numpy as np
import torch.utils.data as tdata
from torch.utils.data import DataLoader
from PIL import Image
import mydata.mydatum_pb2 as mydatum_pb2
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from augs.cutmix import cutmix_data
import random

class lmdbDataset(tdata.Dataset):
  def __init__(self, db_path, transform=None):
    self.env = None
    self.txn = None
    self.transform = transform
    self.db_path = db_path
    self.mydatum = mydatum_pb2.myDatum()
    self._init_db()

  def _init_db(self):
    self.env = lmdb.open(self.db_path,
                         readonly=True, lock=False,
                         readahead=False, meminit=False)
    self.txn = self.env.begin()

  def __len__(self):
    return self.env.stat()["entries"]

  def __getitem__(self, index):
    strid = "{:08}".format(index)
    lmdb_data = self.txn.get(strid.encode("ascii"))
    self.mydatum.ParseFromString(lmdb_data)
    dst = np.fromstring(self.mydatum.data, dtype=np.uint8)
    dst = dst.reshape(self.mydatum.height, self.mydatum.width, self.mydatum.channels)
    img = Image.fromarray(dst)
    label = self.mydatum.label
    imgpath = self.mydatum.path

    if self.transform is not None:
      img = self.transform(img)
      # imgp2 = self.transform(img)
      # imgp3 = self.transform(img)
    # outitem = {}
    # outitem["imgp1"] = imgp1
    # outitem["imgp2"] = imgp2
    # outitem["imgp3"] = imgp3
    # outitem["label"] = label
    # outitem["imgpath"] = imgpath
    return img, label, imgpath

class lmdbDatasetwmixup(tdata.Dataset):
  def __init__(self, db_path, transform=None):
    self.env = None
    self.txn = None
    self.transform = transform
    self.db_path = db_path
    self.mydatum = mydatum_pb2.myDatum()
    self._init_db()
    self.len = self.env.stat()["entries"]
    self.uuid = {}

  def _init_db(self):
    self.env = lmdb.open(self.db_path,
                         readonly=True, lock=False,
                         readahead=False, meminit=False)
    self.txn = self.env.begin()

  def __len__(self):
    return self.env.stat()["entries"]

  def rand_bbox(self, size, lam):
    # tensor
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

  def getitem(self, index):
    strid = "{:08}".format(index)
    lmdb_data = self.txn.get(strid.encode("ascii"))
    self.mydatum.ParseFromString(lmdb_data)
    dst = np.fromstring(self.mydatum.data, dtype=np.uint8)
    dst = dst.reshape(self.mydatum.height, self.mydatum.width, self.mydatum.channels)
    img = Image.fromarray(dst)
    label = self.mydatum.label
    imgpath = self.mydatum.path

    return img, label, imgpath

  def getpairitem(self, index, label):
    rindex = np.random.randint(0, self.len)
    while(index == rindex):
      rindex = np.random.randint(0, self.len)
    rimg, rlabel, rimgpath = self.getitem(rindex)

    while(label == rlabel or index == rindex):
      rindex = np.random.randint(0, self.len)
      rimg, rlabel, rimgpath = self.getitem(rindex)

    return rimg, rlabel, rimgpath

  def __getitem__(self, index):
    img, label, imgpath = self.getitem(index)
    rimg, rlabel, rimgpath = self.getpairitem(index, label)
    strtoken = imgpath.split("/")
    strrtoken = rimgpath.split("/")
    if strtoken[5] not in self.uuid.keys():
      self.uuid[strtoken[5]] = len(self.uuid.keys())
    if strrtoken[5] not in self.uuid.keys():
      self.uuid[strrtoken[5]] = len(self.uuid.keys())

    #print (strtoken[5], strrtoken[5], self.uuid[strtoken[5]], self.uuid[strrtoken[5]])
    if self.transform is not None:
      img = self.transform(img)
      rimg = self.transform(rimg)

    lam = np.random.randint(1, 10)
    lam /= 10
    bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.size(), 1 - lam)
    lam = (bbx2 - bbx1) * (bby2 - bby1) / (img.size()[1] * img.size()[2])
    rimg[:, bbx1:bbx2, bby1:bby2] = img[:, bbx1:bbx2, bby1:bby2]

    if rlabel == 1:
      lam = 1.0 - lam
    return img, label, imgpath, rimg, lam, self.uuid[strtoken[5]], self.uuid[strrtoken[5]]

if __name__ == '__main__':

  transforms = T.Compose([T.RandomHorizontalFlip(),
                          #T.RandomRotation(180),
                          T.RandomCrop((256, 256)),
                          T.ToTensor()])  # 0 to 1

  #mydataset = lmdbDataset("/home/user/work_db/v220401_01/Train_v220401_01_CelebA_LDRGB_LD3007_1by1_260x260.db", transforms)
  mydataset = lmdbDatasetwmixup("/home/user/work_db/v4C3/Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260.db/",
                          transforms)
  trainloader = DataLoader(mydataset, batch_size=10, shuffle=True, num_workers=0, pin_memory=False)
  for imgp1, label, imgpath, rimg, lam in trainloader:
    rand_idx = torch.randperm(rimg.shape[0])
    vimgs = torch.cat((imgp1, rimg[rand_idx[0:rimg.shape[0]//2],]), dim=0)
    vlabels = torch.cat((label, lam[rand_idx[0:rimg.shape[0]//2]]), dim=0)
    print (label)
    print(lam[rand_idx[0:rimg.shape[0]//2]])
    print (vlabels)
    vlabels = vlabels * 10
    vlabels = vlabels.type(torch.LongTensor)
    print(vlabels)
    #print (vimgs.shape, vlabels.shape)
    # imgp1 = item["imgp1"]
    # imgp2 = item["imgp2"]
    # imgp3 = item["imgp3"]
    # label = item["label"]
    # imgpath = item["imgpath"]
    #imgmix, mixlabel = cutmix_data(imgp1, label)
    # print(imgp1.shape)
    # print (rimg.shape)
    # print(label.shape)
    # print (lam.shape)
    to_pil_image(imgp1[0]).show()
    to_pil_image(rimg[0]).show()
    break


    liveidx = torch.where(label == 1)
    fakeidx = torch.where(label == 0)
    print(label)
    print(liveidx)
    print(label[0], imgpath)
    to_pil_image(imgp1[0]).show()
    print(imgp1[0].shape)
    break
    # 
    # print (imgp1.shape, imgp2.shape, imgp3.shape, label.shape, imgpath[0])
    for iii, fff in enumerate(label):
      print (fff, imgpath[iii], label[iii])
      # to_pil_image(imgp1[iii]).show()
      # break
    # break