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

class lmdbDatasetwmixupview(tdata.Dataset):
  def __init__(self, db_path, transform=None):
    self.env = None
    self.txn = None
    self.transform = transform
    self.db_path = db_path
    self.db_path_img = "{}{}".format(self.db_path, ".path")
    self.videopath = {}
    self.videokeys = []
    self.setsinglevideo()
    self.mydatum = mydatum_pb2.myDatum()
    self._init_db()
    self.uuid = {}
    self.len = len(self.videokeys)

  def _init_db(self):
    self.env = lmdb.open(self.db_path,
                         readonly=True, lock=False,
                         readahead=False, meminit=False)
    self.txn = self.env.begin()

  def setkeys(self, strkey, strpath):
    if strkey in self.videopath.keys():
      self.videopath[strkey].append(strpath)
    else:
      self.videopath[strkey] = []
      self.videopath[strkey].append(strpath)

  def setsinglevideo(self):
    fpath = open(self.db_path_img, "r")
    strlines = fpath.readlines()

    for index, strline in enumerate(strlines):
      strline = strline.strip()
      if "MSU-MFSD" in strline or "REPLAY-ATTACK" in strline:
        if ".mov" in strline:
          strtokens = strline.split(".mov")
          strkey = "{}.mov".format(strtokens[0])
        else:
          strtokens = strline.split(".mp4")
          strkey = "{}.mp4".format(strtokens[0])
        self.setkeys(strkey, index)
      elif "OULU-NPU" in strline or "CASIA-MFSD" in strline:
        strkey = os.path.dirname(strline)
        self.setkeys(strkey, index)
    self.videokeys = list(self.videopath.keys())

  def __len__(self):
    return len(self.videokeys)

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

  def getpairitem(self, videoindex, label):
    revideoindex = np.random.randint(0, self.len)
    strkey = self.videokeys[revideoindex]
    listofindex = self.videopath[strkey]

    ridx = np.random.randint(0, len(listofindex))
    rindex = listofindex[ridx]

    while(videoindex == revideoindex):
      revideoindex = np.random.randint(0, self.len)
      strkey = self.videokeys[revideoindex]
      listofindex = self.videopath[strkey]
      ridx = np.random.randint(0, len(listofindex))
      rindex = listofindex[ridx]

    rimg, rlabel, rimgpath = self.getitem(rindex)

    while(label == rlabel or videoindex == revideoindex):
      revideoindex = np.random.randint(0, self.len)
      strkey = self.videokeys[revideoindex]
      listofindex = self.videopath[strkey]
      ridx = np.random.randint(0, len(listofindex))
      rindex = listofindex[ridx]

      rimg, rlabel, rimgpath = self.getitem(rindex)

    return rimg, rlabel, rimgpath

  def __getitem__(self, reindex):
    strkey = self.videokeys[reindex]
    listofindex = self.videopath[strkey]
    ridx = np.random.randint(0, len(listofindex))
    index = listofindex[ridx]

    img, label, imgpath = self.getitem(index)
    rimg, rlabel, rimgpath = self.getpairitem(reindex, label)
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
      rimg_clone = rimg.clone()

    lam = np.random.randint(1, 10)
    lam /= 10
    bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.size(), 1 - lam)
    lam = (bbx2 - bbx1) * (bby2 - bby1) / (img.size()[1] * img.size()[2])
    # print(lam, imgpath, rimgpath)
    rimg[:, bbx1:bbx2, bby1:bby2] = img[:, bbx1:bbx2, bby1:bby2]

    if rlabel == 1:
      lam = 1.0 - lam
    return img, label, imgpath, rimg, rimg_clone, rlabel, lam, self.uuid[strtoken[5]], self.uuid[strrtoken[5]]

if __name__ == '__main__':

  transforms = T.Compose([T.RandomHorizontalFlip(),
                          #T.RandomRotation(180),
                          #T.RandomCrop((256, 256)),
                          T.ToTensor()])  # 0 to 1

  mydataset = lmdbDatasetwmixupview("/home/user/data2/work_db/v4C3/Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260.db",
                          transforms)
  trainloader = DataLoader(mydataset, batch_size=100, shuffle=True, num_workers=0, pin_memory=False)
  for img, label, imgpath, rimg, rimg_clone, rlabel, lam, dname1, dname2 in trainloader:
    print (imgpath[4], lam[4])
    for idx, iti in enumerate(label):
      if iti == 1:
        to_pil_image(img[idx]).save("./aa_live_{}_{}.png".format(idx, dname1[idx]))
        to_pil_image(rimg_clone[idx]).save("./aa_fake_{}_{}.png".format(idx, dname2[idx]))
        to_pil_image(rimg[idx]).save("./aa_live_fake_cutmix_{}_{}.png".format(idx, lam[idx]))
      elif iti == 0:
        to_pil_image(img[idx]).save("./aa_fake_{}_{}.png".format(idx, dname1[idx]))
        to_pil_image(rimg_clone[idx]).save("./aa_live_{}_{}.png".format(idx, dname2[idx]))
        to_pil_image(rimg[idx]).save("./aa_fake_live_cutmix_{}_{}.png".format(idx, lam[idx]))

    break
