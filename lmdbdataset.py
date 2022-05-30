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

if __name__ == '__main__':
  transforms = T.Compose([T.RandomHorizontalFlip(),
                          #T.RandomRotation(180),
                          T.RandomCrop((256, 256)),
                          T.ToTensor()])  # 0 to 1

  #mydataset = lmdbDataset("/home/user/work_db/v220401_01/Train_v220401_01_CelebA_LDRGB_LD3007_1by1_260x260.db", transforms)
  mydataset = lmdbDataset("/home/user/work_db/v4C3/Test_Protocal_4C3_OULU_1by1_260x260.db/",
                          transforms)
  trainloader = DataLoader(mydataset, batch_size=12, shuffle=True, num_workers=0, pin_memory=False)
  for imgp1, label, imgpath in trainloader:
    # imgp1 = item["imgp1"]
    # imgp2 = item["imgp2"]
    # imgp3 = item["imgp3"]
    # label = item["label"]
    # imgpath = item["imgpath"]

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