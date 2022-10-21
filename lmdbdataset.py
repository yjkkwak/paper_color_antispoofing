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
    self.db_path_img = "{}{}".format(self.db_path, ".path")
    self.videopath = {}
    self.videokeys = []
    self.setsinglevideo()
    self.mydatum = mydatum_pb2.myDatum()
    self._init_db()
    self.uuid = {}


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

  def __getitem__(self, reindex):
    strkey = self.videokeys[reindex]
    listofindex = self.videopath[strkey]

    ridx = np.random.randint(0, len(listofindex))
    index = listofindex[ridx]

    strid = "{:08}".format(index)
    lmdb_data = self.txn.get(strid.encode("ascii"))
    self.mydatum.ParseFromString(lmdb_data)
    dst = np.fromstring(self.mydatum.data, dtype=np.uint8)
    dst = dst.reshape(self.mydatum.height, self.mydatum.width, self.mydatum.channels)
    img = Image.fromarray(dst)
    label = self.mydatum.label
    imgpath = self.mydatum.path
    strtoken = imgpath.split("/")
    if strtoken[5] not in self.uuid.keys():
      self.uuid[strtoken[5]] = len(self.uuid.keys())

    if self.transform is not None:
      img = self.transform(img)
    outitem = {}
    outitem["img"] = img
    outitem["label"] = label
    outitem["imgpath"] = imgpath
    outitem["uuid"] = self.uuid[strtoken[5]]
    return outitem

class lmdbDatasettest(tdata.Dataset):
  def __init__(self, db_path, frames_total=8, transform=None):
    self.env = None
    self.txn = None
    self.transform = transform
    self.db_path = db_path
    self.db_path_img = "{}{}".format(self.db_path, ".path")
    self.videopath = {}
    self.videopath2 = {}
    self.videokeys = []
    self.setsinglevideo()
    self.mydatum = mydatum_pb2.myDatum()
    self._init_db()
    self.uuid = {}
    self.frames_total = frames_total

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
    # for kk in self.videokeys:
    #   print (kk)
    #   for vvv in self.videopath[kk]:
    #     print (vvv)
    #   break

  def __len__(self):
    return len(self.videokeys)

  def __getitem__(self, reindex):
    strkey = self.videokeys[reindex]
    listofindex = self.videopath[strkey]

    #frames_total = 8
    frames_total = self.frames_total # 1
    files_total = len(listofindex)
    interval = files_total // 12#12(casia)
    imgs = torch.zeros((frames_total, 3, 256, 256))
    # imgpath = ""
    for ii in range(frames_total):
      image_idx = ii * interval + 1
      index = listofindex[image_idx]

      strid = "{:08}".format(index)
      lmdb_data = self.txn.get(strid.encode("ascii"))
      self.mydatum.ParseFromString(lmdb_data)
      dst = np.fromstring(self.mydatum.data, dtype=np.uint8)
      dst = dst.reshape(self.mydatum.height, self.mydatum.width, self.mydatum.channels)
      img = Image.fromarray(dst)
      label = self.mydatum.label
      imgpath = self.mydatum.path
      # imgpath = "{}_{}".format(imgpath,self.mydatum.path)
      #print (index, imgpath)
      strtoken = imgpath.split("/")
      if strtoken[5] not in self.uuid.keys():
        self.uuid[strtoken[5]] = len(self.uuid.keys())

      if self.transform is not None:
        img = self.transform(img)
        imgs[ii,:] = img

    outitem = {}
    outitem["imgs"] = imgs
    outitem["label"] = label
    outitem["imgpath"] = imgpath
    outitem["uuid"] = self.uuid[strtoken[5]]
    return outitem

class lmdbDatasettestAllimages(tdata.Dataset):
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

    outitem = {}
    outitem["imgs"] = img
    outitem["label"] = label
    outitem["imgpath"] = imgpath
    return outitem


class lmdbDatasetwmixup(tdata.Dataset):
  def __init__(self, db_path, transform=None, lk=11):
    # 11 -1 -> 10
    self.lk = lk-1
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

    lam = np.random.randint(1, self.lk)
    lam /= self.lk
    bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.size(), 1 - lam)
    lam = (bbx2 - bbx1) * (bby2 - bby1) / (img.size()[1] * img.size()[2])
    # print(lam, imgpath, rimgpath)
    ## WO LD
    rimg[:, bbx1:bbx2, bby1:bby2] = img[:, bbx1:bbx2, bby1:bby2]

    if rlabel == 1:
      lam = 1.0 - lam

    ### R2 WO LE
    # if lam > 0.5:
    #   lam = 1.0
    # else:
    #   lam = 0.0
    ##
    return img, label, imgpath, rimg, lam, self.uuid[strtoken[5]], self.uuid[strrtoken[5]]


class lmdbDatasetwmixupwlimit(tdata.Dataset):
  def __init__(self, db_path, strinclude, transform=None):

    self.env = None
    self.txn = None
    self.strinclude = strinclude
    self.transform = transform
    #Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260.db.path
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
    # Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260.db.path
    # REPLAY-ATTACK MSU-MFSD
    for index, strline in enumerate(strlines):
      strline = strline.strip()
      # ignore  -> limit src
      # if "CASIA-MFSD" in strline: continue
      # if "OULU-NPU" in strline: continue
      # only contaion -> cross modal
      if self.strinclude in strline:
        if "MSU-MFSD" in strline or "REPLAY-ATTACK" in strline:
          if ".mov" in strline:
            strtokens = strline.split(".mov")
            strkey = "{}.mov".format(strtokens[0])
          else:
            strtokens = strline.split(".mp4")
            strkey = "{}.mp4".format(strtokens[0])
          self.setkeys(strkey, index)
          # print(strline, strkey, index)
        elif "OULU-NPU" in strline or "CASIA-MFSD" in strline:
          strkey = os.path.dirname(strline)
          self.setkeys(strkey, index)
    # self.videokeys = list(self.videopath.keys())
    self.videokeys = []
    self.videokeys.extend(30*list(self.videopath.keys()))

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
    #/home/user/work_db/PublicDB/REPLAY-ATTACK/test_jpg/attack/hand/attack_highdef_client104_session01_highdef_video_controlled.mov_34.jpg
    #/home/user/work_db/PublicDB/REPLAY-ATTACK/test_jpg/attack/hand/attack_mobile_client019_session01_mobile_video_adverse.mov_38.jpg
    # tmpkey1 = 0 # adverse 0
    # tmpkey2 = 0
    # if "controlled" in imgpath:
    #   tmpkey1 = 1
    # if "controlled" in rimgpath:
    #   tmpkey2 = 1

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
    # print(lam, imgpath, rimgpath)
    rimg[:, bbx1:bbx2, bby1:bby2] = img[:, bbx1:bbx2, bby1:bby2]

    if rlabel == 1:
      lam = 1.0 - lam
    return img, label, imgpath, rimg, lam, self.uuid[strtoken[5]], self.uuid[strrtoken[5]]
    # return img, label, imgpath, rimg, lam, tmpkey1, tmpkey2

#############################################################################################################################################
#############################################################################################################################################
class lmdbDatasetwmixupwlimit2(tdata.Dataset):
  def __init__(self, db_path, strinclude, transform=None):

    self.env = None
    self.txn = None
    self.strinclude = strinclude
    self.transform = transform
    #Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260.db.path
    self.db_path = db_path
    self.db_path_img = "{}{}".format(self.db_path, ".path")
    self.videopath = {}
    self.setsinglevideo()
    self.mydatum = mydatum_pb2.myDatum()
    self._init_db()
    self.uuid = {}
    self.len = len(self.videopath.keys())

  def _init_db(self):
    self.env = lmdb.open(self.db_path,
                         readonly=True, lock=False,
                         readahead=False, meminit=False)
    self.txn = self.env.begin()

  def setsinglevideo(self):
    fpath = open(self.db_path_img, "r")
    strlines = fpath.readlines()
    # Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260.db.path
    # REPLAY-ATTACK MSU-MFSD
    numcnt = 0
    for index, strline in enumerate(strlines):
      strline = strline.strip()
      # ignore  -> limit src
      # if "CASIA-MFSD" in strline: continue
      # only contaion -> cross modal
      if self.strinclude in strline:
        self.videopath[numcnt] = index
        numcnt = numcnt + 1

  def __len__(self):
    return len(self.videopath.keys())

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

  def getpairitem(self, label):
    randidx = np.random.randint(0, self.len)
    rindex = self.videopath[randidx]
    rimg, rlabel, rimgpath = self.getitem(rindex)

    while(label == rlabel):
      randidx = np.random.randint(0, self.len)
      rindex = self.videopath[randidx]

      rimg, rlabel, rimgpath = self.getitem(rindex)

    return rimg, rlabel, rimgpath

  def __getitem__(self, reindex):
    index = self.videopath[reindex]

    img, label, imgpath = self.getitem(index)
    rimg, rlabel, rimgpath = self.getpairitem(label)
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
    # print(lam, imgpath, rimgpath)
    rimg[:, bbx1:bbx2, bby1:bby2] = img[:, bbx1:bbx2, bby1:bby2]

    if rlabel == 1:
      lam = 1.0 - lam
    return img, label, imgpath, rimg, lam, self.uuid[strtoken[5]], self.uuid[strrtoken[5]]
#############################################################################################################################################
#############################################################################################################################################
if __name__ == '__main__':

  transforms = T.Compose([T.RandomHorizontalFlip(),
                          #T.RandomRotation(180),
                          T.RandomCrop((256, 256)),
                          T.ToTensor()])  # 0 to 1

  mydataset = lmdbDatasettest("/home/user/data2/work_db/v4C3/Test_Protocal_4C3_MSU_1by1_260x260.db.sort", 8, transforms)
  # mydataset = lmdbDatasetwmixup("/home/user/work_db/v4C3/Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260.db",
  #                         transforms)
  trainloader = DataLoader(mydataset, batch_size=100, shuffle=True, num_workers=0, pin_memory=False)
  # for imgp1, label, imgpath, rimg, lam, uid1, uid2 in trainloader:
  for imgmap in trainloader:
    imgs = imgmap["imgs"]
  #   for iii, fff in enumerate(label):
  #     print (fff, imgpath[iii])
  #   break
  #   for sid in range(imgs.shape[1]):
  #     print(sid, imgs[0,sid,:,:,:].shape)
  #     print(sid, imgs[0, sid, :, :, :].flatten()[0:10])
  #     print(sid, imgs[1, sid, :, :, :].flatten()[0:10])
      # print(subi, images_x[0, subi, :, :, :].flatten()[0:10])
      # print(subi, images_x[1, subi, :, :, :].flatten()[0:10])
      # to_pil_image(imgs[0,sid,:,:,:]).show()
    break
    # rand_idx = torch.randperm(rimg.shape[0])
    # vimgs = torch.cat((imgp1, rimg[rand_idx[0:rimg.shape[0]],]), dim=0)
    # vlabels = torch.cat((label, lam[rand_idx[0:rimg.shape[0]]]), dim=0)
    # print (label)
    # print(lam[rand_idx[0:rimg.shape[0]]])
    # print (vlabels)
    # vlabels = vlabels * 10
    # vlabels = vlabels.type(torch.LongTensor)
    # print(vlabels)
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


    # liveidx = torch.where(label == 1)
    # fakeidx = torch.where(label == 0)
    # print(label)
    # print(liveidx)
    # print(label[0], imgpath)
    # to_pil_image(imgp1[0]).show()
    # print(imgp1[0].shape)
    # break
    # #
    # # print (imgp1.shape, imgp2.shape, imgp3.shape, label.shape, imgpath[0])
    # for iii, fff in enumerate(label):
    #   print (fff, imgpath[iii], label[iii])
    #   # to_pil_image(imgp1[iii]).show()
    #   # break
    # # break