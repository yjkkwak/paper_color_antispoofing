import glob

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import DataLoader
from eval.performance import ssan_performances_val

from networks import getbaseresnet18, getmetricresnet18
from lmdbdataset import lmdbDataset
from utils import AverageMeter, accuracy, getbasenamewoext, genfarfrreer, gentprwonlylive
import os
import shortuuid
from datetime import datetime


#
# def load_ckpt(model):
#   print("Load ckpt from {}".format(args.ckptpath))
#   checkpoint = torch.load(args.ckptpath)
#   model.load_state_dict(checkpoint['model_state_dict'])
#   # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#   epoch = checkpoint['epoch']
#   print ("Loaded epoch {}".format(epoch))

def testmetricmodel(epoch, model, testdbpath, strckptpath):
  """
    """
  # print ("test db {} based on {}".format(testdbpath, strckptpath))
  averagemetermap = {}
  averagemetermap["acc_am"] = AverageMeter()

  strscorebasepath = os.path.join(strckptpath, getbasenamewoext(os.path.basename(testdbpath)))
  if os.path.exists(strscorebasepath) == False:
    os.makedirs(strscorebasepath)
  strscorepath = "{}/{:02d}.score".format(strscorebasepath, epoch)
  the_file = open(strscorepath, "w")
  if "260x260" in testdbpath:
    transforms = T.Compose([T.CenterCrop((256, 256)),
                            T.ToTensor()])  # 0 to 1
  elif "244x324" in testdbpath:
    transforms = T.Compose([T.CenterCrop((320, 240)),
                            T.ToTensor()])  # 0 to 1
  testdataset = lmdbDataset(testdbpath, transforms)

  # print(testdataset)
  testloader = DataLoader(testdataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

  model.eval()
  probsm = nn.Softmax(dim=1)

  writelist = []
  for index, (images, labels, imgpath) in enumerate(testloader):
    images, labels = images.cuda(), labels.cuda()
    logit = model(images)
    prob = probsm(logit)
    acc = accuracy(logit, labels)
    averagemetermap["acc_am"].update(acc[0].item())
    for idx, imgpathitem in enumerate(imgpath):
      writelist.append(
        "{:.5f} {:.5f} {:.5f}\n".format(labels[idx].detach().cpu().numpy(), float(prob[idx][0]), float(prob[idx][1])))

  for witem in writelist:
    the_file.write(witem)
  the_file.close()

  ssan_performances_val(strscorepath)

def testmodel(epoch, model, testdbpath, strckptpath):
  """
  """
  # print ("test db {} based on {}".format(testdbpath, strckptpath))
  averagemetermap = {}
  averagemetermap["acc_am"] = AverageMeter()

  strscorebasepath = os.path.join(strckptpath, getbasenamewoext(os.path.basename(testdbpath)))
  if os.path.exists(strscorebasepath) == False:
    os.makedirs(strscorebasepath)
  strscorepath = "{}/{:02d}.score".format(strscorebasepath, epoch)
  the_file = open(strscorepath, "w")
  transforms = T.Compose([T.CenterCrop((256, 256)),
                          T.ToTensor()])  # 0 to 1

  testdataset = lmdbDataset(testdbpath, transforms)

  # print(testdataset)
  testloader = DataLoader(testdataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

  model.eval()
  probsm = nn.Softmax(dim=1)

  writelist = []
  for index, (images, labels, imgpath) in enumerate(testloader):
    images, labels = images.cuda(), labels.cuda()
    logit = model(images)
    prob = probsm(logit)
    acc = accuracy(logit, labels)
    averagemetermap["acc_am"].update(acc[0].item())
    for idx, imgpathitem in enumerate(imgpath):
      writelist.append(
        "{:.5f} {:.5f} {:.5f}\n".format(labels[idx].detach().cpu().numpy(), float(prob[idx][0]), float(prob[idx][1])))

  for witem in writelist:
    the_file.write(witem)
  the_file.close()

  ssan_performances_val(strscorepath)


def testwckpt(model, strckptfilepath, testdbpath, strckptpath):
  """
  """
  # print ("test db {} based on {}".format(testdbpath, strckptpath))
  averagemetermap = {}
  averagemetermap["acc_am"] = AverageMeter()

  if model is None:
    # model = getmetricresnet18()
    # model = getresnet18()
    model = getbaseresnet18()
    model = model.cuda()

  checkpoint = torch.load(strckptfilepath)
  model.load_state_dict(checkpoint['model_state_dict'])
  epoch = checkpoint['epoch']

  strscorebasepath = os.path.join(strckptpath, getbasenamewoext(os.path.basename(testdbpath)))
  if os.path.exists(strscorebasepath) == False:
    os.makedirs(strscorebasepath)
  strscorepath = "{}/{:02d}.score".format(strscorebasepath, epoch)
  if os.path.exists(strscorepath):
    print ("exists {}".format(strscorepath))
    return
  the_file = open(strscorepath, "w")

  transforms = T.Compose([T.CenterCrop((256, 256)),
                          T.ToTensor()])# 0 to 1

  testdataset = lmdbDataset(testdbpath, transforms)

  # print(testdataset)
  testloader = DataLoader(testdataset, batch_size=330, shuffle=False, num_workers=0, pin_memory=True)

  model.eval()
  probsm = nn.Softmax(dim=1)
  writelist = []
  for index, (images, labels, imgpath) in enumerate(testloader):
    images, labels = images.cuda(), labels.cuda()
    logit = model(images)
    prob = probsm(logit)
    acc = accuracy(logit, labels)
    averagemetermap["acc_am"].update(acc[0].item())
    for idx, imgpathitem in enumerate(imgpath):
      writelist.append("{:.5f} {:.5f} {:.5f}\n".format(labels[idx].detach().cpu().numpy(), float(prob[idx][0]), float(prob[idx][1])))

  for witem in writelist:
    the_file.write(witem)
  the_file.close()

  ssan_performances_val(strscorepath)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='anti-spoofing testing')
  # parser.add_argument('--ckptpath', type=str,
  #                     default='/home/user/model_2022', help='ckpt path')
  # parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
  parser.add_argument('--GPU', type=int, default=2, help='specify which gpu to use')
  # parser.add_argument('--works', type=int, default=4, help='works')
  args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU)  # Set the GPU to use

  print(args)

  # basepath = "/home/user/model_2022/v4C3/Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260_220530_QDVJ3zhtjCMDBRf8SaUR4Y_bsize128_optadam_lr0.0001_gamma_0.9_epochs_40_meta_clsloss_resnet18_adam/"
  # testdbpath = "/home/user/work_db/v4C3/Test_Protocal_4C3_REPLAY_1by1_260x260.db"
  # ckptlist = glob.glob("{}/**/*.ckpt".format(basepath), recursive=True)
  # for ckptpath in ckptlist:
  #   ffff = getbasenamewoext(ckptpath)
  #   print (ffff)
  #   testwckpt(None,
  #             ckptpath,
  #             testdbpath,
  #             basepath)
  #
  #
  #
  # basepath = "/home/user/model_2022/v4C3/Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260_220530_KQdwNVgBih3Ws8epSM4PTt_bsize128_optadam_lr0.0001_gamma_0.9_epochs_40_meta_clsloss_resnet18_adam/"
  # testdbpath = "/home/user/work_db/v4C3/Test_Protocal_4C3_OULU_1by1_260x260.db"
  # ckptlist = glob.glob("{}/**/*.ckpt".format(basepath), recursive=True)
  # for ckptpath in ckptlist:
  #   ffff = getbasenamewoext(ckptpath)
  #   print (ffff)
  #   testwckpt(None,
  #             ckptpath,
  #             testdbpath,
  #             basepath)

  basepath = "/home/user/model_2022/v4C3/Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260_220530_YzxoCrrET3wSVPY7CkWjE9_bsize128_optadam_lr0.0001_gamma_0.9_epochs_40_meta_clsloss_resnet18_adam/"
  testdbpath = "/home/user/work_db/v4C3/Test_Protocal_4C3_MSU_1by1_260x260.db"
  ckptlist = glob.glob("{}/**/*.ckpt".format(basepath), recursive=True)
  for ckptpath in ckptlist:
    ffff = getbasenamewoext(ckptpath)
    print(ffff)
    testwckpt(None,
              ckptpath,
              testdbpath,
              basepath)

  basepath = "/home/user/model_2022/v4C3/Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260_220530_J4p8ZsKf79woquYJSS2Jo5_bsize128_optadam_lr0.0001_gamma_0.9_epochs_40_meta_clsloss_resnet18_adam/"
  testdbpath = "/home/user/work_db/v4C3/Test_Protocal_4C3_CASIA_1by1_260x260.db"
  ckptlist = glob.glob("{}/**/*.ckpt".format(basepath), recursive=True)
  for ckptpath in ckptlist:
    ffff = getbasenamewoext(ckptpath)
    print(ffff)
    testwckpt(None,
              ckptpath,
              testdbpath,
              basepath)