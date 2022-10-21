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

from networks import getbaseresnet18, getmetricresnet18, getbasesiameseresnet18wgrl, getbaseresnet18wgrl
from lmdbdataset import lmdbDataset, lmdbDatasettest, lmdbDatasettestAllimages
from utils import AverageMeter, accuracy, getbasenamewoext, genfarfrreer, gentprwonlylive
import os
# import shortuuid
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

def testmodel(epoch, model, testdbpath, strckptpath, lk):
  """
  """
  print ("test db {} based on {}".format(testdbpath, strckptpath))
  averagemetermap = {}
  averagemetermap["acc_am"] = AverageMeter()

  strscorebasepath = os.path.join(strckptpath, getbasenamewoext(os.path.basename(testdbpath)))
  if os.path.exists(strscorebasepath) == False:
    os.makedirs(strscorebasepath)
  strscorepath = "{}/{:02d}.score".format(strscorebasepath, epoch)
  the_file = open(strscorepath, "w")
  transforms = T.Compose([T.CenterCrop((256, 256)),
                          T.ToTensor()])  # 0 to 1

  frames_total = 8
  #testdataset = lmdbDatasettest(testdbpath, transforms)
  testdataset = lmdbDatasettest(testdbpath, frames_total, transforms)

  # print(testdataset)
  testloader = DataLoader(testdataset, batch_size=200, shuffle=False, num_workers=0, pin_memory=True)

  model.eval()

  writelist = []
  regrsteps = torch.linspace(0, 1.0, steps=lk).cuda()
  probsm = nn.Softmax(dim=1)
  for index, outitem in enumerate(testloader):
    images = outitem["imgs"]
    labels = outitem["label"]
    imgpath = outitem["imgpath"]
    images, labels = images.cuda(), labels.cuda()
    # b f c w h
    map_score = 0
    for subi in range(images.shape[1]):
      logit, dislogit = model(images[:, subi, :, :, :])
      # logit = model(images[:, subi, :, :, :])
      # expectprob = probsm(logit)
      # map_score += expectprob.detach().cpu().numpy()[:, 1]
      prob = probsm(logit)
      expectprob = torch.sum(regrsteps * prob, dim=1)
      map_score += expectprob.detach().cpu().numpy()
    map_score = map_score / images.shape[1]

    tmplogit = torch.zeros(images.size(0), 2).cuda()
    tmplogit[:, 1] = torch.from_numpy(map_score)
    tmplogit[:, 0] = 1.0 - tmplogit[:, 1]

    acc = accuracy(tmplogit, labels)
    averagemetermap["acc_am"].update(acc[0].item())
    for idx, imgpathitem in enumerate(imgpath):
      writelist.append(
        "{:.5f} {:.5f} {:.5f}\n".format(labels[idx].detach().cpu().numpy(), float(tmplogit[idx][0]), float(tmplogit[idx][1])))

  for witem in writelist:
    the_file.write(witem)
  the_file.close()

  hter = ssan_performances_val(strscorepath)
  return hter

def testmodelallimgs(epoch, model, testdbpath, strckptpath, lk):
  """
  """
  print ("test db {} based on {}".format(testdbpath, strckptpath))
  averagemetermap = {}
  averagemetermap["acc_am"] = AverageMeter()

  strscorebasepath = os.path.join(strckptpath, getbasenamewoext(os.path.basename(testdbpath)))
  if os.path.exists(strscorebasepath) == False:
    os.makedirs(strscorebasepath)
  strscorepath = "{}/{:02d}.score".format(strscorebasepath, epoch)
  the_file = open(strscorepath, "w")
  transforms = T.Compose([T.CenterCrop((256, 256)),
                          T.ToTensor()])  # 0 to 1

  frames_total = 8
  #testdataset = lmdbDatasettest(testdbpath, transforms)
  testdataset = lmdbDatasettestAllimages(testdbpath, transforms)

  # print(testdataset)
  testloader = DataLoader(testdataset, batch_size=200, shuffle=False, num_workers=0, pin_memory=True)

  model.eval()

  writelist = []
  regrsteps = torch.linspace(0, 1.0, steps=lk).cuda()
  probsm = nn.Softmax(dim=1)
  for index, outitem in enumerate(testloader):
    images = outitem["imgs"]
    labels = outitem["label"]
    imgpath = outitem["imgpath"]
    images, labels = images.cuda(), labels.cuda()

    logit, dislogit = model(images)
    # logit = model(images[:, subi, :, :, :])
    # expectprob = probsm(logit)
    # map_score += expectprob.detach().cpu().numpy()[:, 1]
    prob = probsm(logit)
    expectprob = torch.sum(regrsteps * prob, dim=1)
    map_score = expectprob.detach().cpu().numpy()

    tmplogit = torch.zeros(images.size(0), 2).cuda()
    tmplogit[:, 1] = torch.from_numpy(map_score)
    tmplogit[:, 0] = 1.0 - tmplogit[:, 1]

    acc = accuracy(tmplogit, labels)
    averagemetermap["acc_am"].update(acc[0].item())
    for idx, imgpathitem in enumerate(imgpath):
      writelist.append(
        "{:.5f} {:.5f} {:.5f}\n".format(labels[idx].detach().cpu().numpy(), float(tmplogit[idx][0]), float(tmplogit[idx][1])))

  for witem in writelist:
    the_file.write(witem)
  the_file.close()

  hter = ssan_performances_val(strscorepath)
  return hter



def testsiamesemodel(epoch, model, testdbpath, strckptpath, frames_total=8):
  """
  """
  print ("test db {} based on {}".format(testdbpath, strckptpath))
  averagemetermap = {}
  averagemetermap["acc_am"] = AverageMeter()

  strscorebasepath = os.path.join(strckptpath, getbasenamewoext(os.path.basename(testdbpath)), str(frames_total))
  if os.path.exists(strscorebasepath) == False:
    os.makedirs(strscorebasepath)
  strscorepathreg = "{}/{:02d}.score.reg".format(strscorebasepath, epoch)
  strscorepathcls = "{}/{:02d}.score.cls".format(strscorebasepath, epoch)
  the_file_reg = open(strscorepathreg, "w")
  the_file_cls = open(strscorepathcls, "w")
  transforms = T.Compose([T.CenterCrop((256, 256)),
                          T.ToTensor()])  # 0 to 1

  testdataset = lmdbDatasettest(testdbpath, frames_total, transforms)

  # print(testdataset)
  testloader = DataLoader(testdataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

  model.eval()

  writelist_reg = []
  writelist_cls = []
  regrsteps = torch.linspace(0, 1.0, steps=21).cuda()
  probsm = nn.Softmax(dim=1)
  for index, outitem in enumerate(testloader):
    images = outitem["imgs"]
    labels = outitem["label"]
    imgpath = outitem["imgpath"]
    images, labels = images.cuda(), labels.cuda()
    # b f c w h
    map_score_cls = 0
    map_score_reg = 0
    for subi in range(images.shape[1]):
      logit_reg, logit_cls, dislogit_reg, dislogit_cls = model(images[:, subi, :, :, :], images[:, subi, :, :, :])

      # cls
      expectprob = probsm(logit_cls)
      map_score_cls += expectprob.detach().cpu().numpy()[:, 1]
      # reg
      prob = probsm(logit_reg)
      expectprob = torch.sum(regrsteps * prob, dim=1)
      map_score_reg += expectprob.detach().cpu().numpy()
    map_score_cls = map_score_cls / images.shape[1]
    map_score_reg = map_score_reg / images.shape[1]

    tmplogit_cls = torch.zeros(images.size(0), 2).cuda()
    tmplogit_cls[:, 1] = torch.from_numpy(map_score_cls)
    tmplogit_cls[:, 0] = 1.0 - tmplogit_cls[:, 1]

    tmplogit_reg = torch.zeros(images.size(0), 2).cuda()
    tmplogit_reg[:, 1] = torch.from_numpy(map_score_reg)
    tmplogit_reg[:, 0] = 1.0 - tmplogit_reg[:, 1]

    acc = accuracy(tmplogit_reg, labels)
    averagemetermap["acc_am"].update(acc[0].item())
    for idx, imgpathitem in enumerate(imgpath):
      writelist_cls.append(
        "{:.5f} {:.5f} {:.5f}\n".format(labels[idx].detach().cpu().numpy(), float(tmplogit_cls[idx][0]),
                                        float(tmplogit_cls[idx][1])))
      writelist_reg.append(
        "{:.5f} {:.5f} {:.5f}\n".format(labels[idx].detach().cpu().numpy(), float(tmplogit_reg[idx][0]),
                                        float(tmplogit_reg[idx][1])))


  for witem in writelist_cls:
    the_file_cls.write(witem)
  the_file_cls.close()

  for witem in writelist_reg:
    the_file_reg.write(witem)
  the_file_reg.close()

  htercls = ssan_performances_val(strscorepathcls)#4
  hter = ssan_performances_val(strscorepathreg)#5

  if htercls < hter:
    hter = htercls

  return hter


#########################################################
#########################################################

def testsiamesewckpt(model, strckptfilepath, testdbpath, strckptpath, frames_total=8):
  """
  """
  print ("test db {} based on {}".format(testdbpath, strckptpath))
  averagemetermap = {}
  averagemetermap["acc_am"] = AverageMeter()


  if model is None:
    model = getbasesiameseresnet18wgrl()
    model = model.cuda()

  print (model)

  checkpoint = torch.load(strckptfilepath)
  model.load_state_dict(checkpoint['model_state_dict'])
  epoch = checkpoint['epoch']


  strscorebasepath = os.path.join(strckptpath, getbasenamewoext(os.path.basename(testdbpath)), str(frames_total))
  if os.path.exists(strscorebasepath) == False:
    os.makedirs(strscorebasepath)
  strscorepathreg = "{}/{:02d}.score.reg".format(strscorebasepath, epoch)
  strscorepathcls = "{}/{:02d}.score.cls".format(strscorebasepath, epoch)
  strprobpathcls = "{}/{:02d}.prob.cls".format(strscorebasepath, epoch)
  the_file_reg = open(strscorepathreg, "w")
  the_file_cls = open(strscorepathcls, "w")
  the_file_prob_cls = open(strprobpathcls, "w")
  transforms = T.Compose([T.CenterCrop((256, 256)),
                          T.ToTensor()])  # 0 to 1

  testdataset = lmdbDatasettest(testdbpath, frames_total, transforms)

  # print(testdataset)
  testloader = DataLoader(testdataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

  model.eval()

  writelist_reg = []
  writelist_cls = []
  regrsteps = torch.linspace(0, 1.0, steps=11).cuda()
  probsm = nn.Softmax(dim=1)
  for index, outitem in enumerate(testloader):
    images = outitem["imgs"]
    labels = outitem["label"]
    imgpath = outitem["imgpath"]
    images, labels = images.cuda(), labels.cuda()
    # b f c w h
    map_score_cls = 0
    map_score_reg = 0
    for subi in range(images.shape[1]):
      logit_reg, logit_cls, dislogit_reg, dislogit_cls = model(images[:, subi, :, :, :], images[:, subi, :, :, :])

      # cls
      expectprob = probsm(logit_cls)
      map_score_cls += expectprob.detach().cpu().numpy()[:, 1]
      # reg
      prob = probsm(logit_reg)
      # print(prob.shape)

      #b , 1 => out b -> 11
      for bi in range(images.shape[0]):
        # print(prob[bi].shape)
        # print(prob[bi].detach().cpu().numpy()[0])
        #labels[idx].detach().cpu().numpy()
        the_file_prob_cls.write("{} {:.5f} ".format(imgpath[bi], labels[bi].detach().cpu().numpy()))
        for pi in range(11):
          the_file_prob_cls.write("{:.5f} ".format(prob[bi].detach().cpu().numpy()[pi]))
        the_file_prob_cls.write("\n")

      the_file_prob_cls.write("\n")
      the_file_prob_cls.write("\n")

      expectprob = torch.sum(regrsteps * prob, dim=1)
      map_score_reg += expectprob.detach().cpu().numpy()
    map_score_cls = map_score_cls / images.shape[1]
    map_score_reg = map_score_reg / images.shape[1]

    tmplogit_cls = torch.zeros(images.size(0), 2).cuda()
    tmplogit_cls[:, 1] = torch.from_numpy(map_score_cls)
    tmplogit_cls[:, 0] = 1.0 - tmplogit_cls[:, 1]

    tmplogit_reg = torch.zeros(images.size(0), 2).cuda()
    tmplogit_reg[:, 1] = torch.from_numpy(map_score_reg)
    tmplogit_reg[:, 0] = 1.0 - tmplogit_reg[:, 1]

    acc = accuracy(tmplogit_reg, labels)
    averagemetermap["acc_am"].update(acc[0].item())
    for idx, imgpathitem in enumerate(imgpath):
      writelist_cls.append(
        "{:.5f} {:.5f} {:.5f}\n".format(labels[idx].detach().cpu().numpy(), float(tmplogit_cls[idx][0]),
                                        float(tmplogit_cls[idx][1])))
      writelist_reg.append(
        "{:.5f} {:.5f} {:.5f}\n".format(labels[idx].detach().cpu().numpy(), float(tmplogit_reg[idx][0]),
                                        float(tmplogit_reg[idx][1])))


  for witem in writelist_cls:
    the_file_cls.write(witem)
  the_file_cls.close()

  for witem in writelist_reg:
    the_file_reg.write(witem)
  the_file_reg.close()
  the_file_prob_cls.close()

  htercls = ssan_performances_val(strscorepathcls)#4
  hter = ssan_performances_val(strscorepathreg)#5

  if htercls < hter:
    hter = htercls

  return hter

def testwckpt(model, strckptfilepath, testdbpath, strckptpath, lk):
  """
  """
  # print ("test db {} based on {}".format(testdbpath, strckptpath))
  averagemetermap = {}
  averagemetermap["acc_am"] = AverageMeter()

  if model is None:
    # model = getmetricresnet18()
    # model = getresnet18()
    #model = getbaseresnet18()
    model = getbaseresnet18wgrl(numclasses=lk)
    model = model.cuda()

  checkpoint = torch.load(strckptfilepath)
  model.load_state_dict(checkpoint['model_state_dict'])
  epoch = checkpoint['epoch']

  strscorebasepath = os.path.join(strckptpath, getbasenamewoext(os.path.basename(testdbpath)))
  if os.path.exists(strscorebasepath) == False:
    os.makedirs(strscorebasepath)
  strscorepath = "{}/{:02d}.score".format(strscorebasepath, epoch)
  # if os.path.exists(strscorepath):
  #   print ("exists {}".format(strscorepath))
  #   return
  the_file = open(strscorepath, "w")

  transforms = T.Compose([T.CenterCrop((256, 256)),
                          T.ToTensor()])# 0 to 1

  frames_total = 8
  # testdataset = lmdbDatasettest(testdbpath, transforms)
  testdataset = lmdbDatasettest(testdbpath, frames_total, transforms)

  # print(testdataset)
  testloader = DataLoader(testdataset, batch_size=20, shuffle=False, num_workers=0, pin_memory=True)

  model.eval()
  writelist = []
  regrsteps = torch.linspace(0, 1.0, steps=lk).cuda()
  probsm = nn.Softmax(dim=1)
  for index, outitem in enumerate(testloader):
    images = outitem["imgs"]
    labels = outitem["label"]
    imgpath = outitem["imgpath"]
    images, labels = images.cuda(), labels.cuda()
    # b f c w h
    map_score = 0
    for subi in range(images.shape[1]):
      logit, dislogit = model(images[:, subi, :, :, :])
      # logit = model(images[:, subi, :, :, :])
      # expectprob = probsm(logit)
      # map_score += expectprob.detach().cpu().numpy()[:, 1]
      prob = probsm(logit)
      expectprob = torch.sum(regrsteps * prob, dim=1)
      map_score += expectprob.detach().cpu().numpy()

    map_score = map_score / images.shape[1]

    tmplogit = torch.zeros(images.size(0), 2).cuda()
    tmplogit[:, 1] = torch.from_numpy(map_score)
    tmplogit[:, 0] = 1.0 - tmplogit[:, 1]

    acc = accuracy(tmplogit, labels)
    averagemetermap["acc_am"].update(acc[0].item())
    for idx, imgpathitem in enumerate(imgpath):
      writelist.append(
        "{:.5f} {:.5f} {:.5f} {}\n".format(labels[idx].detach().cpu().numpy(), float(tmplogit[idx][0]),
                                        float(tmplogit[idx][1]), imgpathitem))

  for witem in writelist:
    the_file.write(witem)
  the_file.close()

  hter = ssan_performances_val(strscorepath)



  # Replay_attack, Video vs real
  with open(strscorepath, "r") as the_file:
    testlists = the_file.readlines()
    the_file.close()

  # printed photo
  # strscorepath_pp = "{}_printedphoto".format(strscorepath)
  # with open(strscorepath_pp, "w") as the_file_pp:
  #   for imgpath in testlists:
  #     if "/real/" in imgpath:
  #       the_file_pp.write("{}".format(imgpath))
  #     if "/attack_print" in imgpath:
  #       the_file_pp.write("{}".format(imgpath))
  #   the_file.close()
  #
  # hter = ssan_performances_val(strscorepath_pp)

  strscorepath_C1 = "{}_C1".format(strscorepath)
  with open(strscorepath_C1, "w") as the_file_pp:
    for imgpath in testlists:
      if "/1/" in imgpath or "/3/" in imgpath or "/4/" in imgpath or "/5/" in imgpath:
        the_file_pp.write("{}".format(imgpath))

    the_file.close()
  hter = ssan_performances_val(strscorepath_C1)

  strscorepath_C2 = "{}_C2".format(strscorepath)
  with open(strscorepath_C2, "w") as the_file_pp:
    for imgpath in testlists:
      if "/2/" in imgpath or "/6/" in imgpath or "/7/" in imgpath or "/8/" in imgpath:
        the_file_pp.write("{}".format(imgpath))

    the_file.close()
  hter = ssan_performances_val(strscorepath_C2)
#90.31 91.83 97.84
#96.02 91.82 90.38 96.80 91.01 98.37
  strscorepath_C3 = "{}_C3".format(strscorepath)
  with open(strscorepath_C3, "w") as the_file_pp:
    for imgpath in testlists:
      if "/HR_" in imgpath:
        the_file_pp.write("{}".format(imgpath))

    the_file.close()
  hter = ssan_performances_val(strscorepath_C3)
  return 0

  #real/2_2_45_1   phone session

  # strscorepath_P1 = "{}_P1".format(strscorepath)
  # with open(strscorepath_P1, "w") as the_file_pp:
  #   for imgpath in testlists:
  #     strtoken = imgpath.split()
  #     strtoken = strtoken[3].split("/")
  #     Ptype = strtoken[8].split("_")[0]
  #     if Ptype == "1":
  #       the_file_pp.write("{}".format(imgpath))
  #   the_file_pp.close()
  # hter = ssan_performances_val(strscorepath_P1)
  #
  # strscorepath_P1 = "{}_P2".format(strscorepath)
  # with open(strscorepath_P1, "w") as the_file_pp:
  #   for imgpath in testlists:
  #     strtoken = imgpath.split()
  #     strtoken = strtoken[3].split("/")
  #     Ptype = strtoken[8].split("_")[0]
  #     if Ptype == "2":
  #       the_file_pp.write("{}".format(imgpath))
  #   the_file_pp.close()
  # hter = ssan_performances_val(strscorepath_P1)
  #
  # strscorepath_P1 = "{}_P3".format(strscorepath)
  # with open(strscorepath_P1, "w") as the_file_pp:
  #   for imgpath in testlists:
  #     strtoken = imgpath.split()
  #     strtoken = strtoken[3].split("/")
  #     Ptype = strtoken[8].split("_")[0]
  #     if Ptype == "3":
  #       the_file_pp.write("{}".format(imgpath))
  #   the_file_pp.close()
  # hter = ssan_performances_val(strscorepath_P1)
  #
  # strscorepath_P1 = "{}_P4".format(strscorepath)
  # with open(strscorepath_P1, "w") as the_file_pp:
  #   for imgpath in testlists:
  #     strtoken = imgpath.split()
  #     strtoken = strtoken[3].split("/")
  #     Ptype = strtoken[8].split("_")[0]
  #     if Ptype == "4":
  #       the_file_pp.write("{}".format(imgpath))
  #   the_file_pp.close()
  # hter = ssan_performances_val(strscorepath_P1)
  #
  # strscorepath_P1 = "{}_P5".format(strscorepath)
  # with open(strscorepath_P1, "w") as the_file_pp:
  #   for imgpath in testlists:
  #     strtoken = imgpath.split()
  #     strtoken = strtoken[3].split("/")
  #     Ptype = strtoken[8].split("_")[0]
  #     if Ptype == "5":
  #       the_file_pp.write("{}".format(imgpath))
  #   the_file_pp.close()
  # hter = ssan_performances_val(strscorepath_P1)
  #
  # strscorepath_P1 = "{}_P6".format(strscorepath)
  # with open(strscorepath_P1, "w") as the_file_pp:
  #   for imgpath in testlists:
  #     strtoken = imgpath.split()
  #     strtoken = strtoken[3].split("/")
  #     Ptype = strtoken[8].split("_")[0]
  #     if Ptype == "6":
  #       the_file_pp.write("{}".format(imgpath))
  #   the_file_pp.close()
  # hter = ssan_performances_val(strscorepath_P1)

  # real/2_2_45_1   phone session

  strscorepath_S1 = "{}_S1".format(strscorepath)
  with open(strscorepath_S1, "w") as the_file_pp:
    for imgpath in testlists:
      strtoken = imgpath.split()
      strtoken = strtoken[3].split("/")
      Ptype = strtoken[8].split("_")[1]
      if Ptype == "1":
        the_file_pp.write("{}".format(imgpath))
    the_file_pp.close()
  hter = ssan_performances_val(strscorepath_S1)

  strscorepath_S1 = "{}_S2".format(strscorepath)
  with open(strscorepath_S1, "w") as the_file_pp:
    for imgpath in testlists:
      strtoken = imgpath.split()
      strtoken = strtoken[3].split("/")
      Ptype = strtoken[8].split("_")[1]
      if Ptype == "2":
        the_file_pp.write("{}".format(imgpath))
    the_file_pp.close()
  hter = ssan_performances_val(strscorepath_S1)

  strscorepath_S1 = "{}_S3".format(strscorepath)
  with open(strscorepath_S1, "w") as the_file_pp:
    for imgpath in testlists:
      strtoken = imgpath.split()
      strtoken = strtoken[3].split("/")
      Ptype = strtoken[8].split("_")[1]
      if Ptype == "3":
        the_file_pp.write("{}".format(imgpath))
    the_file_pp.close()
  hter = ssan_performances_val(strscorepath_S1)

  return hter


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

  # basepath = "/home/user/model_2022/v4C3/Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260_220530_YzxoCrrET3wSVPY7CkWjE9_bsize128_optadam_lr0.0001_gamma_0.9_epochs_40_meta_clsloss_resnet18_adam/"
  # testdbpath = "/home/user/work_db/v4C3/Test_Protocal_4C3_MSU_1by1_260x260.db"
  # ckptlist = glob.glob("{}/**/*.ckpt".format(basepath), recursive=True)
  # for ckptpath in ckptlist:
  #   ffff = getbasenamewoext(ckptpath)
  #   print(ffff)
  #   testwckpt(None,
  #             ckptpath,
  #             testdbpath,
  #             basepath)
  #
  # basepath = "/home/user/model_2022/v4C3/Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260_220530_J4p8ZsKf79woquYJSS2Jo5_bsize128_optadam_lr0.0001_gamma_0.9_epochs_40_meta_clsloss_resnet18_adam/"
  # testdbpath = "/home/user/work_db/v4C3/Test_Protocal_4C3_CASIA_1by1_260x260.db"
  # ckptlist = glob.glob("{}/**/*.ckpt".format(basepath), recursive=True)
  # for ckptpath in ckptlist:
  #   ffff = getbasenamewoext(ckptpath)
  #   print(ffff)
  #   testwckpt(None,
  #             ckptpath,
  #             testdbpath,
  #             basepath)

#  def testsiamesewckpt(model, strckptfilepath, testdbpath, strckptpath, frames_total=8):
  basesavescorepath = "./testmodel"
  strckpt = "/home/user/model_2022/model_rebuttal_RX_lamda1.0/Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260_221001_fnsAcRP6FzUUR4wup2sU2Y_bsize16_optadam_lr0.00018_gamma_0.99_epochs_300_meta_mseregloss_resnet18_adam_rebuttal_RX_samelamda_lamda_1.0/epoch_63.ckpt"
  testdbpath = "/home/user/work_db/v4C3/Test_Protocal_4C3_CASIA_1by1_260x260.db.sort"
  # testsiamesewckpt(None, strckpt, testdbpath, basesavescorepath, 8)

#
  # basesavescorepath = "./testmodel"
  # strckpt = "/home/user/model_2022/v4C3_sample/Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260_220609_g6EFVegwp5sEGnM9CXoqNe_bsize16_optadam_lr0.0001_gamma_0.99_epochs_1000_meta_msegrlloss_resnet18_adam_full/epoch_121.ckpt"
  # testdbpath = "/home/user/work_db/v4C3/Test_Protocal_4C3_REPLAY_1by1_260x260.db.sort"
  # testwckpt(None, strckpt, testdbpath, basesavescorepath, 11)

  basesavescorepath = "./testmodel"
  #strckpt = "/home/user/model_2022/v4C3_sample_K/Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260_220715_g7bNVmgBGbBgquj2fhw96v_bsize16_optadam_lr0.0001_gamma_0.99_epochs_1000_meta_mseregloss_resnet18_adam_baseline_10_seed_20200102_k_10/epoch_62.ckpt"
  #strckpt = "/home/user/data2/model_rebuttal_R1/Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260_220903_6LDsBLvTuamV9kbETYdibu_bsize16_optadam_lr0.00016_gamma_0.99_epochs_1000_meta_mseregloss_resnet18_adam_rebuttal_R1_lamda_0.75//epoch_143.ckpt"
  strckpt = "/home/user/data2/model_rebuttal_RX_reproduce/Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260_221003_TLhvDSzmrMGPyzsYvx4Tvb_bsize16_optadam_lr0.00018_gamma_0.99_epochs_150_meta_mseregloss_resnet18_adam_rebuttal_reproduce_lamda_1.0/epoch_63.ckpt"
  #
  testdbpath = "/home/user/data2/work_db/v4C3/Test_Protocal_4C3_CASIA_1by1_260x260.db.sort"
  testwckpt(None, strckpt, testdbpath, basesavescorepath, 11)

  # basesavescorepath = "./testmodel"
  # strckpt = "/home/user/model_2022/v4C3_sample/Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260_220610_PAFXdBobwA9vmiucF3Z5kf_bsize16_optadam_lr0.0001_gamma_0.99_epochs_1000_meta_msegrlloss_resnet18_adam_half/epoch_82.ckpt"
  # testdbpath = "/home/user/work_db/v4C3/Test_Protocal_4C3_OULU_1by1_260x260.db.sort"
  # testwckpt(None, strckpt, testdbpath, basesavescorepath, 11)

