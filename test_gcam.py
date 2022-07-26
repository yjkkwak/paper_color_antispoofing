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
from torchvision.transforms.functional import to_pil_image
from networks import getbaseresnet18, getmetricresnet18, getbasesiameseresnet18wgrl, getbaseresnet18wgrl
from lmdbdataset import lmdbDataset, lmdbDatasettest, lmdbDatasettestAllimages
from utils import AverageMeter, accuracy, getbasenamewoext, genfarfrreer, gentprwonlylive
import os
from datetime import datetime

import cv2

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

def testmodelallimgswckpt(model, strckptfilepath, testdbpath, strckptpath, lk):
  """
  """
  print ("test db {} based on {}".format(testdbpath, strckptpath))
  averagemetermap = {}
  averagemetermap["acc_am"] = AverageMeter()


  if model is None:
    model = getbaseresnet18wgrl(lk)
    model = model.cuda()

  checkpoint = torch.load(strckptfilepath)
  model.load_state_dict(checkpoint['model_state_dict'])
  epoch = checkpoint['epoch']


  strscorebasepath = os.path.join(strckptpath, getbasenamewoext(os.path.basename(testdbpath)))
  if os.path.exists(strscorebasepath) == False:
    os.makedirs(strscorebasepath)
  strscorepath = "{}/{:02d}.score".format(strscorebasepath, epoch)
  the_file = open(strscorepath, "w")
  transforms = T.Compose([T.CenterCrop((256, 256)),
                          T.ToTensor()])  # 0 to 1

  frames_total = 8
  testdataset = lmdbDatasettestAllimages(testdbpath, transforms)

  # print(testdataset)
  testloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

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

def returnCAM(feature_conv, weight_softmax, class_idx):
  # generate the class activation maps upsample to 256x256
  size_upsample = (256, 256)
  print (feature_conv.shape)
  bz, nc, h, w = feature_conv.shape

  output_cam = []
  for idx in class_idx:
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
  return output_cam

def testmodelallimgswckpt_gcam(model, strckptfilepath, testdbpath, strckptpath, lk):
  """
  """
  print ("test db {} based on {}".format(testdbpath, strckptpath))
  averagemetermap = {}
  averagemetermap["acc_am"] = AverageMeter()


  if model is None:
    model = getbaseresnet18wgrl(lk)
    model = model.cuda()

  checkpoint = torch.load(strckptfilepath)
  model.load_state_dict(checkpoint['model_state_dict'])
  epoch = checkpoint['epoch']


  strscorebasepath = os.path.join(strckptpath, getbasenamewoext(os.path.basename(testdbpath)))
  if os.path.exists(strscorebasepath) == False:
    os.makedirs(strscorebasepath)
  strscorepath = "{}/{:02d}.score".format(strscorebasepath, epoch)
  the_file = open(strscorepath, "w")
  transforms = T.Compose([T.CenterCrop((256, 256)),
                          T.ToTensor()])  # 0 to 1

  frames_total = 8
  testdataset = lmdbDatasettestAllimages(testdbpath, transforms)

  # print(testdataset)
  testloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

  model.eval()

  classes = {}#{0: 'cat', 1: 'dog'}
  for ff in range(0, lk):
    classes[ff] = "{}".format(ff)

  print (classes)

  # hook the feature extractor
  features_blobs = []
  # print (model)
  # print (model.named_parameters())
  # for name, p in model.named_parameters():
  #   print (name)
  # params = list(model.parameters())
  # weight_softmax = np.squeeze(params[-4].data.cpu().numpy())
  # print (weight_softmax.shape)
  # print (model._modules)
  # print(model._modules.get('embedder')._modules.get('bottleneck_layer_fc'))


  def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

  model._modules.get('embedder')._modules.get('layer4').register_forward_hook(hook_feature)

  print (features_blobs)


  params = list(model.parameters())
  weight_softmax = np.squeeze(params[-4].data.cpu().numpy())

  probsm = nn.Softmax(dim=1)
  for index, outitem in enumerate(testloader):
    images = outitem["imgs"]
    labels = outitem["label"]
    imgpath = outitem["imgpath"]
    # print (imgpath)
    # if "/home/user/work_db/PublicDB/MSU-MFSD/test_jpg/attack/attack_client001_laptop_SD_printed_photo_scene01.mov_32.jpg" not in imgpath[0]:
    # if "/home/user/work_db/PublicDB/MSU-MFSD/test_jpg/attack/attack_client001_laptop_SD_iphone_video_scene01.mov_103.jpg" not in imgpath[0]:
    if "attack" in imgpath[0]:
      continue


    to_pil_image(images[0]).save("./img_real2.png")

    images, labels = images.cuda(), labels.cuda()

    logit, dislogit = model(images)

    h_x = probsm(logit)
    h_x = h_x.data.squeeze()
    # print(h_x)
    probs, idx = h_x.sort(0, True)
    # print (probs)
    # print (idx)

    # output: the prediction
    for i in range(0, lk):
      line = '{:.8f} -> {}'.format(probs[i], classes[idx[i].item()])
      print(line)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])

    # render the CAM and output
    img = cv2.imread("img_real2.png")
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('cam_real_{}.jpg'.format(lk), result)

    break


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='anti-spoofing testing')
  # parser.add_argument('--ckptpath', type=str,
  #                     default='/home/user/model_2022', help='ckpt path')
  # parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
  parser.add_argument('--GPU', type=int, default=0, help='specify which gpu to use')
  # parser.add_argument('--works', type=int, default=4, help='works')
  args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU)  # Set the GPU to use


  basesavescorepath = "./testmodel"

  testdbpath = "/home/user/work_db/v4C3/Test_Protocal_4C3_MSU_1by1_260x260.db.sort"
  # strckpt = "/home/user/model_2022/v4C3_sample_K/Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260_220725_npbzUdpRQBqyns6gNcYiMw_bsize16_optadam_lr0.00016_gamma_0.99_epochs_1000_meta_mseregloss_resnet18_adam_baseline_6_again_allimgtest_seed_20220406_k_6/epoch_10.ckpt"
  # testmodelallimgswckpt_gcam(None, strckpt, testdbpath, basesavescorepath, 6)
  #
  # strckpt = "/home/user/model_2022/v4C3_sample_K/Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260_220725_Eymf7QzSBbtetV9aeP4bUf_bsize16_optadam_lr0.00016_gamma_0.99_epochs_1000_meta_mseregloss_resnet18_adam_baseline_11_again_allimgtest_seed_20220406_k_11/epoch_09.ckpt"
  # testmodelallimgswckpt_gcam(None, strckpt, testdbpath, basesavescorepath, 11)
  #
  # strckpt = "/home/user/model_2022/v4C3_sample_K/Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260_220725_VWbTmUnwMz2BTAcLfBfoqK_bsize16_optadam_lr0.00016_gamma_0.99_epochs_1000_meta_mseregloss_resnet18_adam_baseline_21_again_allimgtest_seed_20220406_k_21/epoch_73.ckpt"
  # testmodelallimgswckpt_gcam(None, strckpt, testdbpath, basesavescorepath, 21)
  #
  # strckpt = "/home/user/model_2022/v4C3_sample_K/Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260_220725_G3wTq2DsjhCA6WHeHibAUJ_bsize16_optadam_lr0.00016_gamma_0.99_epochs_1000_meta_mseregloss_resnet18_adam_baseline_31_again_allimgtest_seed_20220406_k_31/epoch_42.ckpt"
  # testmodelallimgswckpt_gcam(None, strckpt, testdbpath, basesavescorepath, 31)
  #
  # strckpt = "/home/user/model_2022/v4C3_sample_K/Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260_220725_JgMBJBGWgeyt8AUkdPmYiU_bsize16_optadam_lr0.00016_gamma_0.99_epochs_1000_meta_mseregloss_resnet18_adam_baseline_41_again_allimgtest_seed_20220406_k_41/epoch_131.ckpt"
  # testmodelallimgswckpt_gcam(None, strckpt, testdbpath, basesavescorepath, 41)

  strckpt = "/home/user/model_2022/v4C3_sample_ablation/Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260_220612_KYi5fUhoVpxgrrb4orHVCe_bsize16_optadam_lr0.0001_gamma_0.99_epochs_1000_meta_clsloss_resnet18_adam_baseline_again/epoch_01.ckpt"
  testmodelallimgswckpt_gcam(None, strckpt, testdbpath, basesavescorepath, 2)
  #

  #

  ##
