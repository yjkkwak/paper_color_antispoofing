import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import DataLoader
from augs.cutmix import cutmix_data, mixup_criterion, cutmix_data_wocuda
from lmdbdataset import lmdbDatasetwmixup
import matplotlib.pyplot as plt
from networks import getbaseresnet18wgrl,getbaseresnet18
from utils import AverageMeter, accuracy, Timer, getbasenamewoext, Logger
import os
import shortuuid
from datetime import datetime
from test import testmodel
from shutil import copyfile
import glob
random_seed = 20220406# (ours)

torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
def trainmodel():
  transforms = T.Compose([T.RandomCrop((256, 256)),
                          T.RandomHorizontalFlip(),
                          T.ToTensor()])  # 0 to 1
  # lmdbpath = "/home/user/work_db/v4C3/Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260.db"

  lmdbpath = "/home/user/work_db/vOP/Train_OULU_Protocol_4_1_1by1_260x260.db.sort"
  batch_size = 128
  traindataset = lmdbDatasetwmixup(lmdbpath, transforms)

  trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=16,
                           pin_memory=True)

  successratio = []
  for index, (tmpimages, tmplabels, imgpath, rimg, rlab, tmpuid1, tmpuid2) in enumerate(trainloader):
    images = tmpimages
    labels = tmplabels
    #images, labels = images.cuda(), labels.cuda()
    cm_images, cm_mixlabel = cutmix_data_wocuda(images, labels)
    #print (images.shape, labels.shape, cm_mixlabel[0].shape, cm_mixlabel[1].shape)
    #print(cm_mixlabel[0], cm_mixlabel[1])
    issame = cm_mixlabel[0] == cm_mixlabel[1]
    issamecond = torch.where(issame == True, 1, 0)
    #print((batch_size - torch.sum(issamecond))/batch_size)
    ffff = (batch_size - torch.sum(issamecond))/batch_size
    # print (ffff.item())
    successratio.append(ffff.item())

  npfff = np.array(successratio)
  # print(npfff)
  print (np.average(npfff), np.std(npfff))
#lmdbpath = "/home/user/work_db/v4C3/Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260.db"
#4  0.237 0.277
#8  0.277 0.189
#16 0.299 0.129
#32 0.306 0.104
#64 0.320 0.081
#128 0.319 0.111

# lmdbpath = "/home/user/work_db/v4C3/Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260.db"
#8   0.271 0.18
#16  0.29  0.14
#32  0.307 0.10
#64  0.323 0.084
#128 0.334 0.071

#8   0.37  0.20
#16  0.40  0.15
#32  0.44  0.09
#64  0.45  0.12
#128 0.48  0.08

def drawerrorbar():
  CB91_Blue = '#2CBDFE'
  CB91_Green = '#47DBCD'
  CB91_Pink = '#F3A0F2'
  CB91_Purple = '#9D2EC5'
  CB91_Violet = '#661D98'
  CB91_Amber = '#F5B14C'

  # defining our function

  # x = [4, 8, 16, 32, 64, 128]
  # y = [0.237, 0.277, 0.299, 0.306, 0.320, 0.319]
  # ystd = [0.277, 0.189, 0.129, 0.104, 0.081, 0.111]

  x = [8, 16, 32, 64, 128]
  y = [0.277, 0.299, 0.306, 0.320, 0.319]
  y_error = [0.189, 0.129, 0.104, 0.081, 0.111]
  y_error = np.array(y_error) * 100
  y = np.array(y)*100
  y1 = [0.37, 0.40, 0.44, 0.45, 0.48]
  y_error1 = [0.2, 0.15, 0.09, 0.12, 0.08]
  y_error1 = np.array(y_error1) * 100
  y1 = np.array(y1) * 100

  # 8   0.37  0.20
  # 16  0.40  0.15
  # 32  0.44  0.09
  # 64  0.45  0.12
  # 128 0.48  0.08

  # defining our error
  xticks = x
  # plotting our function and
  # error bar
  threshold = 100.0
  plt.figure(figsize=(11, 5))
  plt.axhline(threshold, color=CB91_Blue, linestyle='-', lw=2, label='Ours (PDLE)')

  #plt.axhline(threshold, color='black', linestyle='-', lw=2, label='Ours (PDLE)')

  plt.plot(x, y1, color=CB91_Pink, linestyle='-', lw=2)
  plt.errorbar(x, y1, yerr=y_error1, color=CB91_Pink, ecolor=CB91_Pink, lw=10, alpha=0.7,
               capsize=15.0, capthick=2.0, elinewidth=5, fmt='X', label='CutMix (w/ OULU-NPU)')

  plt.plot(x, y, color=CB91_Green, linestyle='-', lw=2)
  plt.errorbar(x, y, yerr=y_error, color=CB91_Green, ecolor=CB91_Green, lw=10,
               capsize=15.0, capthick=2.0, elinewidth=5, fmt='D', alpha=0.7, label='CutMix (w/ O&M&I to C)')





  plt.scatter(x, y, s=50, marker="D", color=CB91_Green)
  plt.scatter(x, y1, s=70, marker="X", color=CB91_Pink)


  plt.yticks([0.0, 25, 50, 75, 100.0])
  plt.xticks(xticks)
  #
  plt.ylabel("Ratio(%)", fontsize=15)
  plt.xlabel("Batch Size", fontsize=15)
  plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95))#prop={'size': 15}
  # plt.show()
  plt.savefig("./s_fig_1_pdf.pdf", bbox_inches='tight')

if __name__ == '__main__':
  # trainmodel()
  drawerrorbar()