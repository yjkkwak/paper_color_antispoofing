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

from networks import getbaseresnet18, getmetricresnet18, getbasesiameseresnet18wgrl
from lmdbdataset import lmdbDataset, lmdbDatasettest
from utils import AverageMeter, accuracy, getbasenamewoext, genfarfrreer, gentprwonlylive
import os
# import shortuuid
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn.calibration import calibration_curve
from datetime import datetime, timedelta

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'


def drawsubfig():
  date = [2, 3,4,5]#, 40, 50, 60, 70, 100]
  xticks = [2, 3,4,5]#, 40, 50, 60, 70, 100]
  temperature = [16.37, 12.25, 14.75, 14.0]
  price = [91.11, 91.10, 92.54, 92.99]


  fig, ax1 = plt.subplots(1, 2, figsize=(10, 3))
  #ax2 = ax1.twinx()

  threshold = 16.25
  ax1[0].axhline(threshold, color=CB91_Pink, linestyle=':', lw=2, label='Baseline (ResNet18)')

  threshold = 85.06
  ax1[1].axhline(threshold, color=CB91_Blue, linestyle=':', lw=2, label='Baseline (ResNet18)')

  ax1[0].plot(date, temperature, color=CB91_Pink, marker='o', linestyle='-', lw=2, label='Our approach with K')
  ax1[1].plot(date, price, color=CB91_Blue, marker='o', linestyle='-', lw=2, label='Our approach with K')

  ax1[0].set_xlabel("K")
  ax1[0].set_ylabel("HTER (%)", fontsize=20)
  ax1[0].set_xticks(xticks)
  ax1[0].legend(prop={'size': 12})
  # ax1[0].grid(True)
  # ax1[0].set_title("Impact of K in our method", fontsize=20)

  ax1[1].set_xlabel("K")
  ax1[1].set_ylabel("AUC (%)", fontsize=20)
  ax1[1].set_xticks(xticks)
  ax1[1].legend(prop={'size': 12})
  # ax1[1].grid(True)

  # ax1[1].set_title("Impact of K in our method", fontsize=20)

  #fig.suptitle("Impact of K in our method", fontsize=30)
  # fig.suptitle("BCFp", fontsize=20)


  #plt.show()
  plt.savefig("./fig_3_pdf.pdf",bbox_inches='tight')



def drawfig():
  date = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]#, 40, 50, 60, 70, 100]
  xticks = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]#, 40, 50, 60, 70, 100]
  #temperature = [16.94, 15.45, 14.7, 14.52, 14.71, 13.4, 15.37, 15.64, 18.99]#, 18.8, 18.8, 18.8, 16.94, 20.1]
  price = [91.11, 91.1, 92.54, 92.99, 95.59, 93.84, 95.5, 96.07, 97.6, 95.87, 95.29, 96.25, 94.72, 94.75, 93.07, 94.03]


  # fig, ax1 = plt.subplots(1, 2, figsize=(17, 3))
  #ax2 = ax1.twinx()

  # threshold = 15.45
  fig = plt.figure(figsize=(10, 4))
  threshold = 88.06
  # plt.axhline(threshold, color=CB91_Amber, linestyle=':', lw=2, label='Baseline (ResNet18)')
  plt.plot(date, price, color=CB91_Blue, marker='o', linestyle='-', lw=2, label='Our approach with K')


  plt.xlabel("K")
  plt.ylabel("AUC (%)", fontsize=20)
  plt.xticks(xticks)
  plt.legend(prop={'size': 12})
  # ax1[1].grid(True)

  # ax1[1].set_title("Impact of K in our method", fontsize=20)

  #fig.suptitle("Impact of K in our method", fontsize=30)
  # fig.suptitle("BCFp", fontsize=20)


  plt.show()
  # plt.savefig("./fig_3_pdf.pdf",bbox_inches='tight')

if __name__ == '__main__':
  drawfig()