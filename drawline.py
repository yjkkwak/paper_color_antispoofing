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


def drawece():
  CB91_Blue = '#2CBDFE'
  CB91_Green = '#47DBCD'
  CB91_Pink = '#F3A0F2'
  CB91_Purple = '#9D2EC5'
  CB91_Violet = '#661D98'
  CB91_Amber = '#F5B14C'

  date = [2, 5, 7, 8, 9, 10, 11, 12, 15, 20, 30]#, 40, 50, 60, 70, 100]
  xticks = [2, 5, 7, 8, 9, 10, 11, 12, 15, 20, 30]#, 40, 50, 60, 70, 100]
  temperature = [16.94, 15.45, 14.7, 14.52, 14.71, 13.4, 15.37, 15.64, 18.99, 16.57, 19.92]#, 18.8, 18.8, 18.8, 16.94, 20.1]
  price = [90.48, 88.91, 91.58, 92.9, 92.43, 92.66, 92.04, 89.93, 87.15, 89.86, 85.47]#, 86.17, 88.85, 89.94, 88.02,86.27]


  fig, ax1 = plt.subplots(1, 2, figsize=(17, 3))
  #ax2 = ax1.twinx()

  threshold = 15.45
  ax1[0].axhline(threshold, color=CB91_Pink, linestyle=':', lw=2, label='Baseline (ResNet18)')

  threshold = 89.84
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

if __name__ == '__main__':
  drawece()