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



def readscore(scorefile):
  the_file = open(scorefile, "r")
  strlines = the_file.readlines()
  scorelist = []
  for strline in strlines:
    strtokens = strline.split()
    scorelist.append([float(strtokens[0]), float(strtokens[1]), float(strtokens[2])])
  the_file.close()
  npscore = np.array(scorelist)
  return npscore


def drawece():
  strspathreg = "./testmodel/Test_Protocal_4C3_CASIA_1by1_260x260.db/8/39.score.reg"
  npscorereg = readscore(strspathreg)
  label_test = npscorereg[:, 0]
  logreg_y, logreg_x = calibration_curve(label_test, npscorereg[:, 2], n_bins=10)

  strspathcls = "./testmodel/Test_Protocal_4C3_CASIA_1by1_260x260.db/1/39.score.cls"
  npscorecls = readscore(strspathcls)
  label_test = npscorecls[:, 0]
  logcls_y, logcls_x = calibration_curve(label_test, npscorecls[:, 2], n_bins=10)



  # calibration curves
  fig, ax = plt.subplots()
  plt.plot(logreg_x, logreg_y, marker='o', linewidth=1, label='logreg')
  plt.plot(logcls_x, logcls_y, marker='o', linewidth=1, label='logcls')

  # reference line, legends, and axis labels
  line = mlines.Line2D([0, 1], [0, 1], color='black')
  transform = ax.transAxes
  line.set_transform(transform)
  ax.add_line(line)
  fig.suptitle('Calibration plot for Titanic data')
  ax.set_xlabel('Predicted probability')
  ax.set_ylabel('True probability in each bin')
  plt.legend()
  plt.show()  # it's not rendered correctly on GitHub, check blog post for actual pic


def drawhistgraph():
  strspathcls = "./testmodel/Test_Protocal_4C3_CASIA_1by1_260x260.db/8/39.score.reg"
  npscorecls = readscore(strspathcls)


  fakelbcls = np.where(npscorecls[:, 0] == 0.0)[0]
  livelbcls = np.where(npscorecls[:, 0] == 1.0)[0]


  num_bin = 50
  bin_lims = np.linspace(0, 1, num_bin + 1)
  bin_centers = 0.5 * (bin_lims[:-1] + bin_lims[1:])
  bin_widths = bin_lims[1:] - bin_lims[:-1]

  ##computing the histograms
  hist1, _ = np.histogram(npscorecls[fakelbcls, 2], bins=bin_lims)
  hist2, _ = np.histogram(npscorecls[livelbcls, 2], bins=bin_lims)

  ##normalizing
  hist1b = hist1 / np.max(hist1)
  hist2b = hist2 / np.max(hist2)

  plt.bar(bin_centers, hist1b, width=bin_widths, align='center')
  plt.bar(bin_centers, hist2b, width=bin_widths, align='center', alpha=0.5)

  plt.show()

  return



if __name__ == '__main__':
  # drawhistgraph()
  drawece()