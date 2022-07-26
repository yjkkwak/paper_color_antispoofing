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
import matplotlib.patches as mpatches
import torch.nn.functional as F

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

def calc_bins(preds, labels_oneh):
  # Assign each prediction to a bin
  num_bins = 10
  bins = np.linspace(0.1, 1, num_bins)
  binned = np.digitize(preds, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

  return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds, labels_oneh):
  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels_oneh)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)

  return ECE, MCE


def draw_reliability_graph(preds, labels_oneh, plttitle="Baseline - MSU-MFSE"):
  ECE, MCE = get_metrics(preds, labels_oneh)
  bins, _, bin_accs, _, _ = calc_bins(preds, labels_oneh)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.gca()

  # x/y limits
  ax.set_xlim(0, 1.05)
  ax.set_ylim(0, 1)

  # x/y labels
  plt.title(plttitle, fontsize=18)
  plt.xlabel('Confidence')#, fontsize=15)
  plt.ylabel('Accuracy')#, fontsize=15)
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)

  # Create grid
  ax.set_axisbelow(True)
  ax.grid(color='gray', linestyle='dashed')
  ax.xaxis.label.set_size(16)
  ax.yaxis.label.set_size(16)

  # Error bars
  plt.bar(bins, bins, width=0.1, alpha=0.5, edgecolor='black', color='r', hatch='\\')

  # Draw bars and identity line
  plt.bar(bins, bin_accs, width=0.1, alpha=1.0, edgecolor='black', color='b')
  plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=3)

  # Equally spaced axes
  plt.gca().set_aspect('equal', adjustable='box')

  # ECE and MCE legend
  ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE * 100))
  #MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE * 100))
  #plt.legend(handles=[ECE_patch, MCE_patch])
  plt.legend(handles=[ECE_patch], fontsize=20)

  # plt.show()

  plt.savefig('{}.pdf'.format(plttitle.replace(" ", "").replace("/", "")), bbox_inches='tight')


def drawece():
  # Casia Test
  # strspathreg = "/home/user/model_2022/v4C3_sample_ablation/Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260_220612_oCeY6HHPBmUTu6pvvFDaN9_bsize16_optadam_lr0.0001_gamma_0.99_epochs_1000_meta_clsloss_resnet18_adam_baseline_again/Test_Protocal_4C3_CASIA_1by1_260x260.db/15.score"
  strspathreg = "/home/user/model_2022/v4C3_sample/Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260_220609_j3YFzMKU2H8vHShDoZMrQi_bsize16_optadam_lr0.0001_gamma_0.99_epochs_1000_meta_msegrlloss_resnet18_adam_full/Test_Protocal_4C3_CASIA_1by1_260x260.db/230.score"
  # strspathreg = "./testmodel/Test_Protocal_4C3_CASIA_1by1_260x260.db/8/39.score.reg"

  # # OULU Test
  # strspathreg = "/home/user/model_2022/v4C3_sample_ablation/Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260_220612_TmaCF6tPupXWH4nTq3XtSp_bsize16_optadam_lr0.0001_gamma_0.99_epochs_1000_meta_clsloss_resnet18_adam_baseline_again/Test_Protocal_4C3_OULU_1by1_260x260.db/57.score"
  # strspathreg = "/home/user/model_2022/v4C3_sample/Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260_220609_GA35fDPN2owCK7q5EuiQaM_bsize16_optadam_lr0.0001_gamma_0.99_epochs_1000_meta_msegrlloss_resnet18_adam_full/Test_Protocal_4C3_OULU_1by1_260x260.db/04.score"
  # strspathreg = "/home/user/model_2022/v4C3_sample_siamese/Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260_220609_97kg9j5bFTjHtJLQo8EnTr_bsize16_optadam_lr0.0001_gamma_0.99_epochs_1000_meta_msegrlloss_resnet18_adam_full/Test_Protocal_4C3_OULU_1by1_260x260.db/17.score.reg"

  # Replay Test
  # strspathreg = "/home/user/model_2022/v4C3_sample_ablation/Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260_220612_cSk2cL69CyY6bffWho8mXG_bsize16_optadam_lr0.0001_gamma_0.99_epochs_1000_meta_clsloss_resnet18_adam_baseline_again/Test_Protocal_4C3_REPLAY_1by1_260x260.db/23.score"
  # strspathreg = "/home/user/model_2022/v4C3_sample/Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260_220609_g6EFVegwp5sEGnM9CXoqNe_bsize16_optadam_lr0.0001_gamma_0.99_epochs_1000_meta_msegrlloss_resnet18_adam_full/Test_Protocal_4C3_REPLAY_1by1_260x260.db/121.score"
  # strspathreg = "/home/user/model_2022/v4C3_sample_siamese/Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260_220611_ZpC2qGwyNLL88xpw37BJ8t_bsize16_optadam_lr0.0001_gamma_0.99_epochs_1000_meta_msegrlloss_resnet18_adam_full_siamese_seed_20220408/Test_Protocal_4C3_REPLAY_1by1_260x260.db/246.score.reg"

  # M Test
  # strspathreg = "/home/user/model_2022/v4C3_sample_ablation/Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260_220612_cSk2cL69CyY6bffWho8mXG_bsize16_optadam_lr0.0001_gamma_0.99_epochs_1000_meta_clsloss_resnet18_adam_baseline_again/Test_Protocal_4C3_REPLAY_1by1_260x260.db/23.score"
  # strspathreg = "/home/user/model_2022/v4C3_sample/Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260_220609_bjPF65Wfip7wSvjTC2ZvwH_bsize16_optadam_lr0.0001_gamma_0.99_epochs_1000_meta_msegrlloss_resnet18_adam_full/Test_Protocal_4C3_MSU_1by1_260x260.db/52.score"
  # strspathreg = "/home/user/model_2022/v4C3_sample_siamese/Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260_220611_ZpC2qGwyNLL88xpw37BJ8t_bsize16_optadam_lr0.0001_gamma_0.99_epochs_1000_meta_msegrlloss_resnet18_adam_full_siamese_seed_20220408/Test_Protocal_4C3_REPLAY_1by1_260x260.db/246.score.reg"

  npscorereg = readscore(strspathreg)
  label_test = npscorereg[:, 0]
  # label_test = torch.from_numpy(label_test).type(torch.int64)
  # label_ont = F.one_hot(label_test, num_classes=2)
  preds =npscorereg[:, 2]

  from netcal.presentation import ReliabilityDiagram
  from netcal.metrics import ECE

  n_bins = 10

  plttitle = "Baseline (ResNet18)"
  diagram = ReliabilityDiagram(n_bins, title_suffix=plttitle)
  diagram.plot(preds, label_test).savefig('{}.pdf'.format(plttitle.replace(" ", "").replace("/", "")), bbox_inches='tight') # visualize miscalibration of uncalibrated

  ece = ECE(n_bins)
  aa = ece.measure(preds, label_test)
  print (aa)

#plt.savefig('{}.pdf'.format(plttitle.replace(" ", "").replace("/", "")), bbox_inches='tight')
  # fig, axes = plt.subplots(1, squeeze=True, figsize=(7, 6))
  # ax = axes[1]
  # print (allaxes)


  return


  #
  # ECE, MCE = get_metrics(preds, label_ont)
  # bins, _, bin_accs, _, _ = calc_bins(preds)
  #draw_reliability_graph(preds, label_ont, "Baseline - MSU-MFSD")
  # draw_reliability_graph(preds, label_ont, "PDLE - MSU-MFSD")
  draw_reliability_graph(preds, label_ont, "PDLE w/ aux - MSU-MFSD")
  # print (ECE, MCE)


  #
  # strspathcls = "./testmodel/Test_Protocal_4C3_CASIA_1by1_260x260.db/1/39.score.cls"
  # npscorecls = readscore(strspathcls)
  # label_test = npscorecls[:, 0]
  # F.one_hot(label_test, num_classes=2)



if __name__ == '__main__':
  drawece()