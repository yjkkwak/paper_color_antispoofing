import glob
import numpy as np
import torch
import time
import os
from scipy import interpolate
from utils import readscore
import matplotlib
import matplotlib.pyplot as plt


def drawplotwonlytpr(TPR1, THR1, TPR2, THR2, TPR3, THR3, strtestdb):
  fig = plt.figure()
  plt.subplot(1, 1, 1)# rows, cols, index
  plt.plot(THR1, TPR1, 'r', label='res-s')
  plt.plot(THR2, TPR2, 'r--', label='res-18')
  if THR3 is not None:
    plt.plot(THR3, TPR3, 'r:', label='res-sx2')
  plt.xlabel('THR')
  plt.ylabel('TPR')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.tight_layout()
  fig.suptitle(strtestdb)
  fig.subplots_adjust(top=0.88)

  plt.savefig("./{}.png".format(strtestdb))

def drawplotwvalue(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, strtestdb):
  fig = plt.figure()
  plt.subplot(1, 3, 1)# rows, cols, index
  plt.plot(THR1, FAR1, 'r', label='res-s')
  plt.plot(THR2, FAR2, 'r--', label='res-18')
  plt.xlabel('THR')
  plt.ylabel('FAR')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.subplot(1, 3, 2)  # rows, cols, index
  plt.plot(THR1, FRR1, 'g', label='res-s')
  plt.plot(THR2, FRR2, 'g--', label='res-18')
  plt.xlabel('THR')
  plt.ylabel('FRR')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.subplot(1, 3, 3)  # rows, cols, index
  plt.plot(THR1, EER1, 'b', label='res-s')
  plt.plot(THR2, EER2, 'b--', label='res-18')
  plt.xlabel('THR')
  plt.ylabel('EER')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.tight_layout()
  fig.suptitle(strtestdb)
  fig.subplots_adjust(top=0.88)

  plt.savefig("./{}.png".format(strtestdb))


def drawplotwvalue3(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, strtestdb):
  fig = plt.figure()
  plt.subplot(1, 3, 1)# rows, cols, index
  plt.plot(THR1, FAR1, 'r', label='res-s')
  plt.plot(THR2, FAR2, 'r--', label='res-18')
  plt.plot(THR3, FAR3, 'r:', label='res-sx2')
  plt.xlabel('THR')
  plt.ylabel('FAR')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.subplot(1, 3, 2)  # rows, cols, index
  plt.plot(THR1, FRR1, 'g', label='res-s')
  plt.plot(THR2, FRR2, 'g--', label='res-18')
  plt.plot(THR3, FRR3, 'g:', label='res-sx2')
  plt.xlabel('THR')
  plt.ylabel('FRR')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.subplot(1, 3, 3)  # rows, cols, index
  plt.plot(THR1, EER1, 'b', label='res-s')
  plt.plot(THR2, EER2, 'b--', label='res-18')
  plt.plot(THR3, EER3, 'b:', label='res-sx2')
  plt.xlabel('THR')
  plt.ylabel('EER')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.tight_layout()
  fig.suptitle(strtestdb)
  fig.subplots_adjust(top=0.88)

  plt.savefig("./{}.png".format(strtestdb))


def drawplotwvalue4(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, FAR4, FRR4, TPR4, EER4, THR4, strtestdb):
  fig = plt.figure(figsize=(10, 8))

  plt.subplot(1, 3, 1)# rows, cols, index
  plt.plot(THR1, FAR1, 'r', label='res-s w/o oulu')
  plt.plot(THR2, FAR2, 'b', label='res-18 w/o oulu')
  plt.plot(THR3, FAR3, 'r--', label='res-s w/ oulu')
  plt.plot(THR4, FAR4, 'b--', label='res-18 w/ oulu')
  plt.xlabel('THR')
  plt.ylabel('FAR')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.subplot(1, 3, 2)  # rows, cols, index
  plt.plot(THR1, FRR1, 'r', label='res-s w/o oulu')
  plt.plot(THR2, FRR2, 'b', label='res-18 w/o oulu')
  plt.plot(THR3, FRR3, 'r--', label='res-s w/ oulu')
  plt.plot(THR4, FRR4, 'b--', label='res-18 w/ oulu')
  plt.xlabel('THR')
  plt.ylabel('FRR')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.subplot(1, 3, 3)  # rows, cols, index
  plt.plot(THR1, EER1, 'r', label='res-s w/o oulu')
  plt.plot(THR2, EER2, 'b', label='res-18 w/o oulu')
  plt.plot(THR3, EER3, 'r--', label='res-s w/ oulu')
  plt.plot(THR4, EER4, 'b--', label='res-18 w/ oulu')
  plt.xlabel('THR')
  plt.ylabel('EER')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.tight_layout()
  fig.suptitle(strtestdb)
  fig.subplots_adjust(top=0.88)

  plt.savefig("./{}.png".format(strtestdb))

def drawplot4(strmodelpathforamt1, strmodelpathforamt2, strmodelpathforamt3, strmodelpathforamt4, strtestordev, strtestdb):
  FAR1, FRR1, TPR1, EER1, THR1 = getfarfrreer(
    strmodelpathforamt1.format(
      strtestordev, strtestdb))
  FAR2, FRR2, TPR2, EER2, THR2 = getfarfrreer(
    strmodelpathforamt2.format(
      strtestordev, strtestdb))

  FAR3, FRR3, TPR3, EER3, THR3 = getfarfrreer(
    strmodelpathforamt3.format(
      strtestordev, strtestdb))

  FAR4, FRR4, TPR4, EER4, THR4 = getfarfrreer(
    strmodelpathforamt4.format(
      strtestordev, strtestdb))

  drawplotwvalue4(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, FAR4, FRR4, TPR4, EER4, THR4, strtestdb)


def drawplot(strmodelpathforamt1, strmodelpathforamt2, strtestordev, strtestdb):
  FAR1, FRR1, TPR1, EER1, THR1 = getfarfrreer(
    strmodelpathforamt1.format(
      strtestordev, strtestdb))
  FAR2, FRR2, TPR2, EER2, THR2 = getfarfrreer(
    strmodelpathforamt2.format(
      strtestordev, strtestdb))

  #ensemble score
  # if "OULU" in strtestdb:
  #   FAR3, FRR3, TPR3, EER3, THR3 = getfarfrreer("./../ensemble/Dev_v220419_01_OULUNPU_1by1_260x260Dev_v220419_01_OULUNPU_4by3_244x324.txt")
  #   drawplotwvalue3(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, strtestdb)
  # elif "CelebA" in strtestdb:
  #   FAR3, FRR3, TPR3, EER3, THR3 = getfarfrreer("./../ensemble/Test_v220419_01_CelebA_1by1_260x260Test_v220419_01_CelebA_4by3_244x324.txt")
  #   drawplotwvalue3(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, strtestdb)
  # elif "LDRGB" in strtestdb:
  #   FAR3, FRR3, TPR3, EER3, THR3 = getfarfrreer("./../ensemble/Test_v220419_01_LDRGB_1by1_260x260Test_v220419_01_LDRGB_4by3_244x324.txt")
  #   drawplotwvalue3(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, strtestdb)
  # elif "LD3007" in strtestdb:
  #   FAR3, FRR3, TPR3, EER3, THR3 = getfarfrreer("./../ensemble/Test_v220419_01_LD3007_1by1_260x260Test_v220419_01_LD3007_4by3_244x324.txt")
  #   drawplotwvalue3(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, strtestdb)
  # elif "SiW" in strtestdb:
  #   FAR3, FRR3, TPR3, EER3, THR3 = getfarfrreer("./../ensemble/Test_v220419_01_SiW_1by1_260x260Test_v220419_01_SiW_4by3_244x324.txt")
  #   drawplotwvalue3(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, strtestdb)
  # else:
  #   drawplotwvalue(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, strtestdb)


  drawplotwvalue(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, strtestdb)

def getfarfrreer(scorefile):
  npscore = readscore(scorefile)
  lb = npscore[:, 2]
  livelb = np.where(npscore[:, 2] == 1.0)[0]
  fakelb = np.where(npscore[:, 2] == 0.0)[0]

  """"""
  TPR = []  # TPR = 1 - FRR
  FRR = []  # FRR = 1 - TPR
  FAR = []
  EER = []  # (FAR+FRR) / 2
  THR = []

  thre = np.arange(0.1, 1.0, 0.01)  # Generate an arithmetic list of model thresholds

  class_in = npscore[livelb]
  class_out = npscore[fakelb]

  tmpin = np.where(class_in[:, 1] > 0.5)[0]
  tmpout = np.where(class_out[:, 1] < 0.5)[0]
  acc = (len(tmpin) + len(tmpout)) / len(lb)

  # print (thre)
  for i in range(len(thre)):
    frr = np.sum(class_in[:, 1] < thre[i]) / len(livelb)
    far = np.sum(class_out[:, 1] > thre[i]) / len(class_out)
    tpr = 1.0 - frr
    eer = (frr + far) / 2.0
    FRR.append(frr)
    TPR.append(tpr)
    FAR.append(far)
    EER.append(eer)
    THR.append(thre[i])
    print ("ACC {:0.5} TPR {:0.5} / FAR {:0.5} / EER {:0.5} th {:0.5}".format(acc, tpr, far, eer, thre[i]))

  return FAR, FRR, TPR, EER, THR
  # interpolation.. soon
  # inter_tprwfar = interpolate.interp1d(FAR, TPR, fill_value='extrapolate')
  # inter_tprwthr = interpolate.interp1d(THR, TPR, fill_value='extrapolate')
  # inter_eerwthr = interpolate.interp1d(THR, EER, fill_value='extrapolate')


def gettpronly(scorefile):
  npscore = readscore(scorefile)
  lb = npscore[:, 2]
  livelb = np.where(npscore[:, 2] == 1.0)[0]
  fakelb = np.where(npscore[:, 2] == 0.0)[0]

  """"""
  TPR = []  # TPR = 1 - FRR
  THR = []

  thre = np.arange(0.1, 1.0, 0.01)  # Generate an arithmetic list of model thresholds

  class_in = npscore[livelb]

  tmpin = np.where(class_in[:, 1] > 0.5)[0]

  # print (thre)
  for i in range(len(thre)):
    frr = np.sum(class_in[:, 1] < thre[i]) / len(livelb)
    tpr = 1.0 - frr
    TPR.append(tpr)
    THR.append(thre[i])

  return TPR, THR

def rundrawplot2():
  strmodelpathforamt1 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_1by1_260x260_220502_3uKCX7S9pwbeSTzoTydcgV_lr0.005_gamma_0.92_epochs_80_meta_163264/{}_v220419_01_{}/78.score"
  strmodelpathforamt2 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_1by1_260x260_220502_Pn2ww7BGgZGmhJD5oeG2L6_lr0.005_gamma_0.92_epochs_80_meta_baselineres18/{}_v220419_01_{}/78.score"
  strmodelpathforamt3 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_1by1_260x260_220510_XWtdsCV5xfQ28a8PLyYYke_lr0.005_gamma_0.92_epochs_80_meta_163264/{}_v220419_01_{}/72.score"
  strmodelpathforamt4 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_1by1_260x260_220510_KHi4YQxF4Qx9S6XayeRBkx_lr0.005_gamma_0.92_epochs_80_meta_baselineres18/{}_v220419_01_{}/73.score"
  drawplot4(strmodelpathforamt1, strmodelpathforamt2, strmodelpathforamt3, strmodelpathforamt4, "Dev", "OULUNPU_1by1_260x260")

def rundrawplot():
  # strmodelpathforamt1 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_1by1_260x260_220502_3uKCX7S9pwbeSTzoTydcgV_lr0.005_gamma_0.92_epochs_80_meta_163264/{}_v220419_01_{}/78.score"
  # strmodelpathforamt2 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_1by1_260x260_220502_Pn2ww7BGgZGmhJD5oeG2L6_lr0.005_gamma_0.92_epochs_80_meta_baselineres18/{}_v220419_01_{}/78.score"

  strmodelpathforamt1 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_1by1_260x260_220510_XWtdsCV5xfQ28a8PLyYYke_lr0.005_gamma_0.92_epochs_80_meta_163264/{}_v220419_01_{}/72.score"
  strmodelpathforamt2 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_1by1_260x260_220510_KHi4YQxF4Qx9S6XayeRBkx_lr0.005_gamma_0.92_epochs_80_meta_baselineres18/{}_v220419_01_{}/73.score"

  TPR1, THR1 = gettpronly(
    strmodelpathforamt1.format(
      "Test", "Emotion_1by1_260x260"))

  TPR2, THR2 = gettpronly(
    strmodelpathforamt2.format(
      "Test", "Emotion_1by1_260x260"))

  # TPR3, THR3 = gettpronly("./../ensemble/Test_v220419_01_Emotion_1by1_260x260Test_v220419_01_Emotion_4by3_244x324.txt")
  # drawplotwonlytpr(TPR1, THR1, TPR2, THR2, TPR3, THR3, "Emotion_1by1_260x260")

  drawplotwonlytpr(TPR1, THR1, TPR2, THR2, None, None, "Emotion_1by1_260x260")

  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "SiW_1by1_260x260")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "LDRGB_1by1_260x260")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "LD3007_1by1_260x260")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "CelebA_1by1_260x260")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Dev", "OULUNPU_1by1_260x260")
  
  return

  strmodelpathforamt1 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_4by3_244x324_220504_eNeMv72oynyYhUikgY4mbv_lr0.001_gamma_0.92_epochs_80_meta_163264/{}_v220419_01_{}/69.score"
  strmodelpathforamt2 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_4by3_244x324_220505_edzapQSW8VwscSyfxJjcZr_lr0.005_gamma_0.92_epochs_80_meta_baselineres18/{}_v220419_01_{}/61.score"

  TPR1, THR1 = gettpronly(
    strmodelpathforamt1.format(
      "Test", "Emotion_4by3_244x324"))

  TPR2, THR2 = gettpronly(
    strmodelpathforamt2.format(
      "Test", "Emotion_4by3_244x324"))

  TPR3, THR3 = gettpronly("./../ensemble/Test_v220419_01_Emotion_1by1_260x260Test_v220419_01_Emotion_4by3_244x324.txt")
  drawplotwonlytpr(TPR1, THR1, TPR2, THR2, TPR3, THR3, "Emotion_4by3_244x324")


  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "SiW_4by3_244x324")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "LDRGB_4by3_244x324")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "LD3007_4by3_244x324")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "CelebA_4by3_244x324")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Dev", "OULUNPU_4by3_244x324")

if __name__ == '__main__':
  #rundrawplot()
  rundrawplot2()