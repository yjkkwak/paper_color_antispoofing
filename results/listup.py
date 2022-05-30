import glob
import numpy as np
import torch
import time
import os
from scipy import interpolate
from utils import readscore
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

def getresults(strpatch, strtestdb):
  strpath = "/home/user/model_2022/v220419_01"
  strpattern = "*CelebA_SiW_LDRGB_LD3007_{}*".format(strpatch)
  print (strpath, strpattern)
  mypath = Path(strpath).glob(strpattern)
  for dirpath in mypath:
    strtestpath = "{}/{}/*.eval".format(dirpath, strtestdb)
    evallist = glob.glob(strtestpath)
    print (strtestpath)
    evallist.sort()
    for evalpath in evallist:
      print (os.path.basename(evalpath))

def printlist():
  strpatchs = ["4by3_244x324"]#, "4by3_244x324"]#1by1_260x260
  strtestdb = "Dev_v220419_01_OULUNPU_"
  # strtestdb = "Test_v220419_01_LDRGB_1by1_260x260"
  for strpatch in strpatchs:
    getresults(strpatch, "{}{}".format(strtestdb, strpatch))

if __name__ == '__main__':
  print ("eeff")
  printlist()