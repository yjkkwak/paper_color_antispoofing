import glob
import os
import numpy as np
import torch
import random
from itertools import combinations
random_seed = 20220321

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

datatypes = ["Train", "Test"]
datapaths = ["/home/user/work_db/PublicDB/CASIA-MFSD/",  # train_jpg / test_jpg  --> fd          --> real spoof
             "/home/user/work_db/PublicDB/MSU-MFSD/",  # train_jpg / test_jpg    --> fd          --> real attack
             "/home/user/work_db/PublicDB/OULU-NPU/",  # train_jpg / test_jpg / devel_jpg/       --> real spoof
             "/home/user/work_db/PublicDB/REPLAY-ATTACK/",  # train_jpg / test_jpg / devel_jpg   --> real attack
             ]
liveitems = ["/live/", "real"]
spoofitems = ["/spoof/", "attack"]
datakeys = {}

def getStatistics():
  for dbpaths in datapaths:
    for dbtypes in datatypes:
      print (dbpaths, dbtypes)
      jpgitems = glob.glob("{}/**/*.jpg.fd".format(os.path.join(dbpaths, dbtypes)), recursive=True)
      imgitmes = jpgitems
      liveimages = [item for item in imgitmes if any(liveitem in item for liveitem in liveitems)]
      spoofimges = [item for item in imgitmes if any(spoofitem in item for spoofitem in spoofitems)]
      numoflive = len(liveimages)
      numofspoof = len(spoofimges)
      print (numoflive, numofspoof, len(imgitmes))

def gentrainlist(strver, dbtypes, datacompipaths):
  allimagelist = []
  dbnameconcat = ""
  for dbidxpath in datacompipaths:
    dbpaths = dbidxpath[1]
    if "CASIA" in dbpaths:
      dbnameconcat = "{}_CASIA".format(dbnameconcat)
    if "MSU" in dbpaths:
      dbnameconcat = "{}_MSU".format(dbnameconcat)
    if "OULU" in dbpaths:
      dbnameconcat = "{}_OULU".format(dbnameconcat)
    if "REPLAY" in dbpaths:
      dbnameconcat = "{}_REPLAY".format(dbnameconcat)
    print (os.path.join(dbpaths, "{}_jpg".format(dbtypes.lower())))
    jpgitems = glob.glob("{}/**/*.jpg.fd".format(os.path.join(dbpaths, "{}_jpg".format(dbtypes.lower()))), recursive=True)

    if dbpaths in datakeys.keys():
      allimagelist.extend(datakeys[dbpaths])
      continue
    else:
      datakeys[dbpaths] = []
      datakeys[dbpaths].extend(jpgitems)

    print(len(jpgitems), dbpaths, dbtypes)
    allimagelist.extend(jpgitems)

  print (len(allimagelist))

  strtrainlist = "./{}_{}{}.list".format(dbtypes, strver, dbnameconcat)
  with open(strtrainlist, "w") as the_file:
    for imgpath in allimagelist:
      the_file.write("{}\n".format(imgpath.replace(".fd","")))
    the_file.close()

def gentestlist(strver, dbtypes, datacompipaths):
  for dbpaths in datacompipaths:
    dbnameconcat = ""
    if "CASIA" in dbpaths:
      dbnameconcat = "{}_CASIA".format(dbnameconcat)
    if "MSU" in dbpaths:
      dbnameconcat = "{}_MSU".format(dbnameconcat)
    if "OULU" in dbpaths:
      dbnameconcat = "{}_OULU".format(dbnameconcat)
    if "REPLAY" in dbpaths:
      dbnameconcat = "{}_REPLAY".format(dbnameconcat)

    jpgitems = glob.glob("{}/**/*.jpg.fd".format(os.path.join(dbpaths, "{}_jpg".format(dbtypes.lower()))), recursive=True)

    print(len(jpgitems), dbpaths, dbtypes)
    allimagelist = []
    allimagelist.extend(jpgitems)

    strtrainlist = "./{}_{}{}.list".format(dbtypes, strver, dbnameconcat)
    with open(strtrainlist, "w") as the_file:
      for imgpath in allimagelist:
        the_file.write("{}\n".format(imgpath.replace(".fd","")))
      the_file.close()

def main():
  print ("HI")
  for datacombi in list(combinations(enumerate(datapaths), 3)):
    gentrainlist("Protocal_4C3", datatypes[0], datacombi)
  gentestlist("Protocal_4C3", datatypes[1], datapaths)

if __name__ == '__main__':
  main()
