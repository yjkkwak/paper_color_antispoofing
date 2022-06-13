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
datapaths = ["/home/user/work_db/PublicDB/OULU-NPU/",  # train_jpg / test_jpg / devel_jpg/       --> real spoof
             ]

def genlist(strver, dbtypes, datacompipaths):
  for dbpaths in datacompipaths:
    dbnameconcat = ""
    if "OULU" in dbpaths:
      dbnameconcat = "{}_OULU".format(dbnameconcat)

    jpgitems = glob.glob("{}/**/*.jpg.fd".format(os.path.join(dbpaths, "{}_jpg".format(dbtypes.lower()))), recursive=True)

    print(len(jpgitems), dbpaths, dbtypes)
    allimagelist = []
    allimagelist.extend(jpgitems)

    strtrainlist = "./{}_{}{}.list".format(dbtypes, strver, dbnameconcat)
    with open(strtrainlist, "w") as the_file:
      for imgpath in allimagelist:
        the_file.write("{}\n".format(imgpath.replace(".fd","")))
      the_file.close()

def genprotocol(protocoltype, subptype):
  strbasetrainlist = "./Train_Protocal_4C3_OULU.list"
  strbasetestlist = "./Test_Protocal_4C3_OULU.list"


  with open(strbasetrainlist, "r") as the_file:
    trainlists = the_file.readlines()
    the_file.close()
  with open(strbasetestlist, "r") as the_file:
    testlists = the_file.readlines()
    the_file.close()

  strprotocolpath = "/home/user/work_db/PublicDB/OULU-NPU/Protocols/Protocol_"
  strptrainpath = "{}{}/{}".format(strprotocolpath, protocoltype, "Train_{}.txt".format(subptype))
  strptestpath = "{}{}/{}".format(strprotocolpath, protocoltype, "Test_{}.txt".format(subptype))


  with open(strptrainpath, "r") as the_file:
    trainprotocollists = the_file.readlines()
    the_file.close()
  with open(strptestpath, "r") as the_file:
    testprotocollists = the_file.readlines()
    the_file.close()

  trainids = testids = []
  for strline in trainprotocollists:
    strline = strline.strip()
    trainids.append(strline.split(",")[1])
  for strline in testprotocollists:
    strline = strline.strip()
    testids.append(strline.split(",")[1])

  print (len(trainlists),len(testlists),len(trainids),len(testids))
  print (trainids)

  the_file = open("./Train_OULU_Protocol_{}_{}.list".format(protocoltype, subptype), "w")
  for trainitem in trainlists:
    trainitem = trainitem.strip()
    if any(iditem in trainitem for iditem in trainids):
      the_file.write("{}\n".format(trainitem))
  the_file.close()
  the_file = open("./Test_OULU_Protocol_{}_{}.list".format(protocoltype, subptype), "w")
  for testitem in testlists:
    testitem = testitem.strip()
    if any(iditem in testitem for iditem in testids):
      the_file.write("{}\n".format(testitem))
  the_file.close()

#6_3_48_4
def main():
  print ("HI")
  # genlist("Protocal_4C3", datatypes[0], datapaths)
  # genlist("Protocal_4C3", datatypes[1], datapaths)
  #strbaselist, strprotocolpath, protocoltype
  # genprotocol(1)
  # genprotocol(2)
  for ii in range(7):
    genprotocol(3, ii+1)
    genprotocol(4, ii+1)
  # # protocol 1
  # strprotocolpath = "/home/user/work_db/PublicDB/OULU-NPU/Protocols/Protocol_"
  #
  # strptrainpath = "{}1/{}".format(strprotocolpath, "Train.txt")
  # strptestpath = "{}1/{}".format(strprotocolpath, "Test.txt")


if __name__ == '__main__':
  main()
