import glob
import tarfile
import zipfile
import os
from tqdm import tqdm
import multiprocessing as mp
import cv2
from shutil import copyfile

def mydelcmd(strcmd):
  print (strcmd)
  os.system(strcmd)

def getbasenamewoext(srcfile):
  pathname, extension = os.path.splitext(srcfile)
  return pathname

def extractjpgfromavi(fullpath):
  """worker unzips one file"""
  print ("extracting... {}".format(fullpath))
  fdirname = os.path.dirname(fullpath)
  fname = os.path.basename(fullpath)
  fnamewext = os.path.basename(getbasenamewoext(fullpath))
  dstpath = fdirname.replace("_scene", "_jpg")

  print (dstpath)


  # print (dstpath, fname)
  if os.path.exists(dstpath) == False:
    os.makedirs(dstpath, exist_ok=True)

  vidcap = cv2.VideoCapture(fullpath)
  success, image = vidcap.read()
  count = 0
  while success:
    jpgpath = os.path.join(dstpath, "{}_{}.jpg".format(fname, count))
    cv2.imwrite(jpgpath, image)  # save frame as JPEG file
    # print (jpgpath)
    success, image = vidcap.read()
    #print('Read a new frame: ', success, jpgpath)
    # pass 0.7 sec
    #vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 700))  # added this line
    count += 1

def convertavi2jpg(movpath):
  print(movpath)
  favilist = []
  fmovlist = glob.glob("{}/**/*.mov".format(movpath), recursive=True)
  fmp4list = glob.glob("{}/**/*.mp4".format(movpath), recursive=True)
  favilist.extend(fmovlist)
  favilist.extend(fmp4list)
  # for favipath in favilist:
  #   extractjpgfromavi(favipath)
  #   break

  pool = mp.Pool(min(mp.cpu_count(), len(favilist)))  # number of workers
  pool.map(extractjpgfromavi, favilist, chunksize=10)
  pool.close()

def splittraintest():
  trainids = []
  testids = []
  # strtraininfo = "/home/user/work_db/PublicDB/MSU-MFSD/train_sub_list.txt"
  # strtestinfo = "/home/user/work_db/PublicDB/MSU-MFSD/test_sub_list.txt"

  strtraininfo = "/home/user/data1/DBs/antispoofing/MSU-MFSD/MSU-MFSD/train_sub_list.txt"
  strtestinfo = "/home/user/data1/DBs/antispoofing/MSU-MFSD/MSU-MFSD/test_sub_list.txt"
  with open(strtraininfo, "r") as the_file:
    strlines = the_file.readlines()
    the_file.close()
  for strline in strlines:
    trainids.append(strline.strip())

  with open(strtestinfo, "r") as the_file:
    strlines = the_file.readlines()
    the_file.close()
  for strline in strlines:
    testids.append(strline.strip())
  print(trainids)
  print(testids)
  #videopath = "/home/user/work_db/PublicDB/MSU-MFSD/scene01"

  videopath = "/home/user/data1/DBs/antispoofing/MSU-MFSD/MSU-MFSD/scene01"
  # fmovlist = glob.glob("{}/**/*.mov".format(videopath), recursive=True)
  fmp4list = glob.glob("{}/**/*.mp4".format(videopath), recursive=True)

  # print (len(fmovlist), fmovlist[0])
  print(len(fmp4list), fmp4list[0])

  for fmovitem in fmp4list:

    dname = os.path.dirname(fmovitem)
    fname = os.path.basename(fmovitem)


    strtoken = fname.split("_")[1].replace("client0", "")

    if strtoken in trainids:
      destdname = dname.replace("scene01", "train_scene")
    else:
      destdname = dname.replace("scene01", "test_scene")

    if os.path.exists(destdname) == False:
      os.makedirs(destdname, exist_ok=True)

    destpath = os.path.join(destdname, fname)
    copyfile (fmovitem, destpath)

def main():
  # splittraintest()
  #
  convertavi2jpg("/home/user/work_db/PublicDB/MSU-MFSD/train_scene")
  convertavi2jpg("/home/user/work_db/PublicDB/MSU-MFSD/test_scene")


if __name__ == '__main__':
  main()
