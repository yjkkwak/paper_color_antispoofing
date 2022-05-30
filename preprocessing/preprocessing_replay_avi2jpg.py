import glob
import tarfile
import zipfile
import os
from tqdm import tqdm
import multiprocessing as mp
import cv2

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
  dstpath = fdirname.replace("/real", "_jpg/real").replace("/attack/", "_jpg/attack/")


  print (dstpath)

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
  favilist = glob.glob("{}/**/*.mov".format(movpath), recursive=True)
  # for favipath in favilist:
  #   extractjpgfromavi(favipath)
  #   break

  pool = mp.Pool(min(mp.cpu_count(), len(favilist)))  # number of workers
  pool.map(extractjpgfromavi, favilist, chunksize=10)
  pool.close()

def extractfdfromtxt(fullpath):
  print ("extracting... {}".format(fullpath))
  fdirname = os.path.dirname(fullpath)
  fname = os.path.basename(fullpath)
  dstpath = fdirname.replace("/face-locations/", "/").replace("/real", "_jpg/real").replace("/attack/", "_jpg/attack/")
  print (dstpath)
  #
  with open(fullpath, "r") as the_file:
    strlines = the_file.readlines()
    the_file.close()

  for count, strline in enumerate(strlines):
    fdpath = os.path.join(dstpath, "{}_{}.jpg.fd".format(fname, count))
    fdpath = fdpath.replace(".face", ".mov")
    with open(fdpath, "w") as fd_file:
      #fd_file.write("{}\n".format(strline))
      strtoknes = strline.strip().split()
      fd_file.write(" ".join(e for e in strtoknes[1:]))
      # print(" ".join(e for e in strtoknes[1:]))
      fd_file.close()

def convertface2txt(movpath):
  print(movpath)
  favilist = glob.glob("{}/**/*.face".format(movpath), recursive=True)
  # for favipath in favilist:
  #   extractfdfromtxt(favipath)
  #   break

  pool = mp.Pool(min(mp.cpu_count(), len(favilist)))  # number of workers
  pool.map(extractfdfromtxt, favilist, chunksize=10)
  pool.close()

def main():
  # convertavi2jpg("/home/user/work_db/PublicDB/REPLAY-ATTACK/devel")
  # convertavi2jpg("/home/user/work_db/PublicDB/REPLAY-ATTACK/test")
  # convertavi2jpg("/home/user/work_db/PublicDB/REPLAY-ATTACK/train")
  #
  convertface2txt("/home/user/work_db/PublicDB/REPLAY-ATTACK/face-locations/devel")
  convertface2txt("/home/user/work_db/PublicDB/REPLAY-ATTACK/face-locations/test")
  convertface2txt("/home/user/work_db/PublicDB/REPLAY-ATTACK/face-locations/train")



if __name__ == '__main__':
  main()
