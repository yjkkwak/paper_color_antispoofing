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
  fnamewext = os.path.basename(getbasenamewoext(fullpath))
  dstpath = fdirname.replace("_files", "")

  realorspoof = "spoof"
  # 1=real; 2=print1; 3=print2; 4=video-replay1; 5=video-replay2
  if "_1.avi" in fname:
    fname = fname.replace("_1.avi", "_1_real")
    realorspoof = "real"
  elif "_2.avi" in fname:
    fname = fname.replace("_2.avi", "_2_print1")
  elif "_3.avi" in fname:
    fname = fname.replace("_3.avi", "_3_print2")
  elif "_4.avi" in fname:
    fname = fname.replace("_4.avi", "_4_replay1")
  elif "_5.avi" in fname:
    fname = fname.replace("_5.avi", "_5_replay2")
  dstpath = os.path.join(dstpath, realorspoof, fnamewext)
  print (dstpath)
  if os.path.exists(dstpath) == False:
    os.makedirs(dstpath, exist_ok=True)

  vidcap = cv2.VideoCapture(fullpath)
  success, image = vidcap.read()
  count = 0
  while success:
    jpgpath = os.path.join(dstpath, "{}_{}.jpg".format(fname, count))
    cv2.imwrite(jpgpath, image)  # save frame as JPEG file
    success, image = vidcap.read()
    #print('Read a new frame: ', success, jpgpath)
    # pass 0.7 sec
    #vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 700))  # added this line
    count += 1



def extractfdfromtxt(fullpath):
  print ("extracting... {}".format(fullpath))
  fdirname = os.path.dirname(fullpath)
  fname = os.path.basename(fullpath)
  fnamewext = os.path.basename(getbasenamewoext(fullpath))
  dstpath = fdirname.replace("_files", "")

  realorspoof = "spoof"
  # 1=real; 2=print1; 3=print2; 4=video-replay1; 5=video-replay2
  if "_1.txt" in fname:
    fname = fname.replace("_1.txt", "_1_real")
    realorspoof = "real"
  elif "_2.txt" in fname:
    fname = fname.replace("_2.txt", "_2_print1")
  elif "_3.txt" in fname:
    fname = fname.replace("_3.txt", "_3_print2")
  elif "_4.txt" in fname:
    fname = fname.replace("_4.txt", "_4_replay1")
  elif "_5.txt" in fname:
    fname = fname.replace("_5.txt", "_5_replay2")
  dstpath = os.path.join(dstpath, realorspoof, fnamewext)
  print (dstpath)
  if os.path.exists(dstpath) == False:
    os.makedirs(dstpath, exist_ok=True)


  with open(fullpath, "r") as the_file:
    strlines = the_file.readlines()
    the_file.close()

  for count, strline in enumerate(strlines):
    fdpath = os.path.join(dstpath, "{}_{}.jpg.fd".format(fname, count))
    #print (fdpath)
    with open(fdpath, "w") as fd_file:
      #fd_file.write("{}\n".format(strline))
      strtoknes = strline.strip().split(",")
      fd_file.write(" ".join(e for e in strtoknes[1:]))
      fd_file.close()


def convertavi2jpg(movpath):
  print(movpath)
  favilist = glob.glob("{}/**/*.avi".format(movpath), recursive=True)
  # for favipath in favilist:
  #   extractjpgfromavi(favipath)
  #   break

  pool = mp.Pool(min(mp.cpu_count(), len(favilist)))  # number of workers
  pool.map(extractjpgfromavi, favilist, chunksize=10)
  pool.close()


def convertface2txt(movpath):
  print(movpath)
  favilist = glob.glob("{}/**/*.jpg".format(movpath), recursive=True)
  for favipath in favilist:
    print (favipath)
    extractfdfromtxt(favipath)
    break
  # pool = mp.Pool(min(mp.cpu_count(), len(favilist)))  # number of workers
  # pool.map(extractfdfromtxt, favilist, chunksize=10)
  # pool.close()

def main():
  # convertavi2jpg("/home/user/work_db/PublicDB/OULU-NPU/Dev_files")
  # convertavi2jpg("/home/user/work_db/PublicDB/OULU-NPU/Train_files")
  # convertavi2jpg("/home/user/work_db/PublicDB/OULU-NPU/Test_files")

  convertface2txt("/home/user/work_db/PublicDB/OULU-NPU/devel_jpg")
  convertface2txt("/home/user/work_db/PublicDB/OULU-NPU/train_jpg")
  convertface2txt("/home/user/work_db/PublicDB/OULU-NPU/test_jpg")

if __name__ == '__main__':
  main()
