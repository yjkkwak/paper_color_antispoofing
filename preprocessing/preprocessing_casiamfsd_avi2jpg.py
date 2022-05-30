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
  dstpath = fdirname.replace("_release", "")

  realorspoof = "spoof"
  #if (flag == '1' or flag == '2' or flag == 'HR_1'):
  if "/1.avi" in fullpath:
    fname = fname.replace("1.avi", "1_real")
    realorspoof = "real"
  elif "/2.avi" in fullpath:
    fname = fname.replace("2.avi", "2_real")
    realorspoof = "real"
  elif "/HR_1.avi" in fullpath:
    fname = fname.replace("HR_1.avi", "HR_1_real")
    realorspoof = "real"
  else:
    fname = fname.replace(".avi", "_spoof")

  dstpath = os.path.join(dstpath, realorspoof, fnamewext)
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
  favilist = glob.glob("{}/**/*.avi".format(movpath), recursive=True)
  # for favipath in favilist:
  #   extractjpgfromavi(favipath)
    # break

  pool = mp.Pool(min(mp.cpu_count(), len(favilist)))  # number of workers
  pool.map(extractjpgfromavi, favilist, chunksize=10)
  pool.close()


def main():
  convertavi2jpg("/home/user/work_db/PublicDB/CASIA-MFSD/test_release")
  convertavi2jpg("/home/user/work_db/PublicDB/CASIA-MFSD/train_release")


if __name__ == '__main__':
  main()
