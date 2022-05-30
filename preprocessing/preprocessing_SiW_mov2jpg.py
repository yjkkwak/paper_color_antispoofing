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

def extractjpgfrommov(fullpath):
  """worker unzips one file"""
  print ("extracting... {}".format(fullpath))
  fdirname = os.path.dirname(fullpath)
  fnamewext = getbasenamewoext(fullpath)
  dstpath = fnamewext.replace("SiW/SiW_release", "SiW/SiW_jpg")
  print (dstpath)

  if os.path.exists(dstpath) == False:
    os.makedirs(dstpath, exist_ok=True)

  vidcap = cv2.VideoCapture(fullpath)
  success, image = vidcap.read()
  count = 0
  while success:
    jpgpath = os.path.join(dstpath, "frame{}.jpg".format(count))
    cv2.imwrite(jpgpath, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success, jpgpath)
    # pass 0.7 sec
    vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 700))  # added this line
    count += 1


def convertmov2jpg(movpath):
  print(movpath)
  fmovlist = glob.glob("{}/**/*.mov".format(movpath), recursive=True)
  # for fmovpath in fmovlist:
  #   extractjpgfrommov(fmovpath)
  #   break

  pool = mp.Pool(min(mp.cpu_count(), len(fmovlist)))  # number of workers
  pool.map(extractjpgfrommov, fmovlist, chunksize=10)
  pool.close()

def main():
  # convertmov2jpg("/home/user/data1/DBs/antispoofing/SiW/SiW_release/Train")
  convertmov2jpg("/home/user/data1/DBs/antispoofing/SiW/SiW_release/Test")
  
if __name__ == '__main__':
    main()
