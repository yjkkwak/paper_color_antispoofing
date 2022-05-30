import os
import time

def getbasenamewoext(srcfile):
  pathname, extension = os.path.splitext(srcfile)
  return pathname

def main():
  basemeta = "/home/user/work_2022/Paper_AntiSpoofing/meta/script/Protocal_4C3"
  baselmdb = "/home/user/work_db/v4C3/"
  dblist = [
            "Test_Protocal_4C3_CASIA.list",
            "Test_Protocal_4C3_MSU.list",
            "Test_Protocal_4C3_OULU.list",
            "Test_Protocal_4C3_REPLAY.list",
            "Train_Protocal_4C3_CASIA_MSU_OULU.list",
            "Train_Protocal_4C3_CASIA_MSU_REPLAY.list",
            "Train_Protocal_4C3_CASIA_OULU_REPLAY.list",
            "Train_Protocal_4C3_MSU_OULU_REPLAY.list",
            ]
  patchtypelist = ["1by1_260x260"]

  for dbitem in dblist:
    for patchitem in patchtypelist:
      strpythoncmd = "python -u lmdbwriter.py "
      stroptions = " -listpath {} -dbpath {} -patchtype {}".format(
        os.path.join(basemeta, dbitem),
        baselmdb,
        patchitem)
      strcmd = "{} {}".format(strpythoncmd, stroptions)
      usedpathname = "{}_{}.db.path".format(getbasenamewoext(dbitem), patchitem)
      usedimagepath = os.path.join(baselmdb, "{}".format(usedpathname))
      print (usedimagepath)
  #    time.sleep(10)
      os.system(strcmd)
      while True:
        if os.path.exists(usedimagepath):
          break
        time.sleep(100)
        print ("Wait!, generating lmdb", usedimagepath)

  print ("Done!!!")



if __name__ == '__main__':
  main()
