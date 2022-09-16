import os
import time

def getbasenamewoext(srcfile):
  pathname, extension = os.path.splitext(srcfile)
  return pathname

def main():
  basemeta = "/home/user/work_2022/Paper_AntiSpoofing/meta/script/Protocal_4C3"
  # basemeta = "/home/user/work_2022/Paper_AntiSpoofing/meta/script/OULU_P"
  baselmdb = "/home/user/work_db/v4C3/"
  dblist = [
            # "Test_OULU_Protocol_1.list",
            # "Test_OULU_Protocol_2.list",
            # "Test_OULU_Protocol_3_1.list",
            # "Test_OULU_Protocol_3_2.list",
            # "Test_OULU_Protocol_3_3.list",
            # "Test_OULU_Protocol_3_4.list",
            # "Test_OULU_Protocol_3_5.list",
            # "Test_OULU_Protocol_3_6.list",
            # "Test_OULU_Protocol_4_1.list",
            # "Test_OULU_Protocol_4_2.list",
            # "Test_OULU_Protocol_4_3.list",
            # "Test_OULU_Protocol_4_4.list",
            # "Test_OULU_Protocol_4_5.list",
            # "Test_OULU_Protocol_4_6.list",
            # "Train_OULU_Protocol_1.list",
            # "Train_OULU_Protocol_2.list",
            # "Train_OULU_Protocol_3_1.list",
            # "Train_OULU_Protocol_3_2.list",
            # "Train_OULU_Protocol_3_3.list",
            # "Train_OULU_Protocol_3_4.list",
            # "Train_OULU_Protocol_3_5.list",
            # "Train_OULU_Protocol_3_6.list",
            # "Train_OULU_Protocol_4_1.list",
            # "Train_OULU_Protocol_4_2.list",
            # "Train_OULU_Protocol_4_3.list",
            # "Train_OULU_Protocol_4_4.list",
            # "Train_OULU_Protocol_4_5.list",
            # "Train_OULU_Protocol_4_6.list"
            # "Test_Protocal_4C3_MSU.list",
            # "Test_Protocal_4C3_CASIA.list",
            # "Test_Protocal_4C3_OULU.list",
            # "Test_Protocal_4C3_REPLAY.list",
            # "Train_Protocal_4C3_CASIA_MSU_OULU.list",
            # "Train_Protocal_4C3_CASIA_MSU_REPLAY.list",
            "Train_Protocal_4C3_CASIA_OULU_REPLAY.list",
            # "Train_Protocal_4C3_MSU_OULU_REPLAY.list",
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
      usedpathname = "{}_{}.db.sort.path".format(getbasenamewoext(dbitem), patchitem)
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
