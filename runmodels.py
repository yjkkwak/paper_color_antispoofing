import os




def runjobs():
  strbaseckpt = "/home/user/model_2022/v4C3/"
  strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_mixup.py"

  strlr = 0.0001
  strgamma = 0.90
  nepoch = 40
  strbsize = 128
  stropti = "adam"

  strgpu = 0
  strDB = "Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu)

  strgpu = 1
  strDB = "Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu)

  strgpu = 2
  strDB = "Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu)

  strgpu = 3
  strDB = "Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu)

def send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu):
  strmeta = "clsloss_resnet18_{}_wmixup".format(stropti)
  strlogoption = "log_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(strDB,
                                                      stropti,
                                                      "Clsloss",
                                                      "lr{}".format(strlr),
                                                      "gamma{}".format(strgamma),
                                                      "e{}".format(nepoch),
                                                      "bsize{}".format(strbsize),
                                                      "gpu{}".format(strgpu),
                                                      "meta{}".format(strmeta))
  screenoption = "screen -L -Logfile {}.txt -d -m ".format(strlogoption)
  lmdbpath = "/home/user/work_db/v4C3/{}.db".format(strDB)
  strcmd = "{} {} --ckptpath {} --lmdbpath {} --lr {}  --gamma {} --opt {} --epochs {} --batch_size {} --GPU {} --meta {} ".format(
    screenoption, strpython, strbaseckpt, lmdbpath, strlr, strgamma, stropti, nepoch, strbsize, strgpu, strmeta)
  os.system(strcmd)

if __name__ == '__main__':
  runjobs()
