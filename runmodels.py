import os

def runjobs():
  # strbaseckpt = "/home/user/model_2022/v4C3_sample/"
  strbaseckpt = "/home/user/model_2022/v4C3_sample_siamese/"
  # strbaseckpt = "/home/user/model_2022/v4C3_sample_ablation/"
  # strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrl.py"
  strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrlsiamese.py"
  # strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrl_abla_mixup.py"
  # strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrl_abla_baseline.py"

  strlr = 0.0001
  strgamma = 0.9999
  nepoch = 1000
  strbsize = 16
  stropti = "adam"

  # strgpu = 0
  # strDB = "Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260"
  # send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu)
  #
  # strgpu = 1
  # strDB = "Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260"
  # send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu)
  #
  # strgpu = 2
  # strDB = "Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260"
  # send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu)
  #
  # strgpu = 3
  # strDB = "Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260"
  # send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu)

  strgpu = 0
  strDB = "Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260"
  strseed = "20200511"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

  strgpu = 1
  strDB = "Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260"
  strseed = "20200510"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

  strgpu = 2
  strDB = "Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260"
  strseed = "20200509"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

  strgpu = 3
  strDB = "Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260"
  strseed = "20200508"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

def send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed):
  strmeta = "mseregloss_resnet18_{}_baseline_siamese_test1".format(stropti)
  strlogoption = "log_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(strDB,
                                                      stropti,
                                                      "MESREG",
                                                      "lr{}".format(strlr),
                                                      "gamma{}".format(strgamma),
                                                      "e{}".format(nepoch),
                                                      "bsize{}".format(strbsize),
                                                      "gpu{}".format(strgpu),
                                                      "meta{}".format(strmeta),
                                                      "seed{}".format(strseed))
  screenoption = "screen -L -Logfile {}.txt -d -m ".format(strlogoption)
  lmdbpath = "/home/user/work_db/v4C3/{}.db".format(strDB)
  strcmd = "{} {} --ckptpath {} --lmdbpath {} --lr {}  --gamma {} --opt {} --epochs {} --batch_size {} --GPU {} --meta {} --random_seed {}".format(
    screenoption, strpython, strbaseckpt, lmdbpath, strlr, strgamma, stropti, nepoch, strbsize, strgpu, strmeta, strseed)
  os.system(strcmd)

if __name__ == '__main__':
  runjobs()
