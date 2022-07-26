import os

def runjobs():
  # strbaseckpt = "/home/user/model_2022/v4C3_sample/"
  strbaseckpt = "/home/user/model_2022/v4C3_sample_K/"
  # strbaseckpt = "/home/user/model_2022/v4C3_sample_siamese/"
  #strbaseckpt = "/home/user/model_2022/v4C3_sample_ablation/"
  strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrl_k.py"
  # strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrl.py"
  # strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrlsiamese.py"
  # strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrlsiamese_K21.py"
  # strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrlsiamese_withBCE.py"
  # strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrl_abla_mixup.py"
  # strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrl_abla_baseline.py"

  strlr = 0.00016
  strgamma = 0.99
  nepoch = 1000
  strbsize = 16
  stropti = "adam"
  strseed = 20220406
  strgpu = 0
  strDB = "Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260"


  # strDB = "Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260"
  strk = 6
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed, strk)
  return
  strseed = 20220406
  strgpu = 1
  strk = 21
  # strDB = "Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed, strk)

  strseed = 20220406
  strgpu = 2
  strk = 31
  # strDB = "Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed, strk)

  strseed = 20220406
  strgpu = 3
  strk = 41
  # strDB = "Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed, strk)

def send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed, strk):
  strmeta = "mseregloss_resnet18_{}_baseline_{}_again_allimgtest".format(stropti, strk)
  strlogoption = "log_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(strDB,
                                                      stropti,
                                                      "MESREG",
                                                      "lr{}".format(strlr),
                                                      "gamma{}".format(strgamma),
                                                      "e{}".format(nepoch),
                                                      "bsize{}".format(strbsize),
                                                      "gpu{}".format(strgpu),
                                                      "meta{}".format(strmeta),
                                                      "seed{}".format(strseed),
                                                      strk)
  screenoption = "screen -L -Logfile {}.txt -d -m ".format(strlogoption)
  lmdbpath = "/home/user/work_db/v4C3/{}.db".format(strDB)
  strcmd = "{} {} --ckptpath {} --lmdbpath {} --lr {}  --gamma {} --opt {} --epochs {} --batch_size {} --GPU {} --meta {} --random_seed {} --lk {}".format(
    screenoption, strpython, strbaseckpt, lmdbpath, strlr, strgamma, stropti, nepoch, strbsize, strgpu, strmeta, strseed, strk)
  os.system(strcmd)

if __name__ == '__main__':
  runjobs()
