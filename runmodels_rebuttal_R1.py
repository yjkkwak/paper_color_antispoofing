import os

def runjobs():
  # strbaseckpt = "/home/user/model_2022/v4C3_sample/"
  #strbaseckpt = "/home/user/model_2022/v4C3_sample_K/"
  strbaseckpt = "/home/user/data2/model_rebuttal_RX_lamda1.0/"
  #strbaseckpt = "//home/user/data2/model_rebuttal_R1_lamda1.0/"
  # strbaseckpt = "/home/user/model_2022/v4C3_sample_siamese/"
  #strbaseckpt = "/home/user/model_2022/v4C3_sample_ablation/"
  strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrl_rebuttal_R1.py"
  #strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrl_k.py"
  # strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrl.py"
  # strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrlsiamese.py"
  # strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrlsiamese_K21.py"
  # strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrlsiamese_withBCE.py"
  # strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrl_abla_mixup.py"
  # strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrl_abla_baseline.py"

  strseed = 20220406
  strlr = 0.0001
  strgamma = 0.99
  nepoch = 150
  strbsize = 16
  stropti = "adam"


  # strgpu = 0
  # strDB = "Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260"
  # send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)
  #
  # strgpu = 1
  # strDB = "Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260"
  # send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)
  #
  # strgpu = 2
  # strDB = "Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260"
  # send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)
  #
  # strgpu = 3
  # strDB = "Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260"
  # send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

  strlr = 0.000166
  strseed = 20220408
  # strgpu = 0
  # strDB = "Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260"
  # send4C4jobs(s trpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)
  strlr = 0.00018
  strseed = 20200908
  strgpu = 1
  strDB = "Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

  strlr = 0.00018
  strseed = 20210908
  strgpu = 2
  strDB = "Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

  strlr = 0.00018
  strseed = 20190908
  strgpu = 3
  strDB = "Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)



def send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed):
  strmeta = "mseregloss_resnet18_{}_rebuttal_RX_samelamdas".format(stropti)
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
  lmdbpath = "/home/user/data2/work_db/v4C3/{}.db".format(strDB)
  strcmd = "{} {} --ckptpath {} --lmdbpath {} --lr {}  --gamma {} --opt {} --epochs {} --batch_size {} --GPU {} --meta {} --random_seed {}".format(
    screenoption, strpython, strbaseckpt, lmdbpath, strlr, strgamma, stropti, nepoch, strbsize, strgpu, strmeta, strseed)
  os.system(strcmd)

if __name__ == '__main__':
  runjobs()
