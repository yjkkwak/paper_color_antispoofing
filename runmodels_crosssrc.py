import os

def runjobs():
  strbaseckpt = "/home/user/model_2022/v4C3_sample_ablation/"
  strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrlsiamese_crosssrc.py"

  strlr = 0.0001
  strgamma = 0.99
  nepoch = 1000
  strbsize = 16

  stropti = "adam"
  #C , sgd
  strlr = 0.0001
  strrandom_seed = 20220406
  strgpu = 0
  strDB = "Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strrandom_seed)

  strlr = 0.0005
  strgpu = 1
  strDB = "Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strrandom_seed)

  #R
  strlr = 0.0001
  strrandom_seed = 20220408
  strgpu = 2
  strDB = "Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strrandom_seed)

  strlr = 0.00005
  strgpu = 3
  strDB = "Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strrandom_seed)


def send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strrandom_seed):
  strmeta = "msegrlloss_resnet18_{}_full_siamese_crosssrc_sax5".format(stropti)
  strlogoption = "log_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(strDB,
                                                      stropti,
                                                      "MSEADV",
                                                      "lr{}".format(strlr),
                                                      "gamma{}".format(strgamma),
                                                      "e{}".format(nepoch),
                                                      "bsize{}".format(strbsize),
                                                      "gpu{}".format(strgpu),
                                                      "meta{}".format(strmeta))
  screenoption = "screen -L -Logfile {}.txt -d -m ".format(strlogoption)
  lmdbpath = "/home/user/work_db/v4C3/{}.db".format(strDB)
  strcmd = "{} {} --ckptpath {} --lmdbpath {} --lr {}  --gamma {} --opt {} --epochs {} --batch_size {} --GPU {} --meta {} --random_seed {}".format(
    screenoption, strpython, strbaseckpt, lmdbpath, strlr, strgamma, stropti, nepoch, strbsize, strgpu, strmeta, strrandom_seed)
  os.system(strcmd)

if __name__ == '__main__':
  runjobs()
