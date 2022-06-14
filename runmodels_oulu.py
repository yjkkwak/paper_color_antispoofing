import os

def runjobs():
  strbaseckpt = "/home/user/model_2022/vOP"
  strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrlsiamese_oulu.py"

  strlr = 0.0001
  strgamma = 0.99
  nepoch = 1000
  strbsize = 16
  strrandom_seed = 20220406

  stropti = "adam"


  strgpu = 0
  strDB = "Train_OULU_Protocol_4_1_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strrandom_seed)

  strgpu = 1
  strDB = "Train_OULU_Protocol_4_2_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strrandom_seed)

  strgpu = 2
  strDB = "Train_OULU_Protocol_4_3_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strrandom_seed)

  strgpu = 3
  strDB = "Train_OULU_Protocol_4_4_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strrandom_seed)


def send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strrandom_seed):
  strmeta = "msegrlloss_resnet18_{}_full_siamese_oulu".format(stropti)
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
  lmdbpath = "/home/user/work_db/vOP/{}.db.sort".format(strDB)
  strcmd = "{} {} --ckptpath {} --lmdbpath {} --lr {}  --gamma {} --opt {} --epochs {} --batch_size {} --GPU {} --meta {} --random_seed {}".format(
    screenoption, strpython, strbaseckpt, lmdbpath, strlr, strgamma, stropti, nepoch, strbsize, strgpu, strmeta, strrandom_seed)
  os.system(strcmd)

if __name__ == '__main__':
  runjobs()
