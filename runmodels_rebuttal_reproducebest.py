import os

def runjobs():

  ## wo grl
  ## wo le
  ## wo pd le

  strbaseckpt = "/home/user/data2/model_rebuttal_RX_reproduce/"
  strpython = "python -u /home/user/work_2022/Paper_AntiSpoofing/train_regwgrl_rebuttal_R1.py"

  strgamma = 0.99

  strbsize = 16
  stropti = "adam"

  nepoch = 1000
  strlr = 0.00015
  strseed = 20220406
  strgpu = 0
  strDB = "Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260"
  #/home/user/model_2022/v4C3_sample_retutaal_RX/model_rebuttal_RX_lamda1.0/Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260_220909_hYNRU7DXoYNXWKzgCZeofW_bsize16_optadam_lr0.00015_gamma_0.99_epochs_1000_meta_mseregloss_resnet18_adam_rebuttal_RX_samelamda_lamda_1.0
  # send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

  # reset OK
  nepoch = 150
  strgpu = 1
  strseed = 20200908
  strlr = 0.00017
  #/home/user/data2/model_rebuttal_RX_lamda1.0/Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260_221001_3aZFGcgmY4rVD7QEyQnsJ3_bsize16_optadam_lr0.00017_gamma_0.99_epochs_150_meta_mseregloss_resnet18_adam_rebuttal_RX_samelamda_lamda_1.0/trainlog.txt
  strDB = "Train_Protocal_4C3_CASIA_MSU_REPLAY_1by1_260x260"
  # send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

  # reset OK
  nepoch = 300
  strgpu = 2
  strlr = 0.00013
  strseed = 20220408
  #/home/user/data2/model_rebuttal_RX_lamda1.0/Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260_220930_7bo82FJuX6Fj25z2xe7bq8_bsize16_optadam_lr0.00013_gamma_0.99_epochs_300_meta_mseregloss_resnet18_adam_rebuttal_RX_samelamda_lamda_1.0
  strDB = "Train_Protocal_4C3_CASIA_OULU_REPLAY_1by1_260x260"
  # send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

  # reset OK, //12
  #02.score_acc_59.943978_fpr_40.074906_frr_40.000000_hter_40.037453_auc_63.462339
  #/home/user/data2/model_rebuttal_RX_lamda1.0/Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260_221002_UHBarinHtmPQAhyGzMgtWF_bsize16_optadam_lr0.00018_gamma_0.99_epochs_150_meta_mseregloss_resnet18_adam_rebuttal_RX_samelamdas_lamda_1.0/Test_Protocal_4C3_CASIA_1by1_260x260.db/
  nepoch = 150
  strgpu = 3
  strlr = 0.00018
  strseed = 20200908
  strDB = "Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260"
  # send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)




  ## c only
  nepoch = 150
  strgpu = 0
  strlr = 0.0001
  strseed = 201110901
  strDB = "Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

  nepoch = 150
  strgpu = 1
  strlr = 0.00011
  strseed = 20230902
  strDB = "Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

  nepoch = 150
  strgpu = 2
  strlr = 0.00012
  strseed = 201120938
  strDB = "Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

  nepoch = 150
  strgpu = 3
  strlr = 0.00013
  strseed = 20210904
  strDB = "Train_Protocal_4C3_MSU_OULU_REPLAY_1by1_260x260"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)



def send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed):
  #strmeta = "mseregloss_resnet18_{}_rebuttal_reproduce".format(stropti)
  strmeta = "mseregloss_resnet18_{}_rebuttal_woPD".format(stropti)
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
