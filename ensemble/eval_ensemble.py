from utils import genfarfrreerwth, genfarfrreerwthlist

def main():
  print ("abc")
  getbasemodels()

def getscorewsortbypath(strsocrepath):
  the_file = open(strsocrepath, "r")
  strlines = the_file.readlines()
  scorelist = []
  for strline in strlines:
    strtokens = strline.split()
    scorelist.append([float(strtokens[0]), float(strtokens[1]), strtokens[2]])
  the_file.close()
  scorelist = sorted(scorelist, key = lambda x:x[2])
  return scorelist

def ensemble_scores(db1, db2):
  spath1 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_1by1_260x260_220502_3uKCX7S9pwbeSTzoTydcgV_lr0.005_gamma_0.92_epochs_80_meta_163264/{}/78.score".format(db1)
  spath2 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_4by3_244x324_220504_eNeMv72oynyYhUikgY4mbv_lr0.001_gamma_0.92_epochs_80_meta_163264/{}/69.score".format(db2)

  scorelist1 = getscorewsortbypath(spath1)
  scorelist2 = getscorewsortbypath(spath2)

  wlist = [0.5, 0.5]
  the_file = open("./{}{}.txt".format(db1, db2), "w")
  for item in zip(scorelist1, scorelist2):
    fakescore = wlist[0]*float(item[0][0]) + wlist[1]*float(item[1][0])
    livescore = wlist[0] * float(item[0][1]) + wlist[1] * float(item[1][1])
    imgpath = item[0][2]
    the_file.write("{} {} {}\n".format(fakescore, livescore, imgpath))
  the_file.close()

  genfarfrreerwthlist(spath1)
  genfarfrreerwthlist(spath2)
  genfarfrreerwthlist("./{}{}.txt".format(db1, db2))




def getbasemodels():
  # ensemble_scores("Test_v220419_01_CelebA_1by1_260x260", "Test_v220419_01_CelebA_4by3_244x324")
  # ensemble_scores("Test_v220419_01_LD3007_1by1_260x260", "Test_v220419_01_LD3007_4by3_244x324")
  # ensemble_scores("Test_v220419_01_LDRGB_1by1_260x260", "Test_v220419_01_LDRGB_4by3_244x324")
  #ensemble_scores("Test_v220419_01_SiW_1by1_260x260", "Test_v220419_01_SiW_4by3_244x324")
  #ensemble_scores("Dev_v220419_01_OULUNPU_1by1_260x260", "Dev_v220419_01_OULUNPU_4by3_244x324")

  ensemble_scores("Test_v220419_01_Emotion_1by1_260x260", "Test_v220419_01_Emotion_4by3_244x324")


if __name__ == '__main__':
  main()
