import glob
import numpy as np
import torch
import time
import os
from scipy import interpolate

class Hook():
  def __init__(self, module, backward=False):
    if backward == False:
      self.hook = module.register_forward_hook(self.hook_fn)
    else:
      self.hook = module.register_backward_hook(self.hook_fn)

  def hook_fn(self, module, input, output):
    self.m = module
    self.input = input
    self.output = output

  def close(self):
    self.hook.remove()


def nested_children(model: torch.nn.Module):
  children = list(model.children())
  flatt_children = []
  if children == []:
    # if model has no children; model is last child! :O
    return model
  else:
    # look for children from children... to the last child!
    for child in children:
      try:
        flatt_children.extend(nested_children(child))
      except TypeError:
        flatt_children.append(nested_children(child))
  return flatt_children


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

#https://github.com/bearpaw/pytorch-classification/blob/master/utils/logger.py
class Logger(object):
  '''Save training process to log file with simple plot function.'''
  def __init__(self, fpath):
    if os.path.exists(fpath):
      self.file = open(fpath, "a")
    else:
      self.file = open(fpath, "w")

  def print(self, strlog):
    self.file.write("{}\n".format(strlog))
    #self.file.flush()
    print(strlog)

  def close(self):
    if self.file is not None:
      self.file.close()

class Timer(object):
  """A simple timer."""

  def __init__(self):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.average_time = 0.

  def tic(self):
    # using time.time instead of time.clock because time time.clock
    # does not normalize for multithreading
    self.start_time = time.time()

  def toc(self, average=True):
    self.diff = time.time() - self.start_time
    self.total_time += self.diff
    self.calls += 1
    self.average_time = self.total_time / self.calls
    if average:
      return self.average_time
    else:
      return self.diff

  def clear(self):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.average_time = 0.

def getbasenamewoext(srcfile):
  pathname, extension = os.path.splitext(srcfile)
  return pathname

def readscore(scorefile):
  the_file = open(scorefile, "r")
  strlines = the_file.readlines()
  scorelist = []
  for strline in strlines:
    strtokens = strline.split()
    label = 0
    if "live" in strtokens[2] or "real" in strtokens[2] or "emotion" in strtokens[2]:
      label = 1
    scorelist.append([float(strtokens[0]), float(strtokens[1]), label])
  the_file.close()
  npscore = np.array(scorelist)
  return npscore

def genfarfrreerwth(scorefile, thre):
  print(scorefile)
  # https://cambridge-archive.blogspot.com/2014/04/frr-far-tpr-fpr-roc-curve-acc-spc-ppv.html
  npscore = readscore(scorefile)
  lb = npscore[:, 2]
  livelb = np.where(npscore[:, 2] == 1.0)[0]
  fakelb = np.where(npscore[:, 2] == 0.0)[0]
  """"""

  class_in = npscore[livelb]
  class_out = npscore[fakelb]

  tmpin = np.where(class_in[:, 1] > 0.5)[0]
  tmpout = np.where(class_out[:, 1] < 0.5)[0]
  acc = (len(tmpin) + len(tmpout)) / len(lb)


  frr = np.sum(class_in[:, 1] < thre) / len(livelb)
  far = np.sum(class_out[:, 1] > thre) / len(class_out)
  tpr = 1.0 - frr
  eer = (frr + far) / 2.0

  print("acc {:0.2} / tpr {:0.2} at far {:0.2} / eer {:0.2}".format(acc, tpr, far, eer, thre))


def genfarfrreerwthlist(scorefile):
  print(scorefile)
  # https://cambridge-archive.blogspot.com/2014/04/frr-far-tpr-fpr-roc-curve-acc-spc-ppv.html
  npscore = readscore(scorefile)
  lb = npscore[:, 2]
  livelb = np.where(npscore[:, 2] == 1.0)[0]
  fakelb = np.where(npscore[:, 2] == 0.0)[0]
  """"""

  class_in = npscore[livelb]
  class_out = npscore[fakelb]

  tmpin = np.where(class_in[:, 1] > 0.5)[0]
  tmpout = np.where(class_out[:, 1] < 0.5)[0]
  acc = (len(tmpin) + len(tmpout)) / len(lb)

  thre = np.arange(0.1, 1.0, 0.1)
  for i in range(len(thre)):
    frr = np.sum(class_in[:, 1] < thre[i]) / len(livelb)
    far = np.sum(class_out[:, 1] > thre[i]) / len(class_out)
    tpr = 1.0 - frr
    eer = (frr + far) / 2.0
    print("acc {:0.2} / tpr {:0.2} at far {:0.2} / eer {:0.2} thr {:0.2}".format(acc, tpr, far, eer, thre[i]))

def gentprwonlylive(scorefile):
  # https://cambridge-archive.blogspot.com/2014/04/frr-far-tpr-fpr-roc-curve-acc-spc-ppv.html
  npscore = readscore(scorefile)
  lb = npscore[:, 2]
  livelb = np.where(npscore[:, 2] == 1.0)[0]

  """"""
  TPR = []  # TPR = 1 - FRR
  THR = []

  thre = np.arange(0.1, 1.0, 0.001)  # Generate an arithmetic list of model thresholds

  class_in = npscore[livelb]

  for i in range(len(thre)):
    frr = np.sum(class_in[:, 1] < thre[i]) / len(livelb)
    tpr = 1.0 - frr
    TPR.append(tpr)
    THR.append(thre[i])

  inter_tprwthr = interpolate.interp1d(THR, TPR, fill_value='extrapolate')
  tmpin = np.where(class_in[:, 1] > 0.5)[0]
  acc = (len(tmpin)) / len(lb)
  streval = "_acc_{:0>5.2f}_tprthr_{:0>5.2f}_0.7_{:0>5.2f}_0.8".format(acc * 100,
    inter_tprwthr(0.7) * 100,
    inter_tprwthr(0.8) * 100)

  strevalpath = "{}{}.eval".format(scorefile, streval)
  the_file = open(strevalpath, "w")
  the_file.close()
  print(strevalpath)
  ##############################


def genfarfrreer(scorefile):
  # https://cambridge-archive.blogspot.com/2014/04/frr-far-tpr-fpr-roc-curve-acc-spc-ppv.html
  npscore = readscore(scorefile)
  lb = npscore[:,2]
  livelb = np.where(npscore[:,2] == 1.0)[0]
  fakelb = np.where(npscore[:, 2] == 0.0)[0]

  """"""
  TPR = [] # TPR = 1 - FRR
  FRR = [] # FRR = 1 - TPR
  FAR = []
  EER = [] # (FAR+FRR) / 2
  THR = []

  thre = np.arange(0.1, 1.0, 0.001)  # Generate an arithmetic list of model thresholds

  class_in = npscore[livelb]
  class_out = npscore[fakelb]

  tmpin = np.where(class_in[:, 1] > 0.5)[0]
  tmpout = np.where(class_out[:, 1] < 0.5)[0]
  acc = (len(tmpin) + len(tmpout)) / len(lb)

  # print (thre)
  for i in range(len(thre)):
    frr = np.sum(class_in[:, 1] < thre[i]) / len(livelb)
    far = np.sum(class_out[:, 1] > thre[i]) / len(class_out)
    tpr = 1.0 - frr
    eer = (frr+far)/2.0
    FRR.append(frr)
    TPR.append(tpr)
    FAR.append(far)
    EER.append(eer)
    THR.append(thre[i])
    # print ("ACC {:0.5} TPR {:0.5} / FAR {:0.5} / EER {:0.5} th {:0.5}".format(acc, tpr, far, eer, thre[i]))


  # interpolation.. soon
  inter_tprwfar = interpolate.interp1d(FAR, TPR, fill_value='extrapolate')
  inter_tprwthr = interpolate.interp1d(THR, TPR, fill_value='extrapolate')
  inter_eerwthr = interpolate.interp1d(THR, EER, fill_value='extrapolate')

  #https://stackoverflow.com/questions/28845514/how-can-i-format-a-float-with-given-precision-and-zero-padding
  streval = "_acc_{:0>5.2f}_tprfar_{:0>5.2f}_{:0>5.2f}_tpreerthr_{:0>5.2f}_{:0>5.2f}_0.7_{:0>5.2f}_{:0>5.2f}_0.8".format(acc*100,
                                                                                               inter_tprwfar(0.001)*100,
                                                                                               inter_tprwfar(0.0001)*100,
                                                                                               inter_tprwthr(0.7)*100,
                                                                                               inter_eerwthr(0.7)*100,
                                                                                               inter_tprwthr(0.8)*100,
                                                                                               inter_eerwthr(0.8)*100)

  strevalpath = "{}{}.eval".format(scorefile, streval)
  the_file = open(strevalpath, "w")
  thre = np.arange(0.1, 1.0, 0.1)
  for i in range(len(thre)):
    frr = np.sum(class_in[:, 1] < thre[i]) / len(livelb)
    far = np.sum(class_out[:, 1] > thre[i]) / len(class_out)
    tpr = 1.0 - frr
    eer = (frr+far)/2.0
    the_file.write("ACC {:0.5} TPR {:0.5} / FAR {:0.5} / EER {:0.5} th {:0.5}\n".format(acc, tpr, far, eer, thre[i]))
  the_file.close()

  print (strevalpath)


def genstatistics(evalpath, lastk=10):
  evallist = glob.glob("{}/**/*.eval".format(evalpath), recursive=True)
  testdblist = {}
  for evalfile in evallist:
    strtokens = evalfile.split("/")
    # print (strtokens)
    scorename = os.path.basename(evalfile)
    epoch = int(scorename.split(".")[0])
    if int(epoch) < 69:
      continue

    if strtokens[6] not in testdblist.keys():
      testdblist[strtokens[6]] = [scorename]
    else:
      testdblist[strtokens[6]].append(scorename)

  for keys in testdblist.keys():
    scorelist = []
    for scorename in testdblist[keys]:
      if "inf" in scorename or "nan" in scorename:
        # print ("Error", scorename)
        continue
      scoreitems = scorename.split("_")
      # print (keys, scorename, scorename.split("_"))
      # print ((scoreitems[2], scoreitems[4], scoreitems[5], scoreitems[7],scoreitems[8], scoreitems[10],scoreitems[11]))
      if "Emotion" in keys:
        # print (keys, scorename, scorename.split("_"))
        scorelist.append((scoreitems[2], scoreitems[4], scoreitems[6]))
      else:
        scorelist.append((scoreitems[2], scoreitems[4], scoreitems[5], scoreitems[7],scoreitems[8], scoreitems[10],scoreitems[11]))
      # break
    npscorelist = np.array(scorelist,dtype=np.float32)
    # print (npscorelist)
    npavgscore = np.average(npscorelist, axis=0)
    testdblist[keys] = npavgscore

  for keys in testdblist.keys():
    string_list = ["{:0>5.2f}".format(value) for value in testdblist[keys]]
    print (keys, *string_list)

  print ("")


if __name__ == '__main__':
  print ("abc")
  # scorelist = glob.glob("/home/user/model_2022/*/**/*.score", recursive=True)
  # for strscorepath in scorelist:
  #   genfarfrreer(strscorepath)
  #

  # genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_SiW_LDRGB_1by1_260x260_220407_G8P7CnJd3L2kzUFcSrYBsM/")
  # genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_SiW_LD3007_1by1_260x260_220407_XBssHFFicQhFCyuVeNEcjY/")
  # genstatistics("/home/user/model_2022/Train_v220401_01_SiW_LDRGB_LD3007_1by1_260x260_220407_P9WfLogyuHEHgks5NSLn8K/")
  # genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_LDRGB_LD3007_1by1_260x260_220407_CvdQ9GdhnUmD2xD9Fi3ybS")

# ##
#   genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_SiW_LDRGB_1by1_260x260_220408_6rAWNVCK3S72YS9svYtUrC_lr0.01_gamma_0.92/")
#   genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_SiW_LD3007_1by1_260x260_220408_eSCEpkTvFk8aYoTn6RvgKt_lr0.01_gamma_0.92/")
#   genstatistics("/home/user/model_2022/Train_v220401_01_SiW_LDRGB_LD3007_1by1_260x260_220408_6ihoYoTHFkRdiLQcin3CNL_lr0.01_gamma_0.92/")
#   genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_LDRGB_LD3007_1by1_260x260_220408_nB8VRKkTsBdxgUVeKQyqcD_lr0.01_gamma_0.92/")


  # genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_SiW_LDRGB_1by1_260x260_220413_HF69LebabMp5vPno9Z7hqY_lr0.01_gamma_0.92_epochs_80_meta_3264128/")
  # genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_SiW_LD3007_1by1_260x260_220413_6C72t2VtmEMWmwfZmpcUW8_lr0.01_gamma_0.92_epochs_80_meta_3264128/")
  # genstatistics("/home/user/model_2022/Train_v220401_01_SiW_LDRGB_LD3007_1by1_260x260_220413_3vUK95H6k9tBptQBNLhcGL_lr0.01_gamma_0.92_epochs_80_meta_3264128/")
  # genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_LDRGB_LD3007_1by1_260x260_220413_gfTWR9vRpQ9GymiJdgXjkn_lr0.01_gamma_0.92_epochs_80_meta_3264128/")


##
  # genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_SiW_LDRGB_4by3_244x324_220412_YEYEhPmvRPogm2fnjvJvHR_lr0.01_gamma_0.92/")
  # genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_SiW_LD3007_4by3_244x324_220412_krSNtfNzMMAqdMSeM4kmn4_lr0.01_gamma_0.92/")
  # genstatistics("/home/user/model_2022/Train_v220401_01_SiW_LDRGB_LD3007_4by3_244x324_220411_ARtFWpc23jBnAaSMLSc55h_lr0.01_gamma_0.92/")
  # genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_LDRGB_LD3007_4by3_244x324_220412_eMedm6ND2sMEnU9wexbjCq_lr0.01_gamma_0.92/")

##
  # genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_SiW_LDRGB_1by1_260x260_220414_dnNzDP4dF5PX9gfrfGg3dq_lr0.01_gamma_0.92_epochs_80_meta_163264/")
  # genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_SiW_LD3007_1by1_260x260_220414_gXyb3RBNDU7pPVQjwt7KUw_lr0.01_gamma_0.92_epochs_80_meta_163264/")
  # genstatistics("/home/user/model_2022/Train_v220401_01_SiW_LDRGB_LD3007_1by1_260x260_220414_HDZCuMsB2eriabbcwYkRC5_lr0.01_gamma_0.92_epochs_80_meta_163264/")
  # genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_LDRGB_LD3007_1by1_260x260_220414_Rc3qebxDSvyocY4uCgcEKb_lr0.01_gamma_0.92_epochs_80_meta_163264/")

###
  # genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_SiW_LDRGB_4by3_244x324_220415_deXTJvMBCriiGtKuGkuA7Q_lr0.01_gamma_0.92_epochs_80_meta_163264/")
  # genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_SiW_LD3007_4by3_244x324_220415_3FyDQrrgwjuRnD29YTPL4A_lr0.01_gamma_0.92_epochs_80_meta_163264/")
  # genstatistics("/home/user/model_2022/Train_v220401_01_SiW_LDRGB_LD3007_4by3_244x324_220415_9JK2EGmzAk4hgnseEPZ8Ck_lr0.01_gamma_0.92_epochs_80_meta_163264/")
  # genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_LDRGB_LD3007_4by3_244x324_220415_4RnkqoXCLC7kjswpQ7jU5j_lr0.01_gamma_0.92_epochs_80_meta_163264/")


# genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_SiW_LDRGB_1by1_260x260_220418_hspGNeBc4gLmn6yrXuYynk_lr0.01_gamma_0.92_epochs_80_meta_81632/")
# genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_SiW_LD3007_1by1_260x260_220418_VWF883s4ZqhXHWcJMf2vSG_lr0.01_gamma_0.92_epochs_80_meta_81632/")
# genstatistics("/home/user/model_2022/Train_v220401_01_SiW_LDRGB_LD3007_1by1_260x260_220418_ZNd2zCj4LoDiURkSDWVfAC_lr0.01_gamma_0.92_epochs_80_meta_81632/")
# genstatistics("/home/user/model_2022/Train_v220401_01_CelebA_LDRGB_LD3007_1by1_260x260_220418_gFTGsedAJb3AHLNRvdd3QT_lr0.01_gamma_0.92_epochs_80_meta_81632/")


  # genstatistics("/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_1by1_260x260_220430_fDUqAt85BiRUwsYZA9KjX6_lr0.01_gamma_0.92_epochs_80_meta_163264/")
  # genstatistics("/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LD3007_1by1_260x260_220430_AgzaX9dW69E5Kd5pshJWhi_lr0.01_gamma_0.92_epochs_80_meta_163264/")
  # genstatistics("/home/user/model_2022/v220419_01/Train_v220419_01_SiW_LDRGB_LD3007_1by1_260x260_220430_K5hNGp5vRZR9v7tPXHz4xf_lr0.01_gamma_0.92_epochs_80_meta_163264/")
  # genstatistics("/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_LDRGB_LD3007_1by1_260x260_220430_DyM4Gfzo8FGqsXmLzJPudN_lr0.01_gamma_0.92_epochs_80_meta_163264/")


  # genstatistics("/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_1by1_260x260_220502_dV2jCRRuqv6mLnEbgSXFxr_lr0.01_gamma_0.92_epochs_80_meta_163264/")
  # genstatistics("/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_1by1_260x260_220502_3uKCX7S9pwbeSTzoTydcgV_lr0.005_gamma_0.92_epochs_80_meta_163264")
  # genstatistics("/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_1by1_260x260_220502_ASjbjf4YXyMBocMLJEFhbS_lr0.01_gamma_0.92_epochs_80_meta_baselineres18/")
  genstatistics("/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_1by1_260x260_220502_Pn2ww7BGgZGmhJD5oeG2L6_lr0.005_gamma_0.92_epochs_80_meta_baselineres18")