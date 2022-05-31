from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import math

def ssan_performances_val(score_val_filename):
  with open(score_val_filename, 'r') as file:
    lines = file.readlines()
  val_scores = []
  val_labels = []
  data = []
  count = 0.0
  num_real = 0.0
  num_fake = 0.0
  for line in lines:
    try:
      count += 1
      tokens = line.split()
      label = float(tokens[0])  # label
      score = float(tokens[2])  # live socre
      val_scores.append(score)
      val_labels.append(label)
      data.append({'map_score': score, 'label': label})
      if label == 1:
        num_real += 1
      else:
        num_fake += 1
    except:
      continue


  fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)

  auc_test = auc(fpr, tpr)
  val_err, val_threshold, right_index = get_err_threhold(fpr, tpr, threshold)
  best_thr = val_threshold
  type1 = len([s for s in data if s['map_score'] < val_threshold and s['label'] == 1])
  type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

  val_ACC = 1 - (type1 + type2) / count

  FRR = 1 - tpr  # FRR = 1 - TPR

  HTER = (fpr + FRR) / 2.0  # error recognition rate &  reject recognition rate


  streval = "_acc_{:0>5.6f}_fpr_{:0>5.6f}_frr_{:0>5.6f}_hter_{:0>5.6f}_auc_{:0>5.6f}_thr_{:0>5.6f}".format(
    val_ACC * 100,
    fpr[right_index] * 100,
    FRR[right_index] * 100,
    HTER[right_index] * 100,
    auc_test * 100,
    best_thr)

  strevalpath = "{}{}.eval".format(score_val_filename, streval)
  # print (strevalpath)
  the_file = open(strevalpath, "w")
  the_file.close()

  return val_ACC, fpr[right_index], FRR[right_index], HTER[right_index], auc_test, best_thr


def get_err_threhold(fpr, tpr, threshold):
  differ_tpr_fpr_1 = tpr + fpr - 1.0
  right_index = np.argmin(np.abs(differ_tpr_fpr_1))
  best_th = threshold[right_index]
  err = fpr[right_index]
  return err, best_th, right_index


def performances_tpr_fpr(map_score_val_filename):
  with open(map_score_val_filename, 'r') as file:
    lines = file.readlines()
  scores = []
  labels = []
  for line in lines:
    try:
      record = line.split()
      scores.append(float(record[0]))
      labels.append(float(record[1]))
    except:
      continue

  fpr_list = [0.1, 0.01, 0.001, 0.0001]
  threshold_list = get_thresholdtable_from_fpr(scores, labels, fpr_list)
  tpr_list = get_tpr_from_threshold(scores, labels, threshold_list)
  return tpr_list


def get_thresholdtable_from_fpr(scores, labels, fpr_list):
  threshold_list = []
  live_scores = []
  for score, label in zip(scores, labels):
    if label == 1:
      live_scores.append(float(score))
  live_scores.sort()
  live_nums = len(live_scores)
  for fpr in fpr_list:
    i_sample = int(fpr * live_nums)
    i_sample = max(1, i_sample)
    if not live_scores:
      return [0.5] * 10
    threshold_list.append(live_scores[i_sample - 1])
  return threshold_list


# Get the threshold under thresholds
def get_tpr_from_threshold(scores, labels, threshold_list):
  tpr_list = []
  hack_scores = []
  for score, label in zip(scores, labels):
    if label == 0:
      hack_scores.append(float(score))
  hack_scores.sort()
  hack_nums = len(hack_scores)
  for threshold in threshold_list:
    hack_index = 0
    while hack_index < hack_nums:
      if hack_scores[hack_index] >= threshold:
        break
      else:
        hack_index += 1
    if hack_nums != 0:
      tpr = hack_index * 1.0 / hack_nums
    else:
      tpr = 0
    tpr_list.append(tpr)
  return tpr_list


def ssdg_performacne_val(score_val_filename):
  with open(score_val_filename, 'r') as file:
    lines = file.readlines()
  prob_list = []
  label_list = []
  data = []
  count = 0.0
  num_real = 0.0
  num_fake = 0.0
  for line in lines:
    try:
      count += 1
      tokens = line.split()
      label = 0
      if "/real/" in tokens[2]:
        label = 1

      score = float(tokens[1])  # live socre
      prob_list.append(score)
      label_list.append(label)
      data.append({'map_score': score, 'label': label})
      if label == 1:
        num_real += 1
      else:
        num_fake += 1
    except:
      continue
  prob_list = np.array(prob_list)
  label_list = np.array(label_list)


  auc_score = roc_auc_score(label_list, prob_list)
  cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
  ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
  cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)

  return cur_EER_valid, cur_HTER_valid, auc_score, threshold, ACC_threshold * 100

def eval_state(probs, labels, thr):
  predict = probs >= thr
  TN = np.sum((labels == 0) & (predict == False))
  FN = np.sum((labels == 1) & (predict == False))
  FP = np.sum((labels == 0) & (predict == True))
  TP = np.sum((labels == 1) & (predict == True))
  return TN, FN, FP, TP


def calculate_threshold(probs, labels, threshold):
  TN, FN, FP, TP = eval_state(probs, labels, threshold)
  ACC = (TP + TN) / labels.shape[0]
  return ACC

def get_threshold(probs, grid_density):
  Min, Max = min(probs), max(probs)
  thresholds = []
  for i in range(grid_density + 1):
    thresholds.append(0.0 + i * 1.0 / float(grid_density))
  thresholds.append(1.1)
  return thresholds

def get_EER_states(probs, labels, grid_density=10000):
  thresholds = get_threshold(probs, grid_density)
  min_dist = 1.0
  min_dist_states = []
  FRR_list = []
  FAR_list = []
  for thr in thresholds:
    TN, FN, FP, TP = eval_state(probs, labels, thr)
    if (FN + TP == 0):
      FRR = TPR = 1.0
      FAR = FP / float(FP + TN)
      TNR = TN / float(TN + FP)
    elif (FP + TN == 0):
      TNR = FAR = 1.0
      FRR = FN / float(FN + TP)
      TPR = TP / float(TP + FN)
    else:
      FAR = FP / float(FP + TN)
      FRR = FN / float(FN + TP)
      TNR = TN / float(TN + FP)
      TPR = TP / float(TP + FN)
    dist = math.fabs(FRR - FAR)
    FAR_list.append(FAR)
    FRR_list.append(FRR)
    if dist <= min_dist:
      min_dist = dist
      min_dist_states = [FAR, FRR, thr]
  EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
  thr = min_dist_states[2]
  return EER, thr, FRR_list, FAR_list

def get_HTER_at_thr(probs, labels, thr):
  TN, FN, FP, TP = eval_state(probs, labels, thr)
  if (FN + TP == 0):
    FRR = 1.0
    FAR = FP / float(FP + TN)
  elif (FP + TN == 0):
    FAR = 1.0
    FRR = FN / float(FN + TP)
  else:
    FAR = FP / float(FP + TN)
    FRR = FN / float(FN + TP)
  HTER = (FAR + FRR) / 2.0
  return HTER


if __name__ == '__main__':
  # acc, fpr, frr, hter, auc, best_thr = ssan_performances_val(
  #   "/home/user/model_2022/v220513_02/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_1by1_260x260_220525_8LJN426mjCq5E3S6XQxR8C_bsize512_optsgd_lr0.005_gamma_0.92_epochs_81_meta_arcloss163264_w1_1.0_SGD/Dev_v220419_01_OULUNPU_1by1_260x260/78.score")
  # print(acc, fpr, frr, hter, auc, best_thr)

  ssan_performances_val("/home/user/model_2022/v4C3/Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260_220530_QDVJ3zhtjCMDBRf8SaUR4Y_bsize128_optadam_lr0.0001_gamma_0.9_epochs_40_meta_clsloss_resnet18_adam/Test_Protocal_4C3_REPLAY_1by1_260x260/01.score")

  # cur_EER_valid, cur_HTER_valid, auc_score, threshold, ACC_threshold = ssdg_performacne_val("/home/user/model_2022/v220513_02/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_1by1_260x260_220525_8LJN426mjCq5E3S6XQxR8C_bsize512_optsgd_lr0.005_gamma_0.92_epochs_81_meta_arcloss163264_w1_1.0_SGD/Dev_v220419_01_OULUNPU_1by1_260x260/78.score")
  # print (cur_EER_valid, cur_HTER_valid, auc_score, threshold, ACC_threshold)
