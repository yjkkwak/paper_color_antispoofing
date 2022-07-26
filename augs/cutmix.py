import numpy as np
import torch


def mixup_criterion(criterion, pred, y_a, y_b, lam):
  return (1.0 - lam) * criterion(pred, y_a) + lam * criterion(pred, y_b)

def rand_bbox(size, lam):
  W = size[2]
  H = size[3]
  cut_rat = np.sqrt(1.0 - lam)
  cut_w = np.int(W * cut_rat)
  cut_h = np.int(H * cut_rat)

  # uniform
  cx = np.random.randint(W)
  cy = np.random.randint(H)

  bbx1 = np.clip(cx - cut_w // 2, 0, W)
  bby1 = np.clip(cy - cut_h // 2, 0, H)
  bbx2 = np.clip(cx + cut_w // 2, 0, W)
  bby2 = np.clip(cy + cut_h // 2, 0, H)

  return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, beta=1.0, half=False):
  lam = np.random.beta(beta, beta)
  batch_size = x.size()[0]
  rand_index = torch.randperm(batch_size).cuda()
  bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), 1 - lam)
  lam = (bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2])
  x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
  y = (y, y[rand_index], (torch.ones(batch_size) * lam).cuda())

  return x, y

def cutmix_data_wocuda(x, y, beta=1.0, half=False):
  lam = np.random.beta(beta, beta)
  batch_size = x.size()[0]
  rand_index = torch.randperm(batch_size)
  bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), 1 - lam)
  lam = (bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2])
  x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
  y = (y, y[rand_index], (torch.ones(batch_size) * lam))

  return x, y

def cutout(x, beta=0.4):
  lam = np.random.beta(beta, beta)
  bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), 1 - lam)
  zerostensor = torch.ones_like(x)
  x[:, :, bbx1:bbx2, bby1:bby2] = zerostensor[:, :, bbx1:bbx2, bby1:bby2]
  return x