import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import DataLoader

from networks import getmetricresnet18
from mylosses import ArcMarginProduct, AddMarginProduct
from lmdbdataset import lmdbDataset
from utils import AverageMeter, accuracy, Timer, getbasenamewoext, Logger
import os
import shortuuid
from datetime import datetime
from test import testmetricmodel
from shutil import copyfile

random_seed = 20220406
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

parser = argparse.ArgumentParser(description='anti-spoofing training')
parser.add_argument('--lmdbpath', type=str,
                    default='/home/user/work_db/v220419_01/Train_v220419_01_OULUNPU_1by1_260x260.db', help='db path')
parser.add_argument('--ckptpath', type=str,
                    default='/home/user/model_2022/v220513_01/', help='ckpt path')
parser.add_argument('--epochs', type=int, default=80, help='num of epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
parser.add_argument('--GPU', type=int, default=1, help='specify which gpu to use')
parser.add_argument('--works', type=int, default=4, help='works')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.97, help='gamma for scheduler')
parser.add_argument('--opt', type=str, default='sgd', help='sgd or adam')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for scheduler')
parser.add_argument('--meta', type=str, default='meta', help='meta')
parser.add_argument('--resume', type=str, default='', help='resume path')
parser.add_argument('--w1', type=float, default=0.0, help='arc loss weight')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU)  # Set the GPU 2 to use
struuid = "{}_{}_{}_bsize{}_opt{}_lr{}_gamma_{}_epochs_{}_meta_{}".format(getbasenamewoext(os.path.basename(args.lmdbpath)),
                                                            datetime.now().strftime("%y%m%d"),
                                                            shortuuid.uuid(),
                                                            args.batch_size,
                                                            args.opt,
                                                            args.lr,
                                                            args.gamma,
                                                            args.epochs,
                                                            args.meta)

if args.resume != "":
  print ("resume !!!")
  resumedir = os.path.dirname(args.resume)
  struuid = os.path.basename(resumedir)

strckptpath = os.path.join(args.ckptpath, struuid)
strlogpath = "/home/user/work_2022/logworkspace/{}.log".format(struuid)
logger = Logger(strlogpath)
logger.print(args)

dbprefix = "/home/user/work_db/v4C3"

if "CASIA_MSU_OULU" in args.lmdbpath:
  testdbpath = os.path.join(dbprefix, "Test_Protocal_4C3_REPLAY_1by1_260x260.db")
elif "CASIA_MSU_REPLAY" in args.lmdbpath:
  testdbpath = os.path.join(dbprefix, "Test_Protocal_4C3_OULU_1by1_260x260.db")
elif "CASIA_OULU_REPLAY" in args.lmdbpath:
  testdbpath = os.path.join(dbprefix, "Test_Protocal_4C3_MSU_1by1_260x260.db")
elif "MSU_OULU_REPLAY" in args.lmdbpath:
  testdbpath = os.path.join(dbprefix, "Test_Protocal_4C3_CASIA_1by1_260x260.db")


def save_ckpt(epoch, net, metricnet, optimizer):
  if os.path.exists(strckptpath) == False:
    os.makedirs(strckptpath)
  strpath = "{}/epoch_{:02d}.ckpt".format(strckptpath, epoch)
  logger.print ("Save ckpt to {}".format(strpath))
  torch.save({
    'epoch': epoch,
    'model_state_dict': net.state_dict(),
    'metricmodel_state_dict': metricnet.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
  }, strpath)

def trainepoch(epoch, trainloader, model, mymetric, criterion, optimizer, averagemetermap):
  #_t['forward_pass'].tic()
  fbtimer = Timer()
  totaliter = len(trainloader)
  # probsm = nn.Softmax(dim=1)
  for index, (images, labels, imgpath) in enumerate(trainloader):
    fbtimer.tic()
    images, labels = images.cuda(), labels.cuda()
    optimizer.zero_grad()
    logit, fc5 = model(images)
    metricfc5 = mymetric(fc5, labels)
    # prob = probsm(logit)
    logitloss = criterion(logit, labels)
    metircloss = criterion(metricfc5, labels)
    loss = logitloss + args.w1 * metircloss
    acc = accuracy(logit, labels)
    loss.backward()
    optimizer.step()
    averagemetermap["loss_am"].update(loss.item())
    averagemetermap["loss_logit"].update(logitloss.item())
    averagemetermap["loss_metric"].update(metircloss.item())


    averagemetermap["acc_am"].update(acc[0].item())
    if index % 10 == 0:
      fbtimer.toc()
      strprint = "  {}/{} at {}/{} loss:{:.5f} loss_l:{:.5f} loss_m:{:.5f} acc:{:.5f} lr:{:.5f} time:{:.5f}".format(index,
                                                                                     totaliter,
                                                                                     epoch,
                                                                                     args.epochs,
                                                                                     averagemetermap["loss_am"].avg,
                                                                                     averagemetermap["loss_logit"].avg,
                                                                                     averagemetermap["loss_metric"].avg,
                                                                                     averagemetermap["acc_am"].avg,
                                                                                     optimizer.param_groups[0]['lr'],
                                                                                     fbtimer.average_time)

      logger.print (strprint)


def trainmodel():
  """
  """
  averagemetermap = {}
  averagemetermap["loss_am"] = AverageMeter()
  averagemetermap["loss_logit"] = AverageMeter()
  averagemetermap["loss_metric"] = AverageMeter()
  averagemetermap["acc_am"] = AverageMeter()
  epochtimer = Timer()

  mynet = getmetricresnet18()
  mymetric = ArcMarginProduct(64, 2, s=30, m=0.5, easy_margin=False)
  mynet = mynet.cuda()
  mymetric = mymetric.cuda()

  transforms = T.Compose([T.RandomCrop((256, 256)),
                          T.RandomHorizontalFlip(),
                          T.ToTensor()])  # 0 to 1


  traindataset = lmdbDataset(args.lmdbpath, transforms)

  logger.print(mynet)
  logger.print(mymetric)
  logger.print(traindataset)
  trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=args.works, pin_memory=True)
  criterion = nn.CrossEntropyLoss().cuda()
  if args.opt.lower() == "adam":
    optimizer = optim.Adam([{'params': mynet.parameters()}, {'params': mymetric.parameters()}], lr=args.lr,
                           weight_decay=5e-4)
  else:
    # works.
    optimizer = optim.SGD([{'params': mynet.parameters()}, {'params': mymetric.parameters()}], lr=args.lr, momentum=args.momentum,
                          weight_decay=5e-4)
  print(optimizer)
  # https://gaussian37.github.io/dl-pytorch-lr_scheduler/
  # https://sanghyu.tistory.com/113
  # ExponentialLR, LamdaLR same iof gamma is simple
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

  startepoch = 0
  if args.resume != "":
    logger.print("Resume from {}".format(args.resume))
    checkpoint = torch.load(args.resume)
    mynet.load_state_dict(checkpoint['model_state_dict'])
    mymetric.load_state_dict(checkpoint['metricmodel_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    startepoch = checkpoint['epoch'] + 1


  for epoch in range(startepoch, args.epochs):
    mynet.train()
    epochtimer.tic()
    trainepoch(epoch, trainloader, mynet, mymetric, criterion, optimizer, averagemetermap)
    epochtimer.toc()
    strprint = "{}/{} loss:{:.5f} acc:{:.5f} lr:{:.5f} time:{:.5f}".format(epoch, args.epochs, averagemetermap["loss_am"].avg, averagemetermap["acc_am"].avg, optimizer.param_groups[0]['lr'], epochtimer.average_time)
    logger.print (strprint)
    scheduler.step()
    save_ckpt(epoch, mynet, mymetric, optimizer)

    if epoch > 58:
      testmetricmodel(epoch, mynet, testdbpath, strckptpath)


if __name__ == '__main__':
  trainmodel()
  copyfile(strlogpath, "{}/trainlog.txt".format(strckptpath))

