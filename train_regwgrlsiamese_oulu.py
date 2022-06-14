import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import DataLoader
from augs.cutmix import cutmix_data, mixup_criterion


from networks import getbasesiameseresnet18wgrl
from lmdbdataset_ouluprotocal import lmdbDatasetwmixupwlimit_oulu, lmdbDatasetwmixupwlimit2_oulu
from utils import AverageMeter, accuracy, Timer, getbasenamewoext, Logger
import os
import shortuuid
from datetime import datetime
from test import testmodel, testsiamesemodel
from shutil import copyfile
import glob


parser = argparse.ArgumentParser(description='anti-spoofing training')
parser.add_argument('--lmdbpath', type=str,
                    default='/home/user/work_db/v4C3/Train_Protocal_4C3_CASIA_MSU_OULU_1by1_260x260.db', help='db path')
parser.add_argument('--ckptpath', type=str,
                    default='/home/user/model_2022/test', help='ckpt path')
parser.add_argument('--epochs', type=int, default=80, help='num of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--GPU', type=int, default=0, help='specify which gpu to use')
parser.add_argument('--works', type=int, default=4, help='works')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.97, help='gamma for scheduler')
parser.add_argument('--opt', type=str, default='adam', help='sgd or adam')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for scheduler')
parser.add_argument('--meta', type=str, default='meta', help='meta')
parser.add_argument('--resume', type=str, default='', help='resume path')
parser.add_argument('--random_seed', type=int, default=20220408, help='random_seed')

args = parser.parse_args()

# random_seed = 20220406 (ours)
random_seed = args.random_seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU)  # Set the GPU 2 to use
struuid = "{}_{}_{}_bsize{}_opt{}_lr{}_gamma_{}_epochs_{}_meta_{}_seed_{}".format(getbasenamewoext(os.path.basename(args.lmdbpath)),
                                                            datetime.now().strftime("%y%m%d"),
                                                            shortuuid.uuid(),
                                                            args.batch_size,
                                                            args.opt,
                                                            args.lr,
                                                            args.gamma,
                                                            args.epochs,
                                                            args.meta,
                                                            args.random_seed)

if args.resume != "":
  print ("resume !!!")
  resumedir = os.path.dirname(args.resume)
  struuid = os.path.basename(resumedir)

strckptpath = os.path.join(args.ckptpath, struuid)
strlogpath = "/home/user/work_2022/logworkspace/{}.log".format(struuid)
logger = Logger(strlogpath)
logger.print(args)

dbprefix = "/home/user/work_db/vOP"
##CASIA-MFSD REPLAY-ATTACK

strinclude = "OULU-NPU"
testdbpath = args.lmdbpath.replace("Train_OULU", "Test_OULU")


def save_ckpt(epoch, net, optimizer):
  if os.path.exists(strckptpath) == False:
    os.makedirs(strckptpath)
  strpath = "{}/epoch_{:02d}.ckpt".format(strckptpath, epoch)
  logger.print ("Save ckpt to {}".format(strpath))
  torch.save({
    'epoch': epoch,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
  }, strpath)

def trainepoch(epoch, trainloader, model, criterion, optimizer, averagemetermap):
  #_t['forward_pass'].tic()
  fbtimer = Timer()
  totaliter = len(trainloader)
  regrsteps = torch.linspace(0, 1.0, steps=11).cuda()
  probsm = nn.Softmax(dim=1)
  for index, (tmpimages, tmplabels, imgpath, rimg, rlab, tmpuid1, tmpuid2) in enumerate(trainloader):
    fbtimer.tic()
    rand_idx = torch.randperm(rimg.shape[0])
    images = torch.cat((tmpimages, rimg[rand_idx,]), dim=0)
    labels = torch.cat((tmplabels, rlab[rand_idx]), dim=0)
    uid1 = torch.cat((tmpuid1, tmpuid2[rand_idx]), dim=0)
    labels = labels.type(torch.FloatTensor)

    ### reg mix
    images, labels = images.cuda(), labels.cuda()
    uid1 = uid1.cuda()

    ### mixup
    miximages, mixlabels = tmpimages.clone().cuda(), tmplabels.clone().cuda()
    mixuid1 = tmpuid1.clone().cuda()
    miximages, mixlabels = cutmix_data(miximages, mixlabels)
    ###

    optimizer.zero_grad()
    #print(images.shape, miximages.shape)
    logit_reg, logit_cls, dislogit_reg, dislogit_cls = model(images, miximages)
    #print(logit_reg.shape, logit_cls.shape, dislogit_reg.shape, dislogit_cls.shape)

    ###
    prob = probsm(logit_reg)
    expectprob = torch.sum(regrsteps * prob, dim=1)
    #print (expectprob.shape, labels.shape)
    mseloss = criterion["mse"](expectprob, labels)
    advregloss = criterion["cls"](dislogit_reg, uid1)
    ###


    #
    mixuploss = mixup_criterion(criterion["cls"], logit_cls, mixlabels[0], mixlabels[1], mixlabels[2]).mean()
    advmixuploss = criterion["cls"](dislogit_cls, mixuid1)
    #

    loss = mseloss + advregloss + mixuploss + advmixuploss

    tmplogit = torch.zeros(images.size(0), 2).cuda()
    tmplogit[:, 1] = expectprob
    tmplogit[:, 0] = 1.0 - tmplogit[:, 1]
    acc = accuracy(tmplogit[0:tmpimages.shape[0],:], labels[0:tmplabels.shape[0]])
    loss.backward()
    optimizer.step()
    averagemetermap["loss_am"].update(loss.item())
    averagemetermap["acc_am"].update(acc[0].item())
    if index % 10 == 0:
      fbtimer.toc()
      strprint = "  {}/{} at {}/{} loss:{:.5f} acc:{:.5f} lr:{:.5f} time:{:.5f}".format(index,
                                                                                     totaliter,
                                                                                     epoch,
                                                                                     args.epochs,
                                                                                     averagemetermap["loss_am"].avg,
                                                                                     averagemetermap["acc_am"].avg,
                                                                                     optimizer.param_groups[0]['lr'],
                                                                                     fbtimer.average_time)

      logger.print (strprint)




def trainmodel():
  """
  """
  averagemetermap = {}
  averagemetermap["loss_am"] = AverageMeter()
  averagemetermap["acc_am"] = AverageMeter()
  epochtimer = Timer()

  mynet = getbasesiameseresnet18wgrl(numdclasses=5)
  mynet = mynet.cuda()

  transforms = T.Compose([T.RandomCrop((256, 256)),
                          T.RandomHorizontalFlip(),
                          T.ToTensor()])  # 0 to 1

  traindataset = lmdbDatasetwmixupwlimit2_oulu(args.lmdbpath, strinclude, transforms)
  # traindataset = lmdbDatasetwmixupwlimit_oulu(args.lmdbpath, strinclude, transforms)

  logger.print(mynet)
  logger.print(traindataset)
  trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=args.works, pin_memory=True)
  criterion = {}
  criterion["cls"] = nn.CrossEntropyLoss().cuda()
  criterion["mse"] = nn.MSELoss().cuda()
  if args.opt.lower() == "adam":
    # works
    optimizer = optim.Adam(mynet.parameters(), lr=args.lr, weight_decay=5e-4)
  else:
    optimizer = optim.SGD(mynet.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=5e-4)
  print (optimizer)
  # https://gaussian37.github.io/dl-pytorch-lr_scheduler/
  # https://sanghyu.tistory.com/113
  # ExponentialLR, LamdaLR same iof gamma is simple
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

  startepoch = 0
  if args.resume != "":
    logger.print("Resume from {}".format(args.resume))
    checkpoint = torch.load(args.resume)
    mynet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    startepoch = checkpoint['epoch'] + 1

  besthter1 = 100.0
  besthter2 = 100.0
  for epoch in range(startepoch, args.epochs):
    mynet.train()
    epochtimer.tic()
    trainepoch(epoch, trainloader, mynet, criterion, optimizer, averagemetermap)
    epochtimer.toc()
    strprint = "{}/{} loss:{:.5f} acc:{:.5f} lr:{:.8f} time:{:.5f}".format(epoch, args.epochs, averagemetermap["loss_am"].avg, averagemetermap["acc_am"].avg, optimizer.param_groups[0]['lr'], epochtimer.average_time)
    logger.print (strprint)
    scheduler.step()
    if averagemetermap["acc_am"].avg > 95.0:#98
      hter1 = testsiamesemodel(epoch, mynet, testdbpath, strckptpath, 8)
      # hter1_1 = testsiamesemodel(epoch, mynet, testdbpath, strckptpath, 1)
      if besthter1 > hter1:
        besthter1 = hter1
        save_ckpt(epoch, mynet, optimizer)
        copyfile(strlogpath, "{}/trainlog_1.txt".format(strckptpath))


if __name__ == '__main__':
  ###
  trainmodel()
  copyfile(strlogpath, "{}/trainlog.txt".format(strckptpath))

  ###





