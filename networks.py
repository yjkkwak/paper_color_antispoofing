from utils import Hook, nested_children
# from torchvision import models
from models.myresnet import myresnet18
from models.baseresnet import baseresnet18
from models.metricresnet import metricresnet18, metricpathcnet
from models.baseresnetwgrl import bbaseresnet18wgrl, bbasesiameseresnet18wgrl
import torch
import torch.nn as nn


def getresnet18():
  """
  """
  resnet18 = myresnet18(pretrained=False, num_classes=2)

  return resnet18

def getbaseresnet18(numclasses=2):
  resnet18 = baseresnet18(pretrained=True, numofcls=numclasses)
  return resnet18

def getmetricresnet18():
  resnet18 = metricresnet18(pretrained=False, num_classes=2)
  return resnet18

def getbaseresnet18wgrl(numclasses=11):
  resnet18 = bbaseresnet18wgrl(numclasses)
  return resnet18

def getbasesiameseresnet18wgrl():
  resnet18 = bbasesiameseresnet18wgrl()
  return resnet18

def getpatchnetresnet18():
  resnet18 = metricpathcnet()
  return resnet18

def debugmode():
  rinput = torch.randn((1, 3, 256, 256))
  #mynet = getbaseresnet18(numclasses=11)
  # mynet = getmetricresnet18()
  mynet = getbaseresnet18wgrl()
  print (mynet)
  forwardhook = []
  for l in nested_children(mynet):
    forwardhook.append(Hook(l))
  print (forwardhook)
  logit = mynet(rinput)

  print (logit)

  for hook in forwardhook:
    print (hook.m, hook.input[0].shape, hook.output[0].shape)


  model = mynet
  param_size = 0
  for param in model.parameters():
      param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
      buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))

if __name__ == '__main__':
  debugmode()

