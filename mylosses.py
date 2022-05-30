from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

#https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
class ArcMarginProduct(nn.Module):
  r"""Implement of large margin arc distance: :
      Args:
          in_features: size of each input sample
          out_features: size of each output sample
          s: norm of input feature
          m: margin
          cos(theta + m)
      """

  def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
    super(ArcMarginProduct, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.s = s
    self.m = m
    self.weight = Parameter(torch.FloatTensor(out_features, in_features))
    nn.init.xavier_uniform_(self.weight)

    self.easy_margin = easy_margin
    self.cos_m = math.cos(m)
    self.sin_m = math.sin(m)
    self.th = math.cos(math.pi - m)
    self.mm = math.sin(math.pi - m) * m

  def forward(self, input, label):
    # --------------------------- cos(theta) & phi(theta) ---------------------------
    cosine = F.linear(F.normalize(input), F.normalize(self.weight))
    sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
    phi = cosine * self.cos_m - sine * self.sin_m
    if self.easy_margin:
      phi = torch.where(cosine > 0, phi, cosine)
    else:
      phi = torch.where(cosine > self.th, phi, cosine - self.mm)
    # --------------------------- convert label to one-hot ---------------------------
    # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
    one_hot = torch.zeros(cosine.size(), device='cuda')
    one_hot.scatter_(1, label.view(-1, 1).long(), 1)
    # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
    output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
    output *= self.s
    # print(output)

    return output

class AsymArcMarginProduct(nn.Module):
  r"""Implement of large margin arc distance: :
      Args:
          in_features: size of each input sample
          out_features: size of each output sample
          s: norm of input feature
          m: margin
          cos(theta + m)
      """

  def __init__(self, in_features, out_features, s=30.0, m1=0.50, m2=0.50, easy_margin=False):
    super(AsymArcMarginProduct, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.s = s
    self.m1 = m1
    self.weight = Parameter(torch.FloatTensor(out_features, in_features))
    nn.init.xavier_uniform_(self.weight)

    self.easy_margin = easy_margin
    self.cos_m1 = math.cos(m1)
    self.sin_m1 = math.sin(m1)
    self.th1 = math.cos(math.pi - m1)
    self.mm1 = math.sin(math.pi - m1) * m1

    self.cos_m2 = math.cos(m2)
    self.sin_m2 = math.sin(m2)
    self.th2 = math.cos(math.pi - m2)
    self.mm2 = math.sin(math.pi - m2) * m2

  def forward(self, input, label):
    # --------------------------- cos(theta) & phi(theta) ---------------------------
    cosine = F.linear(F.normalize(input), F.normalize(self.weight))
    print (cosine.shape)
    print (label)

    ### check what if label is one side
    m1label = torch.where(label == 1)[0]
    m2label = torch.where(label == 0)[0]
    print(m1label)
    print(m2label)
    sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
    phi1 = cosine * self.cos_m1 - sine * self.sin_m1
    phi2 = cosine * self.cos_m2 - sine * self.sin_m2
    print (phi1.shape)
    print (phi1)
    phi1[m1label, :] = torch.where(cosine[m1label, :] > self.th1, phi1[m1label, :], cosine[m1label, :] - self.mm1)
    phi2[m2label, :] = torch.where(cosine[m2label, :] > self.th2, phi1[m2label, :], cosine[m2label, :] - self.mm2)
    print(phi1)
    print(phi1.shape)
    # --------------------------- convert label to one-hot ---------------------------
    # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
    #one_hot = torch.zeros(cosine.size(), device='cuda')
    one_hot = torch.zeros(cosine.size())
    one_hot.scatter_(1, label.view(-1, 1).long(), 1)
    # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
    output = (one_hot * phi1) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
    output[m2label, :] = (one_hot[m2label, :] * phi2[m2label, :]) + ((1.0 - one_hot[m2label, :]) * cosine[m2label, :])  # you can use torch.where if your torch.__version__ is 0.4
    print (output.shape)


    output *= self.s
    # print(output)

    return output


class AddMarginProduct(nn.Module):
  r"""Implement of large margin cosine distance: :
  Args:
      in_features: size of each input sample
      out_features: size of each output sample
      s: norm of input feature
      m: margin
      cos(theta) - m
  """

  def __init__(self, in_features, out_features, s=30.0, m=0.40):
    super(AddMarginProduct, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.s = s
    self.m = m
    self.weight = Parameter(torch.FloatTensor(out_features, in_features))
    nn.init.xavier_uniform_(self.weight)

  def forward(self, input, label):
    # --------------------------- cos(theta) & phi(theta) ---------------------------
    cosine = F.linear(F.normalize(input), F.normalize(self.weight))
    phi = cosine - self.m
    # --------------------------- convert label to one-hot ---------------------------
    one_hot = torch.zeros(cosine.size(), device='cuda')
    # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
    one_hot.scatter_(1, label.view(-1, 1).long(), 1)
    # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
    output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
    output *= self.s
    # print(output)

    return output

  def __repr__(self):
    return self.__class__.__name__ + '(' \
           + 'in_features=' + str(self.in_features) \
           + ', out_features=' + str(self.out_features) \
           + ', s=' + str(self.s) \
           + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
  r"""Implement of large margin cosine distance: :
  Args:
      in_features: size of each input sample
      out_features: size of each output sample
      m: margin
      cos(m*theta)
  """

  def __init__(self, in_features, out_features, m=4):
    super(SphereProduct, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.m = m
    self.base = 1000.0
    self.gamma = 0.12
    self.power = 1
    self.LambdaMin = 5.0
    self.iter = 0
    self.weight = Parameter(torch.FloatTensor(out_features, in_features))
    nn.init.xavier_uniform(self.weight)

    # duplication formula
    self.mlambda = [
      lambda x: x ** 0,
      lambda x: x ** 1,
      lambda x: 2 * x ** 2 - 1,
      lambda x: 4 * x ** 3 - 3 * x,
      lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
      lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
    ]

  def forward(self, input, label):
    # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
    self.iter += 1
    self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

    # --------------------------- cos(theta) & phi(theta) ---------------------------
    cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
    cos_theta = cos_theta.clamp(-1, 1)
    cos_m_theta = self.mlambda[self.m](cos_theta)
    theta = cos_theta.data.acos()
    k = (self.m * theta / 3.14159265).floor()
    phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
    NormOfFeature = torch.norm(input, 2, 1)

    # --------------------------- convert label to one-hot ---------------------------
    one_hot = torch.zeros(cos_theta.size())
    one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
    one_hot.scatter_(1, label.view(-1, 1), 1)

    # --------------------------- Calculate output ---------------------------
    output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
    output *= NormOfFeature.view(-1, 1)

    return output

  def __repr__(self):
    return self.__class__.__name__ + '(' \
           + 'in_features=' + str(self.in_features) \
           + ', out_features=' + str(self.out_features) \
           + ', m=' + str(self.m) + ')'


def debugloss():
  mymetric = AsymArcMarginProduct(10, 2, s=30, m1=0.4, m2=0.1, easy_margin=False)
  logit = torch.randn((4, 10))
  label = torch.ones(4)
  rand_idx = torch.randperm(label.shape[0])
  label[rand_idx[0:2]] = 0
  print (label, rand_idx)
  lll = mymetric(logit, label)
  print (lll)


if __name__ == '__main__':
  debugloss()