import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import grad
from torch import optim
import math


"""Learning hyperparameters"""
bsize = 10
lr = 1e-12
epochs = 1000
PATH = "./models/1_80_80_80_1_b30_softplus_1e-5"
L = 100

"""Dataset"""
class Dat(Dataset):
    def __init__(self):
        super(Dat).__init__()
        A = np.random.random((L,2))
        self.dat = A
    def __getitem__(self, index):
        return (torch.tensor(self.dat[index, 0], requires_grad=True).unsqueeze(-1),
    torch.tensor(self.dat[index, 1], requires_grad=True).unsqueeze(-1))
    def __len__(self):
        return L


"""Random split of dataset into training and validation set"""
def get_train_valid_sets():
    the_dat = Dat()
    return data.random_split(the_dat, [70, 30])


"""Create data loaders"""
def get_train_valid_loaders(train_set, valid_set):  
    return (
        DataLoader(dataset = train_set, batch_size = bsize, shuffle = True),
        DataLoader(dataset = valid_set, batch_size = bsize, shuffle = False),
    )

train_s, valid_s = get_train_valid_sets()
train_l, valid_l = get_train_valid_loaders(train_s, valid_s)

for rho, vxc in valid_l:
  print(type(rho))
  print(rho.shape)
  exc = rho**(1.0/3.0)
  ones = torch.ones(exc.size(0),1)
  vxc = grad(exc, rho, grad_outputs = ones, create_graph=True)
  print(vxc)
  for i in range(rho.size(0)):
    vxc = grad(exc[i,0], rho, create_graph=True)
    print(vxc)
