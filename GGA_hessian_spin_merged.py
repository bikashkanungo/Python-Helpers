import numpy as np
#from scipy.special import legendre
#from scipy.special import lpmv
import torch
from torch.autograd import grad as grad
from torch.autograd.functional import hessian
import math
import random 
from timeit import default_timer as timer

nx = 5
nspin = 2
rho = torch.rand(nx,nspin)
drho = torch.rand(nx,nspin,3)
ddrhoTmp = torch.rand(nx,nspin,9)
ddrhoTmpRS = torch.reshape(ddrhoTmp, (nx,nspin,3,3))
ddrho = 0.5*(ddrhoTmpRS + torch.transpose(ddrhoTmpRS, -2, -1))
drhoTotal = drho.sum(dim=1)
ddrhoTotal = ddrho.sum(dim=1)
moddrhoTotal = ((drhoTotal**2.0).sum(dim=1))**0.5
u = torch.zeros(nx,nspin)
for iSpin in range(nspin):
    u[:,iSpin] = (drho[:,iSpin]*drhoTotal).sum(dim=1)

v = torch.einsum('ijk,ik->ij',ddrhoTotal,drhoTotal)
w = torch.einsum('ij,ij->i',v,drhoTotal)/moddrhoTotal
rhoLap = torch.diagonal(ddrhoTotal, dim1=-2, dim2=-1).sum(dim=-1)

X = torch.zeros(nx,3)
X[:,0:nspin] = rho
X[:,nspin] = moddrhoTotal 
X.requires_grad_()
#Y = (rhoA**2.0+ rhoB**2.0)*(moddrhoA**2.0 + moddrhoB**2.0)#(rhoA**2.0 + rhoB**3.0)*(moddrhoA**3.0)*(moddrhoB**2.0)
Y = (X[:,0]**2.0 + X[:,1]**3.0)*(X[:,2]**3.0)
g = grad(Y,X,torch.ones_like(Y),retain_graph=True, create_graph=True)[0]

vxc = torch.zeros(nx,nspin)
f = g[:,nspin]
t1 = torch.zeros(nx)
for iSpin in range(nspin):
    t1 = t1 + (grad(f, X, u[:,iSpin], retain_graph=True)[0])[:,iSpin]

t1 = t1 + (grad(f, X, w, retain_graph=True)[0])[:,nspin]
t1 = -1.0*t1/moddrhoTotal
t2 = f*w/(moddrhoTotal**2.0)
t3 = - f*rhoLap/moddrhoTotal
for iSpin in range(nspin):
    vxc[:,iSpin] = g[:,iSpin] + t1 + t2 + t3

z = -3.0*((2.0*rho[:,0]*u[:,0] + 3.0*rho[:,1]**2.0*u[:,1])*moddrhoTotal) +\
          -3.0*(rho[:,0]**2.0 + rho[:,1]**3.0)*w +\
          -3.0*(rho[:,0]**2.0+rho[:,1]**3.0)*moddrhoTotal*rhoLap
vxcA = 2.0*rho[:,0]*moddrhoTotal**3.0 + z 
vxcB = 3.0*rho[:,1]**2.0*moddrhoTotal**3.0 + z 
diffA = vxc[:,0]-vxcA
diffB = vxc[:,1]-vxcB
print(diffA)
print(diffB)
