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
X = torch.zeros(nx,8)
X[:,0:nspin] = rho
for iSpin in range(2):
    index = nspin + 3*iSpin
    X[:,index:index+3] = drho[:,iSpin]

X.requires_grad_()

rhoA = X[:,0]
rhoB = X[:,1]
drhoA = X[:,2:5]
drhoB = X[:,5:8]
moddrhoA = ((drhoA**2.0).sum(dim=1))**0.5
moddrhoB = ((drhoB**2.0).sum(dim=1))**0.5

#Y = (rhoA**2.0+ rhoB**2.0)*(moddrhoA**2.0 + moddrhoB**2.0)#(rhoA**2.0 + rhoB**3.0)*(moddrhoA**3.0)*(moddrhoB**2.0)
Y = (rhoA**2.0 + rhoB**3.0)*(moddrhoA**3.0)*(moddrhoB**2.0)
g = grad(Y,X,torch.ones_like(Y),retain_graph=True, create_graph=True)[0]

vxc = torch.zeros(nx,nspin)
for iSpin in range(nspin):
    id1 = nspin+3*iSpin
    vxc[:,iSpin] = g[:,iSpin]
    gdrho = g[:,id1:id1+3]
    for jSpin in range(nspin):
        id2 = nspin+3*jSpin
        t1 = grad(gdrho,X, drho[:,jSpin], retain_graph=True)[0]
        vxc[:,iSpin] = vxc[:,iSpin] - t1[:,jSpin]
        t21 = grad(gdrho, X, ddrho[:,jSpin,0], retain_graph=True)[0]
        t22 = grad(gdrho, X, ddrho[:,jSpin,1], retain_graph=True)[0]
        t23 = grad(gdrho, X, ddrho[:,jSpin,2], retain_graph=True)[0]
        vxc[:,iSpin] = vxc[:,iSpin] - (t21[:,id2+0] + t22[:,id2+1] + t23[:,id2+2])


uAA = (drhoA*drhoA).sum(dim=1)
uAB = (drhoA*drhoB).sum(dim=1)
uBB = (drhoB*drhoB).sum(dim=1)
vA = torch.einsum('ijk,ik->ij',ddrho[:,0],drhoA)
vB = torch.einsum('ijk,ik->ij',ddrho[:,1],drhoB)
vAA = (vA*drhoA).sum(dim=1)
vAB = (vA*drhoB).sum(dim=1)
vBA = (vB*drhoA).sum(dim=1)
vBB = (vB*drhoB).sum(dim=1)
rhoLapA = torch.diagonal(ddrho[:,0], dim1=-2, dim2=-1).sum(dim=-1)
rhoLapB = torch.diagonal(ddrho[:,1], dim1=-2, dim2=-1).sum(dim=-1)
print(rhoLapA.shape)
#vxcA = 2.0*rhoA*(moddrhoA**2.0 + moddrhoB**2.0) - 4.0*rhoA*uAA\
#       - 4.0*rhoB*uAB -2.0*(rhoA**2.0+rhoB**2.0)*rhoLapA
vxcA = 2.0*rhoA*(moddrhoA**3.0)*(moddrhoB**2.0) -\
       3.0*moddrhoA*(moddrhoB**2.0)*(2.0*rhoA*uAA + 3.0*rhoB**2.0*uAB)-\
       3.0*(rhoA**2.0 + rhoB**3.0)*(moddrhoB**2.0)*vAA/moddrhoA -\
       6.0*(rhoA**2.0 + rhoB**3.0)*moddrhoA*vBA -\
       3.0*(rhoA**2.0 + rhoB**3.0)*moddrhoA*(moddrhoB**2.0)*rhoLapA

diffA = vxc[:,0]-vxcA
print(diffA)
