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
rho = torch.rand(nx,1)
drho = torch.rand(nx,1,3)
ddrhoTmp = torch.rand(nx,1,9)
ddrhoTmpRS = torch.reshape(ddrhoTmp, (nx,1,3,3))
ddrho  = 0.5*(ddrhoTmpRS + torch.transpose(ddrhoTmpRS, -2, -1))
X = torch.zeros(nx,4)
X[:,0] = rho[:,0]
X[:,1:4] = drho[:,0]
X.requires_grad_()

rhoA = X[:,0]
drhoA = X[:,1:4]
moddrhoA = ((drhoA**2.0).sum(dim=1))**0.5

Y = (rhoA**2.0)*(moddrhoA**2.0)
g =grad(Y,X,torch.ones_like(Y),retain_graph=True, create_graph=True)[0]

vxc = torch.zeros(nx,1)
for iSpin in range(1):
    id1 = 1+3*iSpin
    vxc[:,iSpin] = g[:,iSpin]
    gdrho = g[:,id1:id1+3]
    for jSpin in range(1):
       t1 = grad(gdrho,X, drho[:,jSpin], retain_graph=True)[0]
       vxc[:,iSpin] = vxc[:,iSpin] - t1[:,jSpin]
       t21 = grad(gdrho, X, ddrho[:,jSpin,0], retain_graph=True)[0]
       t22 = grad(gdrho, X, ddrho[:,jSpin,1], retain_graph=True)[0]
       t23 = grad(gdrho, X, ddrho[:,jSpin,2], retain_graph=True)[0]
       id2 = 1+3*jSpin
       vxc[:,iSpin] = vxc[:,iSpin] - (t21[:,id2+0] + t22[:,id2+1] + t23[:,id2+2])


uAA = (drhoA*drhoA).sum(dim=1)
vA = torch.einsum('ijk,ik->ij',ddrho[:,0],drhoA)
vAA = (vA*drhoA).sum(dim=1)
rhoLapA = torch.diagonal(ddrho[:,0], dim1=-2, dim2=-1).sum(dim=-1)
vxcA = 2.0*rhoA*moddrhoA**2.0\
       -4.0*rhoA*uAA\
       -2.0*(rhoA**2.0)*rhoLapA
#vxcA = 2.0*rhoA*(moddrhoA**3.0)*(moddrhoB**2.0) -\
#       3.0*moddrhoA*(moddrhoB**2.0)*(2.0*rhoA*uAA + 3.0*rhoB**2.0*uAB)-\
#       3.0*(rhoA**2.0 + rhoB**3.0)*(moddrhoB**2.0)*vAA/moddrhoA -\
#       6.0*(rhoA**2.0 + rhoB**3.0)*moddrhoA*vBA -\
#       3.0*(rhoA**2.0 + rhoB**3.0)*moddrhoA*(moddrhoB**2.0)*rhoLapA

diffA = vxc[:,0]-vxcA
print(diffA)
