import numpy as np
import math
import random
from math import cos
from math import sin

def R(u,v,w):
   R = np.zeros((3,3))
   R[0,0] = cos(v)*cos(w)
   R[0,1] = sin(u)*sin(v)*cos(w) - cos(u)*sin(w)
   R[0,2] = sin(u)*sin(w) + cos(u)*sin(v)*cos(w)
   R[1,0] = cos(v)*sin(w)
   R[1,1] = cos(u)*cos(w) + sin(u)*sin(v)*sin(w)
   R[1,2] = cos(u)*sin(v)*sin(w) - sin(u)*cos(w)
   R[2,0] = -sin(v)
   R[2,1] = sin(u)*cos(v)
   R[2,2] = cos(u)*cos(v)
   return R

def Rx(u):
   R = np.zeros((3,3))
   R[0,0] = 1.0
   R[0,1] = 0.0
   R[0,2] = 0.0
   R[1,0] = 0.0
   R[1,1] = cos(u)
   R[1,2] = -sin(u)
   R[2,0] = 0.0
   R[2,1] = sin(u)
   R[2,2] = cos(u)
   return R


def Ry(v):
   R = np.zeros((3,3))
   R[0,0] = cos(v)
   R[0,1] = 0.0
   R[0,2] = sin(v)
   R[1,0] = 0.0
   R[1,1] = 1.0
   R[1,2] = 0.0
   R[2,0] = -sin(v)
   R[2,1] = 0.0
   R[2,2] = cos(v)
   return R

def Rz(w):
   R = np.zeros((3,3))
   R[0,0] = cos(w)
   R[0,1] = -sin(w)
   R[0,2] = 0.0
   R[1,0] = sin(w)
   R[1,1] = cos(w)
   R[1,2] = 0.0
   R[2,0] = 0.0
   R[2,1] = 0.0
   R[2,2] = 1.0
   return R

def Rxu(u):
   R = np.zeros((3,3))
   R[0,0] = 0.0
   R[0,1] = 0.0
   R[0,2] = 0.0
   R[1,0] = 0.0
   R[1,1] = -sin(u)
   R[1,2] = -cos(u)
   R[2,0] = 0.0
   R[2,1] = cos(u)
   R[2,2] = -sin(u)
   return R

def Ryv(v):
   R = np.zeros((3,3))
   R[0,0] = -sin(v)
   R[0,1] = 0.0
   R[0,2] = cos(v)
   R[1,0] = 0.0
   R[1,1] = 0.0
   R[1,2] = 0.0
   R[2,0] = -cos(v)
   R[2,1] = 0.0
   R[2,2] = -sin(v)
   return R

def Rzw(w):
   R = np.zeros((3,3))
   R[0,0] = -sin(w)
   R[0,1] = -cos(w)
   R[0,2] = 0.0
   R[1,0] = cos(w)
   R[1,1] = -sin(w)
   R[1,2] = 0.0
   R[2,0] = 0.0
   R[2,1] = 0.0
   R[2,2] = 0.0
   return R

def Ru(u,v,w):
   R = np.zeros((3,3))
   R[0,0] = 0.0
   R[0,1] = cos(u)*sin(v)*cos(w) + sin(u)*sin(w)
   R[0,2] = cos(u)*sin(w) - sin(u)*sin(v)*cos(w)
   R[1,0] = 0.0
   R[1,1] = -sin(u)*cos(w) + cos(u)*sin(v)*sin(w)
   R[1,2] = -sin(u)*sin(v)*sin(w)-cos(u)*cos(w)
   R[2,0] = 0.0
   R[2,1] = cos(u)*cos(v)
   R[2,2] = -sin(u)*cos(v)
   return R

def Rv(u,v,w):
   R = np.zeros((3,3))
   R[0,0] = -sin(v)*cos(w)
   R[0,1] = sin(u)*cos(v)*cos(w)
   R[0,2] = cos(u)*cos(v)*cos(w)
   R[1,0] = -sin(v)*sin(w)
   R[1,1] = sin(u)*cos(v)*sin(w)
   R[1,2] = cos(u)*cos(v)*sin(w)
   R[2,0] = -cos(v)
   R[2,1] = -sin(u)*sin(v)
   R[2,2] = -cos(u)*sin(v)
   return R

def Rw(u,v,w):
   R = np.zeros((3,3))
   R[0,0] = -cos(v)*sin(w)
   R[0,1] = -sin(u)*sin(v)*sin(w)-cos(u)*cos(w)
   R[0,2] = sin(u)*cos(w)-cos(u)*sin(v)*sin(w)
   R[1,0] = cos(v)*cos(w)
   R[1,1] = -cos(u)*sin(w) + sin(u)*sin(v)*cos(w)
   R[1,2] = cos(u)*sin(v)*cos(w) + sin(u)*sin(w)
   R[2,0] = 0.0
   R[2,1] = 0.0
   R[2,2] = 0.0
   return R

u = 0.0 #-math.pi + random.random()*math.pi*2.0
v = 0.0 #-math.pi + random.random()*math.pi*2.0
w = 0.0 #-math.pi/2 + random.random()*math.pi
h = 1e-7
Rt = R(u,v,w)
R1 = Rx(u)
R2 = Ry(v)
R3 = Rz(w)
R_ = np.matmul(R3,R2)
R321 = np.matmul(R_,R1)
print("Diff:", np.linalg.norm(Rt-R321,ord='fro'))
R_u = Ru(u,v,w)
R_u_test = np.matmul(np.matmul(R3,R2),Rxu(u))
R_u_fd = (R(u+h,v,w)-Rt)/h
print("Ru", R_u)
print("Diff Ru:", np.linalg.norm(R_u-R_u_test, ord='fro'))
print("Diff Ru FD:", np.linalg.norm(R_u-R_u_fd, ord='fro'))

R_v = Rv(u,v,w)
R_v_test = np.matmul(np.matmul(R3,Ryv(v)),R1)
R_v_fd = (R(u,v+h,w)-Rt)/h
print("Rv", R_v)
print("Diff Rv:", np.linalg.norm(R_v-R_v_test, ord='fro'))
print("Diff Rv FD:", np.linalg.norm(R_v-R_v_fd, ord='fro'))

R_w = Rw(u,v,w)
R_w_test = np.matmul(np.matmul(Rzw(w),R2),R1)
R_w_fd = (R(u,v,w+h)-Rt)/h
print("Rw", R_w)
print("Diff Rw:", np.linalg.norm(R_w-R_w_test, ord='fro'))
print("Diff Rw FD:", np.linalg.norm(R_w-R_w_fd, ord='fro'))
