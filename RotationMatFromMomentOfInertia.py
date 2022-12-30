import numpy as np
import math
dtype = np.float64

def getR(angles):
    R = np.zeros((3,3), dtype)
    u = angles[0]
    v = angles[1]
    w = angles[2]
    R[0,0]	=	math.cos(v)*math.cos(w)
    R[0,1]	=	math.sin(u)*math.sin(v)*math.cos(w)	-	math.cos(u)*math.sin(w)
    R[0,2]	=	math.sin(u)*math.sin(w)	+	math.cos(u)*math.sin(v)*math.cos(w)
    R[1,0]	=	math.cos(v)*math.sin(w)
    R[1,1]	=	math.cos(u)*math.cos(w)	+	math.sin(u)*math.sin(v)*math.sin(w)
    R[1,2]	=	math.cos(u)*math.sin(v)*math.sin(w)	-	math.sin(u)*math.cos(w)
    R[2,0]	=	-math.sin(v)
    R[2,1]	=	math.sin(u)*math.cos(v)
    R[2,2]	=	math.cos(u)*math.cos(v)
    return R

def getdRdU(angles):
    R = np.zeros((3,3), dtype)
    u = angles[0]
    v = angles[1]
    w = angles[2]
    R[0,0]	=	0
    R[0,1]	=	math.cos(u)*math.sin(v)*math.cos(w)	+	math.sin(u)*math.sin(w)
    R[0,2]	=	math.cos(u)*math.sin(w)	-	math.sin(u)*math.sin(v)*math.cos(w)
    R[1,0]	=	0
    R[1,1]	=	-math.sin(u)*math.cos(w)	+	math.cos(u)*math.sin(v)*math.sin(w)
    R[1,2]	=	-math.sin(u)*math.sin(v)*math.sin(w)-math.cos(u)*math.cos(w)
    R[2,0]	=	0
    R[2,1]	=	math.cos(u)*math.cos(v)
    R[2,2]	=	-math.sin(u)*math.cos(v)
    return R

def getdRdV(angles):
    R = np.zeros((3,3), dtype)
    u = angles[0]
    v = angles[1]
    w = angles[2]
    R[0,0]	=	-math.sin(v)*math.cos(w)
    R[0,1]	=	math.sin(u)*math.cos(v)*math.cos(w)
    R[0,2]	=	math.cos(u)*math.cos(v)*math.cos(w)
    R[1,0]	=	-math.sin(v)*math.sin(w)
    R[1,1]	=	math.sin(u)*math.cos(v)*math.sin(w)
    R[1,2]	=	math.cos(u)*math.cos(v)*math.sin(w)
    R[2,0]	=	-math.cos(v)
    R[2,1]	=	-math.sin(u)*math.sin(v)
    R[2,2]	=	-math.cos(u)*math.sin(v)
    return R

def getdRdW(angles):
    R = np.zeros((3,3), dtype)
    u = angles[0]
    v = angles[1]
    w = angles[2]
    R[0,0]	=	-math.cos(v)*math.sin(w)
    R[0,1]	=	-math.sin(u)*math.sin(v)*math.sin(w)-math.cos(u)*math.cos(w)
    R[0,2]	=	math.sin(u)*math.cos(w)-math.cos(u)*math.sin(v)*math.sin(w)
    R[1,0]	=	math.cos(v)*math.cos(w)
    R[1,1]	=	-math.cos(u)*math.sin(w)	+	math.sin(u)*math.sin(v)*math.cos(w)
    R[1,2]	=	math.cos(u)*math.sin(v)*math.cos(w)	+	math.sin(u)*math.sin(w)
    R[2,0]	=	0
    R[2,1]	=	0
    R[2,2]	=	0
    return R

def getDiffMat(IFE, IGaussian, angles):
    R = getR(angles)
    return IGaussian - R.dot(IFE).dot(R.transpose())

IFE = np.loadtxt('MoIFE', dtype)
IGaussian = np.loadtxt('MoIGaussian', dtype)
normFE = np.linalg.norm(IFE, ord='fro')
IFE *= 1.0/normFE
normGaussian = np.linalg.norm(IGaussian, ord='fro')
IGaussian *= 1.0/normGaussian

print(IFE)
print(IGaussian)
NIter = 100000
angles = np.zeros(3, dtype)
angles[0] = 0.0 #2.093310
angles[1] = 0.0 # -0.080799
angles[2] = 0.0 #-0.947304
h = 1e-3
#for i in range(NIter):
#    R = getR(angles)
#    X = np.matmul(R,IFE)
#    Y = np.matmul(X, R.transpose())
#    diffMat = IGaussian - Y
#    print("Iter", i, "Norm:", np.linalg.norm(diffMat, ord='fro'))
#    dRdU = getdRdU(angles)
#    dRdV = getdRdV(angles)
#    dRdW = getdRdW(angles)
#    BU = dRdU.dot(IFE).dot(R.transpose()) + R.dot(IFE).dot(dRdU.transpose())
#    BV = dRdV.dot(IFE).dot(R.transpose()) + R.dot(IFE).dot(dRdV.transpose())
#    BW = dRdW.dot(IFE).dot(R.transpose()) + R.dot(IFE).dot(dRdW.transpose())
#    forces = np.zeros(3, dtype)
#    forces[0] = -2.0*np.sum(np.multiply(diffMat,BU))
#    forces[1] = -2.0*np.sum(np.multiply(diffMat,BV))
#    forces[2] = -2.0*np.sum(np.multiply(diffMat,BW))
#    angles = angles - forces

for i in range(NIter):
    A = getDiffMat(IFE, IGaussian, angles)
    print("Iter", i, "Norm:", np.linalg.norm(A, ord='fro'))
    forces = np.zeros(3, dtype)
    for j in range(3):
        anglesCopy0 = np.copy(angles)
        anglesCopy1 = np.copy(angles)
        anglesCopy0[j] -= h
        anglesCopy1[j] += h
        A0 = getDiffMat(IFE, IGaussian, anglesCopy0)
        A1 = getDiffMat(IFE, IGaussian, anglesCopy1)
        forces[j] = ((np.linalg.norm(A1, ord='fro'))**2-(np.linalg.norm(A0, ord='fro'))**2.0)/(2*h)

    print("Iter", i, "ForceNorm:", np.linalg.norm(forces, ord=2))
    angles = angles - forces

R = getR(angles)
print("Angles", angles)
print('Rotated IFE', R.dot(IFE).dot(R.transpose()))






