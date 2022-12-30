import numpy as np
import autograd
import autograd.numpy as anp
#from scipy.special import legendre
#from scipy.special import lpmv
from autograd import grad
from autograd import elementwise_grad as egrad
import torch
from torch.autograd import grad as torch_grad
from torch.autograd.functional import hessian
import math
import random 
from timeit import default_timer as timer

ANGS_TO_AU = 1.8897259886

def getFDCoeffs(numFDPoints):
    returnValue = np.zeros(numFDPoints)
    if numFDPoints == 3:
        returnValue[0] = -1.0/2.0
        returnValue[1] = 0.0
        returnValue[2] = 1.0/2.0

    elif numFDPoints == 5:
        returnValue[0] = 1.0/12.0
        returnValue[1] = -2.0/3.0
        returnValue[2] = 0.0
        returnValue[3] = 2.0/3.0
        returnValue[4] = -1.0/12.0

    elif numFDPoints == 7:
        returnValue[0] = -1.0/60.0
        returnValue[1] = 3.0/20.0
        returnValue[2] = -3.0/4.0
        returnValue[3] = 0.0
        returnValue[4] = 3.0/4.0
        returnValue[5] = -3.0/20.0
        returnValue[6] = 1.0/60.0

    elif numFDPoints == 9:
        returnValue[0] = 1.0/280.0
        returnValue[1] = -4.0/105.0
        returnValue[2] = 1.0/5.0
        returnValue[3] = -4.0/5.0
        returnValue[4] = 0.0
        returnValue[5] = 4.0/5.0
        returnValue[6] = -1.0/5.0
        returnValue[7] = 4.0/105.0
        returnValue[8] = -1.0/280.0

    elif numFDPoints == 11:
        returnValue[0] = -2.0/2520.0
        returnValue[1] = 25.0/2520.0
        returnValue[2] = -150.0/2520.0
        returnValue[3] = 600.0/2520.0
        returnValue[4] = -2100.0/2520.0
        returnValue[5] = 0.0/2520.0
        returnValue[6] = 2100.0/2520.0
        returnValue[7] = -600.0/2520.0
        returnValue[8] = 150.0/2520.0
        returnValue[9] = -25.0/2520.0
        returnValue[10] = 2.0/2520.0

    elif numFDPoints == 13:
        returnValue[0] = 5.0/27720.0
        returnValue[1] = -72.0/27720.0
        returnValue[2] = 495.0/27720.0
        returnValue[3] = -2200.0/27720.0
        returnValue[4] = 7425.0/27720.0
        returnValue[5] = -23760/27720.0
        returnValue[6] = 0.0/27720.0
        returnValue[7] = 23760.0/27720.0
        returnValue[8] = -7425.0/27720.0
        returnValue[9] = 2200.0/27720.0
        returnValue[10] = -495.0/27720.0
        returnValue[11] = 72.0/27720.0
        returnValue[12] = -5.0/27720.0

    else:
        raise Exception("Invalid number of FD points. Please enter number of FD points as 3, 5, 7, 9, 11 or 13.")

    return returnValue

def cartesian2Spherical(x,y,z):
     r = anp.sqrt(x**2.0 + y**2.0 + z**2.0)
     theta = anp.arccos(z/r)
     phi = anp.arctan2(y,x)
     return r,theta,phi


def Dm(m):
    if m == 0:
        return (2*math.pi)**(-0.5)
    else:
        return math.pi**(-0.5)

def Clm(l,m):
    a = (2.0*l + 1.0)*math.factorial(l-m)
    b = 2.0*math.factorial(l+m)
    return (a/b)**0.5

def Rn(n, alpha, x):
    if(n==1):
        return anp.exp(-alpha*x)
    else:
        return anp.power(x,n-1)*anp.exp(-alpha*x)

def associatedLegendre( n, m, x ):
    #print(type(x))
    if ( m < 0 ):
        modM = abs(m)
        factor = ((-1.0)**m)*math.factorial(l-modM)/math.factorial(l+modM)
        return factor*associatedLegendre(n,modM,x)

    if ( n < m ):
        return anp.zeros(len(x))

    if ( x.any() < -1.0 or 1.0 < x.any()):
        raise Exception("The argument to associated legendre must be in [-1,1]")

    nx = len(x)
    cxM = anp.ones(nx)
    somx2 = anp.sqrt(1.0 - x*x)

    fact = 1.0
    for i in range ( 0, m ):
        cxM = - cxM * fact * somx2
        fact = fact + 2.0

    cx = cxM
    cxMplus = anp.zeros(nx)
    if ( m != n ):
        cxMPlus1 = x * ( 2 * m + 1 ) * cxM
        cx = cxMPlus1

        cxPrev = cxMPlus1
        cxPrevPrev = cxM
        for i in range ( m + 2, n + 1 ):
            cx = ( ( 2 * i     - 1 ) * x * cxPrev \
                    + (   - i - m + 1 ) *     cxPrevPrev ) \
                    / (     i - m     )
            cxPrevPrev = cxPrev
            cxPrev = cx

    return ((-1.0)**m)*cx

def Qm(m, x):
    if m > 0:
        return anp.cos(m*x)
    elif m == 0:
        return anp.ones(x.shape)
    else:
        return anp.sin(abs(m)*x)

def slaterBasisHandle(n,l,m,alpha):
    def slaterBasisFunc(x,y,z):
        absm=abs(m)
        r,theta,phi = cartesian2Spherical(x,y,z)
        C = Clm(l,absm)*Dm(m)
        cosTheta = anp.cos(theta)
        R = Rn(n, alpha, r)
        P = associatedLegendre(l, absm, cosTheta)
        Q = Qm(m, phi)
        return C*R*P*Q

    return slaterBasisFunc

def readCoordFile(coordFile):
    f = open(coordFile, 'r')
    lines = f.readlines()
    atoms = []
    for line in lines:
        words = line.split()
        if len(words) != 5:
            raise Exception("Expects only 5 values in coord file " + coordFile + "Line read: " + line +". Num words: " + str(len(words)))

        atom = {}
        atom['name'] = words[0]
        atom['coord'] = [ANGS_TO_AU*float(w) for w in words[1:4]]
        atom['basisfile'] = words[4]
        atoms.append(atom)

    f.close()
    return atoms


class SlaterPrimitive():
    def __init__(self, n, l, m, a):
        self.n = n
        self.l = l
        self.m = m
        self.a = a
        t1 = (2.0*a)**(n + 1.0/2.0)
        t2 = (np.math.factorial(2*n))**0.5
        self.nrm = t1/t2

    def alpha(self):
        return self.a

    def nlm(self):
        return self.n, self.l, self.m

    def normConst(self):
        return self.nrm


def getAtomSlaterBasis(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    lStringToIntMap = {'s':0, 'p':1, 'd':2, 'f':3, 'g':4, 'h':5}
    basisList = []
    lchars = 'spdfgh'
    #ignore the first line
    for line in lines[1:]:
        words= line.split()
        if len(words) != 2:
            raise Exception("Invalid number of columns in file " + filename + ". Expects 2 values")

        nlString = words[0]
        alpha = float(words[1])
        n = 0
        l = 0
        lchar = nlString[-1].lower()
        found = lchar in lchars
        if found == False:
            raise Exception("Invalid l-character string found in file " + filename)

        n = int(nlString[:-1])
        l = lStringToIntMap[lchar]
        mList = []
        # NOTE: QChem even in the spherical form uses cartesian form for the s and p orbitals.
        # The ordering for cartesian orbitals are lexicographic - i.e., for p orbitals it's ordered
        # as x,y,z. While in the spherical form if one uses -l to +l ordering for the m quantum numbers for l=1 (p orbital),
        # then it translates to ordering the p orbitals as y,z,x.
        # To be consistent with QChem's ordering for p orbital, we do it in an
        # ad-hoc manner by ordering the m quantum numbers as 1,-1,0 for l=1 (p orbital).
        if l == 1:
            mList = [1, -1, 0]

        else:
            mList = list(range(-l,l+1))

        for m in mList:
            basis = SlaterPrimitive(n,l,m,alpha)
            basisList.append(basis)


    f.close()
    return basisList

def getFunctionHigherDerivativeHandle(derIndices, functionHandle):
    numIndices = len(derIndices)
    if numIndices == 0:
        return functionHandle
    else: 
        index = derIndices[0]
        remainingIndices = derIndices[1:]
        return egrad(getFunctionHigherDerivativeHandle(remainingIndices,\
                                                      functionHandle),\
                                                      index)
        

class SlaterDensity():
    def __init__(self, coordFile, DMFiles):
        atoms = readCoordFile(coordFile)
        atomBasisFileSet = set()
        for atom in atoms:
            atomBasisFileSet.add(atom['basisfile'])

        atomBasisMap = {}
        for basisFile in atomBasisFileSet:
            atomBasisMap[basisFile] = getAtomSlaterBasis(basisFile)

        self.basisList = []

        for atom in atoms:
            basisFile= atom['basisfile']
            basisList = atomBasisMap[basisFile]
            for basis in basisList:
                b = {}
                b['atom'] = atom['name']
                b['center'] = atom['coord']
                b['primitive'] = basis
                self.basisList.append(b)

        self.nbasis = len(self.basisList)
        numSpinComponents = len(DMFiles)
        if numSpinComponents > 2:
            raise Exception("More than two density matrices provided. The number of density matrices can be either 1 (for a spin unpolarized system) or 2 (for a spin polarized system)")
        self.DM = []
        for i in range(numSpinComponents):
            DM = np.loadtxt(DMFiles[i],  dtype = np.float64, usecols=range(0,self.nbasis))
            self.DM.append(DM)

    def getDensity(self,x, spinComponent):
        npoints = x.shape[0]
        A = np.zeros((npoints, self.nbasis, self.nbasis))
        basisVals = np.zeros((npoints, self.nbasis))
        for i in range(self.nbasis):
            center = self.basisList[i]['center']
            primitive = self.basisList[i]['primitive']
            n, l, m = primitive.nlm()
            a = primitive.alpha()
            nrm = primitive.normConst()
            slaterFunction = slaterBasisHandle(n,l,m,a)
            xShifted = x-center
            basisVals[:,i] = nrm*slaterFunction(xShifted[:,0], xShifted[:,1],\
                                                xShifted[0:,2])

        for p in range(npoints):
            A[p] = np.outer(basisVals[p], basisVals[p])

        return np.tensordot(A,self.DM[spinComponent],2)

    def getDensityEvalHandleSplit(self, spinComponent):
        basisList = self.basisList
        nbasis = self.nbasis
        DM = self.DM[spinComponent]
        def densityEval(x,y,z):
            npoints = x.shape[0]
            rho = anp.zeros(npoints)
            basisVals = [anp.zeros(npoints)]*nbasis
            for i in range(nbasis):
                center = basisList[i]['center']
                primitive = basisList[i]['primitive']
                n, l, m = primitive.nlm()
                a = primitive.alpha()
                nrm = primitive.normConst()
                slaterFunction = slaterBasisHandle(n,l,m,a)
                basisVals[i] = nrm*slaterFunction(x-center[0], y-center[1],\
                                                    z- center[2])

            for i in range(nbasis):
                for j in range(nbasis):
                    rho = rho + DM[i,j]*basisVals[i]*basisVals[j]

            return rho 

        return densityEval
    
    def getDensityEvalHandle(self, spinComponent):
        basisList = self.basisList
        nbasis = self.nbasis
        DM = self.DM[spinComponent]
        def densityEval(x):
            npoints = x.shape[0]
            A = anp.zeros((npoints, nbasis, nbasis))
            basisVals = anp.zeros((npoints, nbasis))
            for i in range(nbasis):
                center = basisList[i]['center']
                primitive = basisList[i]['primitive']
                n, l, m = primitive.nlm()
                a = primitive.alpha()
                nrm = primitive.normConst()
                slaterFunction = slaterBasisHandle(n,l,m,a)
                xShifted = x - center
                basisVals[:,i] = nrm*slaterFunction(xShifted[:,0], xShifted[:,1],\
                                                    xShifted[:,2])

            for p in range(npoints):
                A[p] = anp.outer(basisVals[p], basisVals[p])

            return anp.tensordot(A,DM,2)

        return densityEval

    def getDensityDerivative(self, x, index, spinComponent):
        basisList = self.basisList
        nbasis = self.nbasis
        DM = self.DM[spinComponent]
        npoints = x.shape[0]
        A = anp.zeros((npoints, nbasis, nbasis))
        basisVals = anp.zeros((npoints, nbasis))
        basisDerVals = anp.zeros((npoints, nbasis))
        for i in range(nbasis):
            center = basisList[i]['center']
            primitive = basisList[i]['primitive']
            n, l, m = primitive.nlm()
            a = primitive.alpha()
            nrm = primitive.normConst()
            slaterFunction = slaterBasisHandle(n,l,m,a)
            slaterFunctionDer =\
            getFunctionHigherDerivativeHandle([index],slaterFunction)
            xShifted = x - center
            basisVals[:,i] = nrm*slaterFunction(xShifted[:,0], xShifted[:,1],\
                                                    xShifted[:,2])
            basisDerVals[:,i] = nrm*slaterFunctionDer(xShifted[:,0],\
                                                      xShifted[:,1],
                                                      xShifted[:,2])
        for p in range(npoints):
            A[p] = 2.0*anp.outer(basisVals[p], basisDerVals[p])

        return anp.tensordot(A,DM,2)
    
    def getDensityDoubleDerivative(self, x, index1, index2, spinComponent):
        basisList = self.basisList
        nbasis = self.nbasis
        DM = self.DM[spinComponent]
        npoints = x.shape[0]
        A = anp.zeros((npoints, nbasis, nbasis))
        basisVals = anp.zeros((npoints, nbasis))
        basisDerVals1 = anp.zeros((npoints, nbasis))
        basisDerVals2 = anp.zeros((npoints, nbasis))
        basisDoubleDerVals = anp.zeros((npoints, nbasis))
        for i in range(nbasis):
            center = basisList[i]['center']
            primitive = basisList[i]['primitive']
            n, l, m = primitive.nlm()
            a = primitive.alpha()
            nrm = primitive.normConst()
            slaterFunction = slaterBasisHandle(n,l,m,a)
            slaterFunctionDer1 =\
            getFunctionHigherDerivativeHandle([index1],slaterFunction)
            slaterFunctionDer2 =\
            getFunctionHigherDerivativeHandle([index2],slaterFunction)
            slaterFunctionDoubleDer =\
            getFunctionHigherDerivativeHandle([index1, index2],slaterFunction)
            xShifted = x - center
            basisVals[:,i] = nrm*slaterFunction(xShifted[:,0], xShifted[:,1],\
                                                    xShifted[:,2])
            basisDerVals1[:,i] = nrm*slaterFunctionDer1(xShifted[:,0],\
                                                      xShifted[:,1],
                                                      xShifted[:,2])
            basisDerVals2[:,i] = nrm*slaterFunctionDer2(xShifted[:,0],\
                                                      xShifted[:,1],
                                                      xShifted[:,2])
            basisDoubleDerVals[:,i] = nrm*slaterFunctionDoubleDer(xShifted[:,0],\
                                                      xShifted[:,1],
                                                      xShifted[:,2])
        for p in range(npoints):
            A[p] = 2.0*(anp.outer(basisVals[p], basisDoubleDerVals[p]) +\
                   anp.outer(basisDerVals1[p], basisDerVals2[p]))

        return anp.tensordot(A,DM,2)


   

def getFunctionHigherOrderDerivativeFD(x,derIndices, numFDPoints, h, functionHandle):
    numPoints = x.shape[0]
    numIndices = len(derIndices)
    higherOrderDer = np.zeros(numPoints)
    if numIndices == 0:
        higherOrderDer = functionHandle(x)
        return higherOrderDer

    else:
        currentIndex = derIndices[numIndices-1]
        indicesNext = derIndices[0:numIndices-1]
        coeffs = getFDCoeffs(numFDPoints)
        for i in range(numFDPoints):
            FDPoint = np.copy(x)
            shiftIndex = i-int(numFDPoints/2)
            FDPoint[:, currentIndex] += shiftIndex*h
            factor = coeffs[i]
            higherOrderDer += factor* \
                    getFunctionHigherOrderDerivativeFD(FDPoint,\
                                                     indicesNext,\
                                                     numFDPoints,\
                                                     h,\
                                                    functionHandle)

        return higherOrderDer/h



def getDensityHigherDerivativeHandle(slaterDensity, spinComponent, derIndices):
    numIndices = len(derIndices)
    if numIndices == 0:
        return slaterDensity.getDensityEvalHandleSplit(spinComponent)
    else: 
        index = derIndices[0]
        remainingIndices = derIndices[1:]
        return egrad(getDensityHigherDerivativeHandle(slaterDensity,\
                                                      spinComponent,\
                                                      remainingIndices),index)


quadBatch = 100000
DMFiles = ['DensityMatrix_inverse_slater_QZ4P_projection0',\
           'DensityMatrix_inverse_slater_QZ4P_projection1']
nSpin = len(DMFiles)
sd = SlaterDensity('AtomicCoords_Slater', DMFiles)
quad = np.loadtxt('QuadWeights', dtype = np.float64, usecols=(0,1,2,4))
nQuadAll = quad.shape[0]
ne = [0.0]*nSpin
nx = 5
numFDPoints = 9
h = 1e-4
densityEval1 = sd.getDensityEvalHandle(0)
densityEval2 = sd.getDensityEvalHandleSplit(0)

for i in range(0,nQuadAll,quadBatch):
    minId = i
    maxId = min(minId+quadBatch, nQuadAll)
    x = quad[minId:maxId, 0:3]
    w = quad[minId:maxId, 3]
    for iSpin in range(nSpin):
        #rho = sd.getDensityEvalHandleSplit(iSpin)
        rho = sd.getDensity(x, iSpin)
        #rho = f(x[:,0], x[:,1], x[:,2])
        ne[iSpin] += np.dot(rho,w)

for iSpin in range(nSpin):
    print ("Num. of spin", iSpin, "electrons:", ne[iSpin])

outFs = []
for iSpin in range(nSpin):
    outF = open("RhoSlaterData"+str(iSpin),"w")
    outFs.append(outF)

for i in range(0,nQuadAll,quadBatch):
    minId = i
    maxId = min(minId+quadBatch, nQuadAll)
    x = quad[minId:maxId, 0:3]
    nQuad = x.shape[0]

    for iSpin in range(nSpin):
        rho = sd.getDensity(x,iSpin)
        drho = np.zeros((nQuad,3))
        ddrho = np.zeros((nQuad,3,3))
        for iComp in range(3):
            drho[:,iComp] = sd.getDensityDerivative(x, iComp, iSpin)
            for jComp in range(iComp,3):
                ddrho[:,iComp,jComp] = sd.getDensityDoubleDerivative(x, iComp,\
                                                                     jComp,\
                                                                     iSpin)
                ddrho[:,jComp,iComp] = ddrho[:,iComp,jComp]

        outF = outFs[iSpin]
        for iQuad in range(nQuad):
            print(''.join([str(y) + " " for y in x[iQuad]]), file=outF, end=' ')
            print(rho[iQuad], file=outF, end=' ')
            print(''.join([str(y) + " " for y in drho[iQuad]]), file=outF, end=' ')
            for iComp in range(3):
                print(''.join([str(y) + " " for y in ddrho[iQuad,iComp]]), file=outF, end=' ')
           
            print(file=outF)



for iSpin in range(nSpin):
    outFs[iSpin].close()
