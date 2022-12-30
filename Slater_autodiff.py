import numpy as np
import autograd.numpy as anp
from autograd import grad
from autograd import elementwise_grad as egrad
from scipy.special import lpmv
import torch
from torch.autograd import grad as torch_grad
from torch.autograd.functional import hessian
import math
import random 

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


def cartesian2Spherical(xyz):
    spherical=anp.zeros(xyz.shape)
    spherical[:,0]= anp.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    spherical[:,1] = anp.arccos(xyz[:,2]/spherical[:,0])
    spherical[:,2] = anp.arctan2(xyz[:,1], xyz[:,0])
    return spherical

def cartesian2Spherical_v2(xyz):
     r = anp.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
     theta = anp.arccos(xyz[:,2]/r)
     phi = anp.arctan2(xyz[:,1], xyz[:,0])
     return r,theta,phi

def cartesian2Spherical_v3(x,y,z):
     r = anp.sqrt(x**2.0 + y**2.0 + z**2.0)
     theta = anp.arccos(z/r)
     phi = anp.arctan2(y,x)
     return r,theta,phi

def cartesian2Spherical_torch(xyz):
     spherical = torch.zeros_like(xyz)
     r = torch.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
     spherical[:,0] = r
     spherical[:,1] = torch.arccos(xyz[:,2]/r)
     spherical[:,2] = torch.atan2(xyz[:,1], xyz[:,0])
     return spherical

def Dm(m):
    if m == 0:
        return (2*math.pi)**(-0.5)
    else:
        return math.pi**(-0.5)

def Clm(l,m):
    a = (2.0*l + 1.0)*math.factorial(l-m)
    b = 2.0*math.factorial(l+m)
    return (a/b)**0.5

def Plm(l,m,x):
    if abs(m) > l:
        return anp.zeros(len(x))

    # The (-1)^m is multiplied to remove the Condon-Shokley factor in
    # the scipy associated Legendre polynomial implementation. This is
    # done to be consistent with the quantum chemistry codes
    return ((-1.0)**m)*lpmv(m,l,x)

def Qm(m, x):
    if m > 0:
        return anp.cos(m*x)
    elif m == 0:
        return anp.ones(x.shape)
    else:
        return anp.sin(abs(m)*x)

def Qm_torch(m, x):
    if m > 0:
        return torch.cos(m*x)
    elif m == 0:
        return torch.ones(x.shape)
    else:
        return torch.sin(abs(m)*x)

def Rn(n, alpha, x):
    if(n==1):
        return anp.exp(-alpha*x)
    else:
        return anp.power(x,n-1)*anp.exp(-alpha*x)

def Rn_torch(n, alpha, x):
    if(n==1):
        return torch.exp(-alpha*x)
    else:
        return torch.pow(x,n-1)*torch.exp(-alpha*x)
#*****************************************************************************80
#
## LEGENDRE_ASSOCIATED evaluates the associated Legendre functions.
#
#  Differential equation:
#
#    (1-X*X) * Y'' - 2 * X * Y + ( N (N+1) - (M*M/(1-X*X)) * Y = 0
#
#  First terms:
#
#    M = 0  ( = Legendre polynomials of first kind P(N)(X) )
#
#    P00 =    1
#    P10 =    1 X
#    P20 = (  3 X^2 -   1)/2
#    P30 = (  5 X^3 -   3 X)/2
#    P40 = ( 35 X^4 -  30 X^2 +   3)/8
#    P50 = ( 63 X^5 -  70 X^3 +  15 X)/8
#    P60 = (231 X^6 - 315 X^4 + 105 X^2 -  5)/16
#    P70 = (429 X^7 - 693 X^5 + 315 X^3 - 35 X)/16
#
#    M = 1
#
#    P01 =   0
#    P11 =   1 * SQRT(1-X*X)
#    P21 =   3 * SQRT(1-X*X) * X
#    P31 = 1.5 * SQRT(1-X*X) * (5*X*X-1)
#    P41 = 2.5 * SQRT(1-X*X) * (7*X*X*X-3*X)
#
#    M = 2
#
#    P02 =   0
#    P12 =   0
#    P22 =   3 * (1-X*X)
#    P32 =  15 * (1-X*X) * X
#    P42 = 7.5 * (1-X*X) * (7*X*X-1)
#
#    M = 3
#
#    P03 =   0
#    P13 =   0
#    P23 =   0
#    P33 =  15 * (1-X*X)^1.5
#    P43 = 105 * (1-X*X)^1.5 * X
#
#    M = 4
#
#    P04 =   0
#    P14 =   0
#    P24 =   0
#    P34 =   0
#    P44 = 105 * (1-X*X)^2
#
#  Recursion:
#
#    if N < M:
#      P(N,M) = 0
#    if N = M:
#      P(N,M) = (2*M-1)!! * (1-X*X)^(M/2) where N!! means the product of
#      all the odd integers less than or equal to N.
#    if N = M+1:
#      P(N,M) = X*(2*M+1)*P(M,M)
#    if M+1 < N:
#      P(N,M) = ( X*(2*N-1)*P(N-1,M) - (N+M-1)*P(N-2,M) )/(N-M)
#
#  Restrictions:
#
#    -1 <= X <= 1
#     0 <= M <= N
#
#  Special values:
#
#    P(N,0)(X) = P(N)(X), that is, for M=0, the associated Legendre
#    function of the first kind equals the Legendre polynomial of the
#    first kind.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    19 February 2015
#
#  Author:
#
#    John Burkardt
#
#  Reference:
#
#    Milton Abramowitz and Irene Stegun,
#    Handbook of Mathematical Functions,
#    US Department of Commerce, 1964.
#
#  Parameters:
#
#    Ianp.t, integer N, the maximum first index of the Legendre
#    function, which must be at least 0.
#
#    Ianp.t, integer M, the second index of the Legendre function,
#    which must be at least 0, and no greater than N.
#
#    Ianp.t, real X, the point at which the function is to be
#    evaluated.  X must satisfy -1 <= X <= 1.
#
#    Output, real CX(1:N+1), the values of the first N+1 function.
#
def legendre_associated( n, m, x ):
    #print(type(x))
    if ( m < 0 ):
        modM = abs(m)
        factor = ((-1.0)**m)*math.factorial(l-modM)/math.factorial(l+modM)
        return factor*legendre_associated(n,modM,x)

    if ( n < m ):
        return 0.0

    if ( x.any() < -1.0 ):
        print ( '' )
        print ( 'LEGENDRE_ASSOCIATED - Fatal error!' )
        print ( '  Ianp.t value of X = %f' % ( x ) )
        print ( '  but X must be no less than -1.' )

    if ( 1.0 < x.any() ):
        print ( '' )
        print ( 'LEGENDRE_ASSOCIATED - Fatal error!' )
        print ( '  Ianp.t value of X = %f' % ( x ) )
        print ( '  but X must be no more than 1.' )

    cx = [0.0]*(n+1)
    cx[m] = 1.0
    somx2 = anp.sqrt ( 1.0 - x * x )

    fact = 1.0
    for i in range ( 0, m ):
        cx[m] = - cx[m] * fact * somx2
        fact = fact + 2.0

    if ( m != n ):
        cx[m+1] = x * ( 2 * m + 1 ) * cx[m]

        for i in range ( m + 2, n + 1 ):
            cx[i] = ( ( 2 * i     - 1 ) * x * cx[i-1] \
                    + (   - i - m + 1 ) *     cx[i-2] ) \
                    / (     i - m     )

    return ((-1.0)**m)*cx[n]


def legendre_associated_v2( n, m, x ):
    #print(type(x))
    if ( m < 0 ):
        modM = abs(m)
        factor = ((-1.0)**m)*math.factorial(l-modM)/math.factorial(l+modM)
        return factor*legendre_associated_v2(n,modM,x)

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

def legendre_associated_torch( n, m, x ):
    #print(type(x))
    if ( m < 0 ):
        modM = abs(m)
        factor = ((-1.0)**m)*math.factorial(l-modM)/math.factorial(l+modM)
        return factor*legendre_associated_v2(n,modM,x)

    if ( n < m ):
        return torch.zeros(len(x))

    #if ( x.any() < -1.0 or 1.0 < x.any()):
    #    raise Exception("The argument to associated legendre must be in [-1,1]")

    cx = [torch.tensor(x.shape[0])]*(n+1)
    cx[m] = torch.ones(x.shape[0])
    somx2 = torch.sqrt(1.0 - x*x)

    fact = 1.0
    for i in range (0, m):
        cx[m] = - cx[m] * fact * somx2
        fact = fact + 2.0

    if ( m != n ):
        cx[m+1] = x * ( 2 * m + 1 ) * cx[m]

        for i in range ( m + 2, n + 1 ):
            cx[i] = ( ( 2 * i     - 1 ) * x * cx[i-1] \
                     + (   - i - m + 1 ) *     cx[i-2] ) \
                    / (     i - m     )

    return ((-1.0)**m)*cx[n]


def legendre_associated_test(l,m,x):
    return lpmv(m,l,x)


def alf(l,m):
    def lpmnvalues(x):
        return lpmv(m,l,x) #legendre_associated(l,m,x)

    return lpmnvalues

def test(x):
    print("type:", type(x))
    return x**2.0


def slaterBasis(n,l,m,alpha):
    def slaterBasisFunc(x):
        absm=abs(m)
        r,theta,phi = cartesian2Spherical_v2(x)
        C = Clm(l,absm)*Dm(m)
        cosTheta = anp.cos(theta)
        R = Rn(n, alpha, r)
        P = legendre_associated_v2(l, absm, cosTheta)
        Q = Qm(m, phi)
        return C*R*P*Q

    return slaterBasisFunc

def slaterBasis_split(n,l,m,alpha):
    def slaterBasisFunc(x,y,z):
        absm=abs(m)
        r,theta,phi = cartesian2Spherical_v3(x,y,z)
        C = Clm(l,absm)*Dm(m)
        cosTheta = anp.cos(theta)
        R = Rn(n, alpha, r)
        P = legendre_associated_v2(l, absm, cosTheta)
        Q = Qm(m, phi)
        return C*R*P*Q

    return slaterBasisFunc

def slaterBasis_torch(n,l,m,alpha):
    def slaterBasisFunc_torch(x):
        absm=abs(m)
        xspherical = cartesian2Spherical_torch(x)
        C = Clm(l,absm)*Dm(m)
        cosTheta = torch.cos(xspherical[:,1])
        R = Rn_torch(n, alpha, xspherical[:,0])
        P = legendre_associated_torch(l, absm, cosTheta)
        Q = Qm_torch(m, xspherical[:,2])
        return C*R*P*Q

    return slaterBasisFunc_torch

def slaterBasisVal(n,l,m,alpha, x):
    absm=abs(m)
    xspherical = cartesian2Spherical(x)
    C = Clm(l,absm)*Dm(m)
    cosTheta = anp.cos(xspherical[:,1])
    R = Rn(n, alpha, xspherical[:,0])
    P = Plm(l, absm, cosTheta)
    Q = Qm(m, xspherical[:,2])
    return C*R*P*Q

def getSlaterBasisHigherDerivativeFD(n, l, m, a, x, derIndices, numFDPoints, h):
    numPoints = x.shape[0]
    numIndices = len(derIndices)
    higherOrderDer = np.zeros(numPoints)
    if numIndices == 0:
        higherOrderDer = slaterBasisVal(n,l,m,a,x)
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
                    getSlaterBasisHigherDerivativeFD(n, l, m, a, FDPoint,\
                                                     indicesNext,\
                                                     numFDPoints,h)

        return higherOrderDer/h

def getSlaterBasisHigherDerivativeAGHandle(n, l, m, a, derIndices):
    numIndices = len(derIndices)
    if numIndices == 0:
        return slaterBasis_split(n,l,m,a)
    else: 
        index = derIndices[0]
        remainingIndices = derIndices[1:]
        return egrad(getSlaterBasisHigherDerivativeAGHandle(n,l,m,a,\
                                                            remainingIndices),index) 
#x1 = 0.5
#x2 = anp.array([0.5])
#f1 = legendre_associated(2,2, x1)
#f2 = legendre_associated_test(2,2,x1)
#print(f1,f2)

#dfdx = grad(f)
#edfdx = elementwise_grad(f)
#print(dfdx(x1))

#x3 = anp.arange(-0.99, 1.0, 0.1)
#nx = x3.shape[0]
#for l in range(6):
#    for m in range(-l,l+1):
#        f = alf(l,m)
#        y2 = legendre_associated_v2(l, m, x3)
#        y1 = anp.zeros_like(x3)
#        for i in range(nx):
#            y1[i] = f(x3[i])
#
#        diff = anp.linalg.norm(y1-y2,2)
#        base = anp.linalg.norm(y1,2)
#        #print("y1",y1)
#        #print("y2",y2)
#        print(l,m,"Abs. diff:", diff, "Rel. diff:", diff/base)

x = anp.random.rand(3)
x1 = anp.random.rand(1,3)
#print(x1)
n,l,m = 2,1,0
alpha = 2.0
f = slaterBasis_split(n,l,m,alpha)
#print(f(x1))
#print(slaterBasisVal(n,l,m,alpha,x1))
dfdx = egrad(f,0)
dfdy = egrad(f,1)
dfdz = egrad(f,2)
d2fdxdx = egrad(dfdx,0)
d2fdxdy = egrad(dfdx,1)
d2fdxdz = egrad(dfdx,2)
d2fdydx = egrad(dfdy,0)
d2fdydy = egrad(dfdy,1)
d2fdydz = egrad(dfdy,2)
d2fdzdx = egrad(dfdz,0)
d2fdzdy = egrad(dfdz,1)
d2fdzdz = egrad(dfdz,2)
print("x", x1)
print(dfdx(x1[:,0], x1[:,1], x1[:,2]), dfdy(x1[:,0], x1[:,1], x1[:,2]), dfdz(x1[:,0], x1[:,1], x1[:,2]))
print(d2fdxdx(x1[:,0], x1[:,1], x1[:,2]), d2fdxdy(x1[:,0], x1[:,1], x1[:,2]),
      d2fdxdz(x1[:,0], x1[:,1], x1[:,2]))
print(d2fdydx(x1[:,0], x1[:,1], x1[:,2]), d2fdydy(x1[:,0], x1[:,1], x1[:,2]),
      d2fdydz(x1[:,0], x1[:,1], x1[:,2]))
print(d2fdzdx(x1[:,0], x1[:,1], x1[:,2]), d2fdzdy(x1[:,0], x1[:,1], x1[:,2]),
      d2fdzdz(x1[:,0], x1[:,1], x1[:,2]))

#dfdx = grad(f)
#print(x)
#print(dfdx(x))
#print(d2fdx2(x))
#with autograd.detect_anomaly():
t = torch.tensor(x1).requires_grad_()
f_torch = slaterBasis_torch(n,l,m,alpha)
y = f_torch(t)
y.backward()
print(t.grad)
h = hessian(f_torch,t)
print(h)

numFDPoints = 9
h = 1e-3
for i in range(3):
    for j in range(3):
        derIndices = [i,j]
        dd = getSlaterBasisHigherDerivativeFD(n,l,m,alpha,x1,derIndices, numFDPoints, h)
        print(dd, end=" ")

    print()

lossD12 = 0.0
lossD23 = 0.0
lossD13 = 0.0
lossDD12 = 0.0
lossDD23 = 0.0
lossDD13 = 0.0
nx = 50
trials = 100
count = 0
for k in range(trials):
    for n in range(1,6):
        for l in range(n):
            for m in range(-l,l+1):
                count += 1
                a = random.random()*10.0
                x = anp.random.rand(nx,3)*50.0
                #x = anp.zeros((5, 3), dtype=np.float32)
                #x[:] = anp.random.randn(*x.shape)
                t = torch.tensor(x).requires_grad_()
                f2 = slaterBasis_split(n,l,m,a)
                f3 = slaterBasis_torch(n,l,m,a)
                
                y1 = slaterBasisVal(n,l,m,a,x)
                y2 = f2(x[:,0],x[:,1],x[:,2])
                y3 = f3(t)
                
    
                df1 = np.zeros_like(x)
                for i in range(3):
                    derIndices = [i]
                    df1[:,i]=getSlaterBasisHigherDerivativeFD(n,l,m,a,x,derIndices,\
                                                                     numFDPoints, h)
    
                dfdx = egrad(f2,0)
                dfdy = egrad(f2,1)
                dfdz = egrad(f2,2)
                
                df2 = np.zeros((x.shape[0],3))
                #df2 = np.array([dfdx(x[:,0], x[:,1], x[:,2]),\
                #       dfdy(x[:,0], x[:,1], x[:,2]),\
                #       dfdz(x[:,0], x[:,1], x[:,2])])
                for i in range(3):
                    derIndices = [i]
                    df = getSlaterBasisHigherDerivativeAGHandle(n,l,m,a,derIndices)
                    df2[:,i] = df(x[:,0],x[:,1],x[:,2])
    
    
                #df2 = np.transpose(df2)
    
    
                
                tmp = torch.ones_like(y3)
                df3 = (torch.autograd.grad(y3, t, grad_outputs = tmp)[0]).numpy()
                                  #retain_graph=True, create_graph=True)[0]
    
                # hessians
                d2fdxdx = egrad(dfdx,0)
                d2fdxdy = egrad(dfdx,1)
                d2fdxdz = egrad(dfdx,2)
                d2fdydx = egrad(dfdy,0)
                d2fdydy = egrad(dfdy,1)
                d2fdydz = egrad(dfdy,2)
                d2fdzdx = egrad(dfdz,0)
                d2fdzdy = egrad(dfdz,1)
                d2fdzdz = egrad(dfdz,2)
    
                d2f1 = np.zeros((x.shape[0],3,3))
                for i in range(3):
                    for j in range(3):
                        derIndices = [i,j]
                        d2f1[:,i,j]=getSlaterBasisHigherDerivativeFD(n,l,m,a,x,derIndices,\
                                                                     numFDPoints, h)
    
                d2f2 = np.zeros((x.shape[0],3,3))
                d2f2[:,0,0] = d2fdxdx(x[:,0], x[:,1], x[:,2])
                d2f2[:,0,1] = d2fdxdy(x[:,0], x[:,1], x[:,2])
                d2f2[:,0,2] = d2fdxdz(x[:,0], x[:,1], x[:,2])
                d2f2[:,1,0] = d2fdydx(x[:,0], x[:,1], x[:,2])
                d2f2[:,1,1] = d2fdydy(x[:,0], x[:,1], x[:,2])
                d2f2[:,1,2] = d2fdydz(x[:,0], x[:,1], x[:,2])
                d2f2[:,2,0] = d2fdzdx(x[:,0], x[:,1], x[:,2])
                d2f2[:,2,1] = d2fdzdy(x[:,0], x[:,1], x[:,2])
                d2f2[:,2,2] = d2fdzdz(x[:,0], x[:,1], x[:,2])
    
                d2f3 = np.zeros((x.shape[0],3,3))
                for i in range(x.shape[0]):
                    u = torch.zeros(1,3, dtype=torch.float64)
                    u[0,:] = torch.tensor(x[i])
                    v = u.clone().requires_grad_()
                    d2f3[i] = hessian(f3,v).detach().numpy()[0,:,0,:]
        
                dfAbs = np.linalg.norm(df2,ord='fro')
                diffDf12 = np.linalg.norm(df1-df2,ord='fro')
                diffDf23 = np.linalg.norm(df2-df3,ord='fro')
                diffDf13 = np.linalg.norm(df1-df3,ord='fro')
                lossD12 =+ diffDf12
                lossD23 =+ diffDf23
                lossD13 =+ diffDf13
                #print(l,m, "Diff. abs. df 12 23 13:", diffDf12, diffDf23,\
                #      diffDf13)
                #print(l,m, "Diff. rel. df 12 23 13:", diffDf12/dfAbs,\
                #      diffDf23/dfAbs, diffDf13/dfAbs)
                
                d2fAbs = np.linalg.norm(d2f2)
                diffD2f12 = np.linalg.norm(d2f1-d2f2)
                diffD2f23 = np.linalg.norm(d2f2-d2f3)
                diffD2f13 = np.linalg.norm(d2f1-d2f3)
                lossDD12 =+ diffD2f12
                lossDD23 =+ diffD2f23
                lossDD13 =+ diffD2f13
                #print(l,m, "Diff. abs. d2f 12 23 13:", diffD2f12, diffD2f23, diffD2f13)
                #print(l,m, "Diff. rel. d2f 12 23 13:", diffD2f12/dfAbs,\
                #      diffD2f23/dfAbs, diffD2f13/dfAbs)

print("count", count)
print("lossD12", lossD12/(count*nx))
print("lossD23", lossD23/(count*nx))
print("lossD13", lossD13/(count*nx))
print("lossDD12", lossDD12/(count*nx))
print("lossDD23", lossDD23/(count*nx))
print("lossDD13", lossDD13/(count*nx))
#dfdx = grad(f)
#print(type(x))
#print(type(f))
#print(type(dfdx))
#print(f(x))
#print(vmap(dfdx)(x))
#y = lpmn_values(2,2,x, False)
#print(y[:,:,0])
