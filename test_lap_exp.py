import math
import copy

ZERO_TOL = 1e-8

def getFDWeights2ndDer(stencilOrder1D):
    if stencilOrder1D == 1:
        return [1.0, -2.0, 1.0]

    elif stencilOrder1D == 2:
        return [-1.0/12.0, 4.0/3.0, -5.0/2.0, 4.0/3.0,-1.0/12.0]

    elif stencilOrder1D == 3:
        return [1.0/90.0,-3.0/20.0, 3.0/2.0,-49.0/18.0, 3.0/2.0, -3.0/20.0,
                1.0/90.0]

    elif stencilOrder1D == 4:
        return [-1.0/560.0, 8.0/315.0, -1.0/5.0, 8.0/5.0, -205.0/72.0, 8.0/5.0,
                -1.0/5.0, 8.0/315.0, -1.0/560.0]

    else:
        raise Exception('''FDWeights2ndDer not implemented beyond stencil size
                        4''')

    return

def radius(x):
    return math.sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])


def f(x, alpha, N):

    r = radius(x)
    return (r**(N-1))*math.exp(-alpha*r)


def fLap(x, alpha, N):
    r = radius(x)
    if r < ZERO_TOL:
        raise Exception("Laplacian of exp(-a*r) is undefined at r=0")
    else:
        e = math.exp(-alpha*r)
        t1 = 0.0
        if N > 1:
            t1 = N*(N-1)*(r**(N-3))

        t2 = -2.0*N*alpha*(r**(N-2))
        t3 = (alpha**2.0)*(r**(N-1))
        return (t1+t2+t3)*e

def fLapFD(x, alpha, N, spacing, stencilOrder, FDWeights):
    lap = 0.0
    fval0 = f(x,alpha,N)
    for i in range(3):
        for j in range(2*stencilOrder+1):
            if j == stencilOrder:
                fval = fval0

            else:
                xcopy = copy.deepcopy(x)
                xcopy[i] = x[i] + (j-stencilOrder)*spacing
                fval = f(xcopy, alpha, N)

            lap += fval*FDWeights[j]

    return lap/(spacing**2.0)


N = 1
alpha = 40.0
x_test = [1e-4, 1e-4, 1e-4]
r_test = radius(x_test)
h = 1e-5
stencilOrder = 3
FDWeights = getFDWeights2ndDer(stencilOrder)
lap = fLap(x_test, alpha, N)
lapFD = fLapFD(x_test, alpha, N, h, stencilOrder, FDWeights)
print("N:", N, "r:", r_test, "Stencil Order:", stencilOrder,  "Spacing:", h, "Lap. analytical:", lap, "Lap. FD", lapFD, 'Rel. Err.:', abs((lap-lapFD)/lap))
