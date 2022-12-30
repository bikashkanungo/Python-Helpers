import numpy as np
from scipy.linalg import lstsq

def getSymMat(N, scale):
    A = scale*np.random.rand(N,N)
    A = 0.5*(A + np.transpose(A))
    return A

def getEigSys(A):
    e, V = np.linalg.eig(A)
    return e, V

def setDirichletIdsZero(x, ids):
    for i in ids:
        x[i] = 0.0

def minresSolve(A, e, b, nIter, tol, dirichletIds):
    N = A.shape[0]
    x = np.zeros(N)
    bnorm = np.linalg.norm(b, ord=2)

    # Q stores the Lanczos vectors.
    # Create it with no columns
    Q = np.zeros((N,0))
    #Add normalized b as the first Lanczos vector
    Q = np.c_[Q, b/bnorm]
    alphas = []
    betas = []
    for i in range(nIter):
        q = Q[:,-1]
        v = np.matmul(A,q)
        setDirichletIdsZero(v, dirichletIds)
        alphas.append(q.dot(v))
        v = v - alphas[-1]*q
        if(i > 0):
            v = v - betas[-1]*Q[:,-2]

        betas.append(np.linalg.norm(v, ord=2))
        v = v/betas[-1]
        m = i + 1
        T = np.zeros((m+1,m))
        for j in range(m-1):
            T[j,j] = alphas[j]
            T[j,j+1] = betas[j]
            T[j+1,j] = betas[j]

        T[m-1, m-1] = alphas[m-1]
        T[m, m-1] = betas[m-1]

        U = Q
        W = np.c_[Q, v]
        C = np.matmul(np.transpose(W), U)
        C = T - e*C
        d = np.zeros(m+1)
        d[0] = bnorm
        y, res, rank, singularVals = lstsq(C, d, cond=None)
        x = np.matmul(Q, y)
        rnorm = np.linalg.norm(b-(np.matmul(A,x)-e*x), ord=2)
        print(i, rnorm)
        if(rnorm < tol):
            break

        else:
            #print(T)
            Q = W



N = 1000
nIter = 100
tol = 1e-9
dirichletIds = [0]
A = getSymMat(N, 1.0)
eigs, V = getEigSys(A)
e = eigs[0]
c = V[:,0]
b = np.random.rand(N)
b = b - (c.dot(b))*c
setDirichletIdsZero(b, dirichletIds)
print("b^Tc", b.dot(c))
x =  minresSolve(A, e, b, nIter, 1e-9, dirichletIds)
