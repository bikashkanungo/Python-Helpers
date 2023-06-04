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

#def minresSolveOpt(A, e, b, nv, nIter, tol):
#    N = A.shape[0]
#    x = np.zeros((N,nv))
#    converged = [False]*nv
#    allConverged = False
#
#    r1 = np.matmul(A,x)-e*x
#    r1 = b - r1
#
#    y = np.copy(r1)
#
#    beta1 = np.diag(np.matmul(np.transpose(r1),y))
#    beta1 = beta1**0.5
#
#    eps = np.full((nv), 1e-15)
#    oldb=np.zeros(nv)
#    beta = np.copy(beta1)
#    dbar = np.zeros(nv)
#    epsln = np.zeros(nv)
#    oldeps = np.copy(nv)
#    qrnorm = np.copy(beta1)
#    phi = np.zeros(nv)
#    phibar = np.copy(beta1)
#    rhs1 = np.copy(beta1)
#    rhs2 = np.zeros(nv)
#    tnorm2 = np.zeros(nv)
#    ynorm2 = np.zeros(nv)
#    cs = np.full((nv),-1.0)
#    sn = np.zeros(nv)
#    gmax = np.zeros(nv)
#    gmin = np.full((nv), 1e20)
#    alpha = np.zeros(nv)
#    gamma = np.zeros(nv)
#    delta = np.zeros(nv)
#    gbar = np.zeros(nv)
#    z = np.zeros(nv)
#
#    w = np.zeros((N,nv))
#    w1 = np.zeros((N,nv))
#    w2 = np.zeros((N,nv))
#    r2 = np.zeros((N,nv))
#    v = np.zeros((N,nv))
#
#    r2 = np.copy(r1)
#    xtmp = np.copy(x)
#
#    for i in range(nIter):
#        v = y/beta
#        y = np.matmul(A,v) - e*v
#        if i > 0:
#            y = y - (beta/oldb)*r1
#
#        alpha = np.diag(np.matmul(np.transpose(v),y))
#        y = y -alpha/beta*r2
#        r1 = np.copy(r2)
#        r2 = np.copy(y)
#        y = np.copy(r2)
#        oldb = np.copy(beta)
#        beta = np.diag(np.matmul(np.transpose(r2),y))
#
#        beta = beta**0.5
#        tnorm2 = tnorm2 + alpha**2.0 + oldb**2.0 + beta**2.0
#
#        oldeps = np.copy(epsln)
#        delta = cs*dbar + sn*alpha
#        gbar = sn*dbar - cs*alpha
#        epsln = sn*beta
#        dbar = -cs*beta
#        root = (gbar**2.0 + dbar**2.0)**0.5
#        Arnorm = phibar*root
#
#        gamma = (gbar**2.0 + beta**2.0)**0.5
#        cs = gbar/gamma
#        sn = beta/gamma
#        phi = cs*phibar
#        phibar = sn*phibar
#
#        denom = 1./gamma
#        w1 = np.copy(w2)
#        w2 = np.copy(w)
#        w = -oldeps*w1 - delta*w2
#        w = denom*v + denom*w
#        xtmp = xtmp + phi*w
#
#        gmax = np.maximum(gmax, gamma)
#        gmin = np.minimum(gmin, gamma)
#        z = rhs1/gamma
#        rhs1 = rhs2 - delta*z
#        rhs2 = -epsln*z
#
#        Anorm = tnorm2**0.5
#        ynorm2 = np.diag(np.matmul(np.transpose(xtmp),xtmp))
#        ynorm = ynorm2**0.5
#        epsa = Anorm*eps
#        epsx = epsa*ynorm
#        epsr = Anorm*ynorm*tol
#        diag = np.copy(gbar)
#        qrnorm = np.copy(phibar)
#        rnorm = np.copy(qrnorm)
#
#        print(i, rnorm)
#
#        for j in range(nv):
#            if rnorm[j] < tol and converged[j]==False:
#                converged[j] = True
#                x[:,j] = xtmp[:,j]
#
#        if all(converged):
#            allConverged = True
#            break
#
#    return x



def minresSolveOpt(A, e, b, nv, nIter, tol):
    N = A.shape[0]
    x = np.zeros((N,nv))
    converged = [False]*nv
    allConverged = False

    r1 = np.matmul(A,x)-e*x
    r1 = b - r1

    y = np.copy(r1)

    beta1 = np.diag(np.matmul(np.transpose(r1),y))
    beta1 = beta1**0.5

    eps = np.full((nv), 1e-15)
    oldb=np.zeros(nv)
    beta = np.copy(beta1)
    dbar = np.zeros(nv)
    epsln = np.zeros(nv)
    oldeps = np.copy(nv)
    qrnorm = np.copy(beta1)
    phi = np.zeros(nv)
    phibar = np.copy(beta1)
    cs = np.full((nv),-1.0)
    sn = np.zeros(nv)
    alpha = np.zeros(nv)
    gamma = np.zeros(nv)
    delta = np.zeros(nv)
    gbar = np.zeros(nv)

    w = np.zeros((N,nv))
    w1 = np.zeros((N,nv))
    w2 = np.zeros((N,nv))
    r2 = np.zeros((N,nv))
    v = np.zeros((N,nv))

    r2 = np.copy(r1)
    xtmp = np.copy(x)

    for i in range(nIter):
        v = y/beta
        y = np.matmul(A,v) - e*v
        if i > 0:
            y = y - (beta/oldb)*r1

        alpha = np.diag(np.matmul(np.transpose(v),y))
        y = y -alpha/beta*r2
        r1 = np.copy(r2)
        r2 = np.copy(y)
        y = np.copy(r2)
        oldb = np.copy(beta)
        beta = np.diag(np.matmul(np.transpose(r2),y))

        beta = beta**0.5

        oldeps = np.copy(epsln)
        delta = cs*dbar + sn*alpha
        gbar = sn*dbar - cs*alpha
        epsln = sn*beta
        dbar = -cs*beta

        gamma = (gbar**2.0 + beta**2.0)**0.5
        gamma = np.maximum(gamma, eps)
        cs = gbar/gamma
        sn = beta/gamma
        phi = cs*phibar
        phibar = sn*phibar

        denom = 1./gamma
        w1 = np.copy(w2)
        w2 = np.copy(w)
        w = -oldeps*w1 - delta*w2
        w = denom*v + denom*w
        xtmp = xtmp + phi*w

        qrnorm = np.copy(phibar)
        rnorm = np.copy(qrnorm)

        print(i, rnorm)

        for j in range(nv):
            if rnorm[j] < tol and converged[j]==False:
                converged[j] = True
                x[:,j] = xtmp[:,j]

        if all(converged):
            allConverged = True
            break

    return x


def minresSolve(A, e, b, nv, nIter, tol):
    N = A.shape[0]
    x = np.zeros((N,nv))
    bnorm = np.zeros(nv)
    converged = [False]*nv
    allConverged  = False
    bnorm = np.diag(np.matmul(np.transpose(b),b))**0.5
    #for i in range(nv):
    #    bnorm[i] = np.linalg.norm(b[:,i], ord=2)


    # Q stores the Lanczos vectors.
    # Create it with no columns
    Q = np.zeros((N,nv,0))
    #Add normalized b as the first Lanczos vector
    Q = np.concatenate((Q, (b/bnorm)[:,:,np.newaxis]), axis=-1)
    alphas = []
    betas = []
    rnorm = np.zeros(nv)
    for i in range(nIter):
        q = Q[:,:,-1]
        v = np.matmul(A,q)-e*q
        #setDirichletIdsZero(v, dirichletIds)
        alpha = np.diag(np.matmul(np.transpose(q),v))
        alphas.append(alpha)
        v = v - alphas[-1]*q
        if i > 0:
            qPrev = Q[:,:,-2]
            v = v - betas[-1]*qPrev

        beta = np.diag(np.matmul(np.transpose(v),v))**0.5 + 1e-15
        betas.append(beta)
        v = v/beta
        #print(i, "alpha", alpha)
        #print(i, "beta", beta)
        m = i + 1
        for j in range(nv):
            T = np.zeros((m+1,m))
            for k in range(m-1):
                T[k,k] = alphas[k][j]
                T[k,k+1] = betas[k][j]
                T[k+1,k] = betas[k][j]

            T[m-1, m-1] = alphas[m-1][j]
            T[m, m-1] = betas[m-1][j]

            U = Q[:,j,:]
            #W = np.c_[U, v[:,j]]
            #Ttmp = np.matmul(np.transpose(W),A)
            #Ttmp = np.matmul(Ttmp, U)
            #print(i, j, "T err", np.linalg.norm(T-Ttmp)/np.linalg.norm(T))
            #C = np.matmul(np.transpose(W), U)
            C = T #- e[j]*C
            d = np.zeros(m+1)
            d[0] = bnorm[j]
            y, res, rank, singularVals = lstsq(C, d, cond=None)
            xtmp = np.matmul(U, y)
            rnorm[j] = np.linalg.norm(b[:,j]-(np.matmul(A,xtmp)-e[j]*xtmp), ord=2)
            if(rnorm[j] < tol and converged[j] == False):
                x[:,j] = xtmp
                converged[j] = True

        print(i, rnorm)
        if all(converged):
            allConverged = True
            break

        else:
            Q = np.concatenate((Q, v[:,:,np.newaxis]), axis=-1)


    return x





N = 1000
nv = 3
nIter = 200
tol = 1e-9
dirichletIds = []
A = getSymMat(N, 1.0)
eigs, V = getEigSys(A)
e = eigs[0:nv]
c = V[:,0:nv]
for j in range(nv):
    print(e[j], c[:,j].dot(np.matmul(A,c[:,j])))

b = np.random.rand(N,nv)
for i in range(nv):
    b[:,i] = b[:,i] - (c[:,i].dot(b[:,i]))*c[:,i]
    print("ivec", i, "b^Tc", b[:,i].dot(c[:,i]))
#setDirichletIdsZero(b, dirichletIds)
x1 =  minresSolve(A, e, b, nv, nIter, 1e-9)
x2 =  minresSolveOpt(A, e, b, nv, nIter, 1e-9)
print("x1-x2", np.linalg.norm(x1-x2))
