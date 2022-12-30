import numpy as np

numMOs = 2
MOFname = "MO"
fracOpsFname = "frac"
DMFname = "D"
DMFile = open(DMFname, 'w') 
MOs = np.loadtxt(MOFname, dtype=np.float64, usecols=range(numMOs))
MOsTrans = MOs.transpose()
fracOps = np.loadtxt(fracOpsFname, dtype=np.float64, usecols=range(numMOs))
F = np.diag(fracOps)
N = MOs.shape[0]
X = np.matmul(MOs, F)
D = np.matmul(X, MOsTrans)

for i in range(N):
    for j in range(N):
        print(D[i][j], end=" ", file = DMFile)

    print(file = DMFile)
