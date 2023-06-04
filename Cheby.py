import numpy as np

def Tm(x, m):
    if m == 0:
        return np.ones_like(x)

    elif m == 1:
        return x

    elif m > 1:
        return 2.0*x*Tm(x,m-1) - Tm(x,m-2)

    else:
        raise Exception("Invalid order " + str(m) + " passed Chebyshev polynomial")



m = 10
a = -2.
b=  1.0
x = np.linspace(a, b, 100)
y = Tm(x, m)

outFilename = "Cheby_" + str(m)
out = open(outFilename, "w")
N = x.shape[0]
for i in range(N):
    print(x[i], y[i], file = out)


