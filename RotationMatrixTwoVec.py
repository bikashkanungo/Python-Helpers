import numpy as np

def normalizeVector(x):
    norm = np.linalg.norm(x)
    return x/norm;

def dotProduct(a,b):
    return np.dot(a,b)

def crossProduct(a, b):
    return np.cross(a,b)

def norm(x):
    return np.linalg.norm(x)

def vecCrossSkewSymMatrix(x):
    M = np.zeros([3,3], dtype=float)
    M[0][0] = 0.0
    M[0][1] = -x[2]
    M[0][2] = x[1]
    M[1][0] = x[2]
    M[1][1] = 0.0
    M[1][2] = -x[0]
    M[2][0] = -x[1]
    M[2][1] = x[0]
    M[2][2] = 0.0
    return M
    

vecFile = open("Vectors", "r")
lines = vecFile.readlines()
vecs = []
for line in lines:
    vec = []
    for word in line.split():
        vec.append(float(word))

    vecNP = np.array(vec, dtype=float)
    vecs.append(vecNP)

vecFile.close()

if len(vecs) !=2:
    raise Exception("Requires two vectors to determine the rotation matrix")

vecs[0] = normalizeVector(vecs[0])
vecs[1] = normalizeVector(vecs[1])
v = crossProduct(vecs[0], vecs[1])
c = dotProduct(vecs[0], vecs[1])
if abs(c+1.0) < 1e-10:
    raise("The two vectors point in opposite directions, cannot find a rotation matrix")

s = norm(v)
I = np.identity(3, dtype = float)
V = vecCrossSkewSymMatrix(v)
R = I + V + (np.matmul(V,V))/(1.0+c)
print("Rotation Matrix:\n", R)
