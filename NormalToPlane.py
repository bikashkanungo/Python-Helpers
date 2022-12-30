import numpy as np

def dotProduct(a, b):
    return  np.dot(a,b) 

def crossProduct(a, b):
    return  np.cross(a,b) 

def normalizeVector(x):
    norm = np.linalg.norm(x)
    return x/norm

def getPlaneEq(x,n):
    d = np.dot(x,n)
    return (n[0], n[1], n[2], d)


coordsFile = open("Coords","r")
lines = coordsFile.readlines()
coords = []
for line in lines:
    coord = []
    for word in line.split():
        coord.append(float(word))

    coordNP = np.array(coord, dtype = float)
    coords.append(coordNP)

coordsFile.close()

if len(coords) != 3:
    raise Exception("Requires 3 points to determine the normal to the plane")

v1 = coords[0] - coords[1]
v2 = coords[1] - coords[2]
n = crossProduct(v1,v2)
n = normalizeVector(n)
print("Normal", n)
a,b,c,d = getPlaneEq(coords[0],n)
print("Eq. of plane:", a, b, c, d)

coordsFileChk = open("Coordinates","r")
lines = coordsFileChk.readlines()
coords = []
for line in lines:
    coord = []
    for word in line.split():
        coord.append(float(word))

    coordNP = np.array(coord, dtype = float)
    coords.append(coordNP)

coordsFileChk.close()

for i in range(len(coords)):
    if abs(np.dot(coords[i],n)-d) > 1e-6:
        print(coords[i], "lies outside the plane")



