import numpy as np
import sys

# given three points in a file (each point's x,y,z in a separate line),
# evaluates the equation of the plane passing through the three points.
# The equation has the form
# ax + by + cz = d
# 

coordFilename = str(sys.argv[1])
dataFilename = str(sys.argv[2])
outFilename = dataFilename + "_relativePos"
data = np.loadtxt(dataFilename, dtype = np.float64)

coords = np.loadtxt(coordFilename, dtype = np.float64)
v1 = coords[0] - coords[1]
v2 = coords[0] - coords[2]

n = np.cross(v1,v2)
nNorm = np.linalg.norm(n, ord = 2)
n = n/nNorm
d = (coords[0]).dot(n)

print("Normal vector: ", n)
print("intercept:",d)

v1Norm = np.linalg.norm(v1, ord= 2)
xAxis = v1/v1Norm

v3 = np.cross(v1,n)
v3Norm = np.linalg.norm(v3, ord = 2)
yAxis = v3/v3Norm

dataRelPos = np.zeros(data.shape)
dataRelPos[:,3:] = data[:,3:]
npoints = data.shape[0]
for i in range(npoints):
    point = data[i,:3]
    dataRelPos[i,0] = (point - coords[0]).dot(xAxis)
    dataRelPos[i,1] = (point - coords[0]).dot(yAxis)
    dataRelPos[i,2] = (point - coords[0]).dot(n)

np.savetxt(outFilename, dataRelPos)

