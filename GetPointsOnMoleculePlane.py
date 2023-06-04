import numpy as np
import sys

# given three points in a file (each point's x,y,z in a separate line),
# evaluates the equation of the plane passing through the three points.
# The equation has the form
# ax + by + cz = d
# 

coordFilename = str(sys.argv[1])
coords = np.loadtxt(coordFilename, dtype = np.float64)
v1 = coords[1] - coords[0]
v2 = coords[2] - coords[0]

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

xlo = -5.0
xhi = 5.0
ylo = -5.0
yhi = 5.0
h = 0.01
npx = (int)((xhi-xlo)/h) + 1
npy = (int)((yhi-ylo)/h) + 1

points = np.zeros((npx*npy,3))
pointsRelative = np.zeros((npx*npy,3))
for i in range(npx):
    x = xlo + i*h
    for j in range(npy):
        y = ylo + j*h
        points[i*npy + j] = x*xAxis + y*yAxis + coords[0]
        pointsRelative[i*npy + j, 0] = x
        pointsRelative[i*npy + j, 1] = y
        if abs(points[i*npy + j].dot(n)-d) > 1e-10 :
            print("Point outside of plane")

np.savetxt("PointsOnPlane", points)
np.savetxt("PointsOnPlaneRelative", pointsRelative)

print("xAxis", xAxis)
print("yAxis", yAxis)
print(v1Norm*xAxis + coords[0])
