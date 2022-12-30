import numpy as np



rotationMatFileName = "RotationMatrix"
R = np.loadtxt(rotationMatFileName, usecols=range(3))
coordsFileName = "Coordinates"
rotatedCoordsFileName = "RotatedCoordinates"
coordsFile = open(coordsFileName,"r")
rotatedCoordsFile = open(rotatedCoordsFileName, "w")
lines = coordsFile.readlines()
coords = []
for line in lines:
    coord = []
    for word in line.split():
        coord.append(float(word))

    coordNP = np.array(coord, dtype=float)
    coords.append(coordNP)

coordsFile.close()

for i in range(len(coords)):
    x = np.dot(R,coords[i])
    for j in range(len(x)):
      print(x[j], file = rotatedCoordsFile, end = " ")
   
    print(file = rotatedCoordsFile)

