import sys
import numpy as np

## command line args: <outfile> <fractional-occupancy-file> <output-density-matrix-file-name> <numBasis>

def getMat(keyword, fname, numBasis):
    f = open(fname, 'r')
    numSkipLines = 1 
    lines = f.readlines()
    startLineNum = 0
    for num, line in enumerate(lines):
        if keyword in line:
            startLineNum = num + 1
            break

    Mat = np.zeros((numBasis, numBasis), dtype=np.float64)
    basisCount = 0
    count = 0
    while basisCount < numBasis:
        startId = startLineNum+count+numSkipLines
        endId = startId+numBasis
        BlockLines = lines[startId:endId]
        numBasisInBlock = len(MOsBlockLines[0].split())-1
        #print("StartId", startId, "MO first line", MOsBlockLines[0], "MOsCount", MOsCount, "MOsInBlock", numMOsInBlock)
        for i in range(numBasis):
            words = BlockLines[i].split()
            line = [float(x) for x in words[1:]]
            for j in range(numBasisInBlock):
                Mat[i][basisCount+j] = line[j]

        count += numSkipLines+numBasis
        basisCount += numBasisInBlock

    f.close()
    return Mat


numArgs = len(sys.argv)
if numArgs != 4:
    raise Exception('''Usage: python ExtractMatrixFromOutFile <file-to-read> \
                    <output-file-name> \
                    <numBasis>''')

keyword = "Overlap Matrix" 
inFname = str(sys.argv[1])
outFname = str(sys.argv[2])
numBasis = int(sys.argv[3])


S = getMat(keyword, inFname, numBasis)
np.savetxt(S, outFname) 
