import sys
import numpy as np

## command line args: <outfile> <fractional-occupancy-file> <output-density-matrix-file-name> <numBasis>

def getMOs(keyword, fname, numBasis):
    f = open(fname, 'r')
    numSkipLines = 1 
    lines = f.readlines()
    startLineNum = 0
    for num, line in enumerate(lines):
        if keyword in line:
            startLineNum = num + 1
            break

    MOs = np.zeros((numBasis, numBasis), dtype=np.float64)
    MOsCount = 0
    count = 0
    while MOsCount < numBasis:
        startId = startLineNum+count+numSkipLines
        endId = startId+numBasis
        MOsBlockLines = lines[startId:endId]
        numMOsInBlock = len(MOsBlockLines[0].split())-1
        #print("StartId", startId, "MO first line", MOsBlockLines[0], "MOsCount", MOsCount, "MOsInBlock", numMOsInBlock)
        for i in range(numBasis):
            words = MOsBlockLines[i].split()
            MOsLine = [float(x) for x in words[1:]]
            for j in range(numMOsInBlock):
                MOs[i][MOsCount+j] = MOsLine[j]

        count += numSkipLines+numBasis
        MOsCount += numMOsInBlock

    f.close()
    return MOs


numArgs = len(sys.argv)
if numArgs != 5:
    raise Exception('''Usage: python MOToDensityMatrixFromOutFile <file-to-read> \
                    <fractional-occupancy-file-name> \
                    <output-density-matrix-file-name> \
                    <numBasis>''')

inFname = str(sys.argv[1])
fracAlphaFname = str(sys.argv[2])+"0"
fracBetaFname = str(sys.argv[2])+"1"
DMAlphaF = open(str(sys.argv[3])+"0", 'w')
DMBetaF = open(str(sys.argv[3])+"1", 'w')
numBasis = int(sys.argv[4])

fracAlphaFile = open(fracAlphaFname, 'r')
fracAlphaLine = fracAlphaFile.readlines()[0]
XAlpha = [float(x) for x in fracAlphaLine.split()]
fracAlpha = np.diag(XAlpha)

fracBetaFile = open(fracBetaFname, 'r')
fracBetaLine = fracBetaFile.readlines()[0]
XBeta = [float(x) for x in fracBetaLine.split()]
fracBeta = np.diag(XBeta)


numFilledMOsAlpha = len(XAlpha)
numFilledMOsBeta = len(XBeta)

MOsAlpha = getMOs("Final Alpha MO Coefficients", inFname, numBasis)
MOsAlphaFilled = MOsAlpha[:,:numFilledMOsAlpha]
MOsBeta = getMOs("Final Beta MO Coefficients", inFname, numBasis)
MOsBetaFilled = MOsBeta[:,:numFilledMOsBeta]


MOsAlphaFilledTrans = MOsAlphaFilled.transpose()
CAlpha = np.matmul(MOsAlphaFilled, fracAlpha)
DAlpha = np.matmul(CAlpha, MOsAlphaFilledTrans)

MOsBetaFilledTrans = MOsBetaFilled.transpose()
CBeta = np.matmul(MOsBetaFilled, fracBeta)
DBeta = np.matmul(CBeta, MOsBetaFilledTrans)

for i in range(numBasis):
    for j in range(numBasis):
        print(DAlpha[i][j], end=" ", file = DMAlphaF)
        print(DBeta[i][j], end=" ", file = DMBetaF)


    print(file = DMAlphaF)
    print(file = DMBetaF)

DMAlphaF.close()
DMBetaF.close()
