import sys

xyzFileName = sys.argv[1]
xyzFile = open(xyzFileName, 'r')
dftFEFileName = sys.argv[2]
dftFEFile = open(dftFEFileName, 'w')

lines = xyzFile


