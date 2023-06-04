import sys
import os
from os.path import exists
import filecmp
import csv

def getSymbolToAtomicNumber():
  pt = open('PeriodicTable.csv')
  ptcsv = csv.reader(pt)
  allRows = list(ptcsv)
  heading = allRows[0]
  elements = allRows[1:]
  pt.close()
  symbolCol = heading.index('Symbol')
  atomicNumberCol = heading.index('NumberofElectrons')
  symbols = {}
  for e in elements:
    symbols[e[symbolCol]] = int(e[atomicNumberCol])

  return symbols


if(len(sys.argv) != 4):
    print("Usage: python loop_dirs_coord.py <dirs-file> <output-unit angs or bohr?> <output-coord-file>")

dirfile = str(sys.argv[1])
outUnit = str(sys.argv[2])
outCoordFilename = str(sys.argv[3])
conversionFactor = 1.0
if outUnit.upper() == 'BOHR' or outUnit.upper() == 'AU':
    conversionFactor = 1.8897259886

rootdir = os.getcwd()
fdir = open(dirfile,"r")
lines = fdir.readlines()
fdir.close()
xyzFilename = 'struc.xyz'
symbolToAtomicNumber = getSymbolToAtomicNumber()
symbols = [x for x in symbolToAtomicNumber]
symbolsUpper = [x.upper() for x in symbolToAtomicNumber]
for line in lines:
    sysName, charge, mult = line.split()
    os.chdir(rootdir + "/" + sysName)
    xyzf = open(xyzFilename, 'r')
    #skip the first two lines
    xyzLines = (xyzf.readlines())[2:]
    outCoordFile = open(outCoordFilename, 'w')
    for xyz in xyzLines:
        words = xyz.split()
        xyzAtomSymbol = words[0]
        pos = symbolsUpper.index(xyzAtomSymbol.upper())
        atomSymbol = symbols[pos]
        atomicNumber = symbolToAtomicNumber[atomSymbol]
        coords = [conversionFactor*float(x) for x in words[1:]]
        print(atomicNumber, atomicNumber, coords[0], coords[1], coords[2], file = outCoordFile)

    outCoordFile.close()
