import sys
import os
from os.path import exists
import filecmp
import copy

def getMeshParams(Z):
    meshParams = {'feorder': 5, 'mesh_around_atom' : 0.2,  'atom_ball_radius': 1.5, 'mesh_at_atom': 0.02}
    meshParams['feorder_electrostat'] = meshParams['feorder'] + 2
    if Z > 10:
        meshParams['feorder'] = 5
        meshParams['feorder_electrostat'] = meshParams['feorder'] + 2
        meshParams['mesh_around_atom'] = 0.2
        meshParams['atom_ball_radius'] = 1.5
        meshParams['mesh_at_atom'] = 0.02

    return meshParams

def getAtomicNumbers(coordsFilename):
    f = open(coordsFilename, 'r')
    lines = f.readlines()
    f.close()
    atomicNums = []
    for line in lines:
        words = line.split()
        atomicNums.append(int(words[0]))

    return atomicNums



dirfile = str(sys.argv[1])
paramsFilename = "parameters_sample.prm"
coordsFilename = "coordinates.inp"

rootdir = os.getcwd()
fdir = open(dirfile,"r")
lines = fdir.readlines()
fdir.close()

paramsFile = open(paramsFilename, 'r')
paramsLines = paramsFile.readlines()
paramsFile.close()

# decides the number of buffer states in Chebyshev filtering
# total number of states is set = N_electron*(1 + buffer_frac)
buffer_frac = 0.2

paramsNewFilename = "parameters.prm"

keysToWords = {"natoms" : "set NATOMS",
              "ntypes" : "set NATOM TYPES",
              "max_iter" : "set MAXIMUM ITERATIONS",
              "temp" : "set TEMPERATURE",
              "scf_tol" : "set TOLERANCE",
              "cheb_tol": "set CHEBYSHEV FILTER TOLERANCE",
              "mesh_around_atom" : "set MESH SIZE AROUND ATOM",
              "atom_ball_radius" : "set ATOM BALL RADIUS",
              "mesh_at_atom" : "set MESH SIZE AT ATOM",
              "feorder" : "set POLYNOMIAL ORDER",
              "feorder_electrostat": "set POLYNOMIAL ORDER ELECTROSTATICS",
              "mult" : "set SPIN POLARIZATION",
              "start_mag" : "set START MAGNETIZATION",
              "norb": "set NUMBER OF KOHN-SHAM WAVEFUNCTIONS"}

keysDefaultVals = {"natoms" : 0,
              "ntypes" : 0,
              "max_iter" : 100,
              "temp" : 100,
              "scf_tol" : 1e-3,
              "cheb_tol": 1e-2,
              "mesh_around_atom" : 0.2,
              "atom_ball_radius" : 1.5,
              "mesh_at_atom" : ["set MESH SIZE AT ATOM", 0.02],
              "feorder" : 5,
              "feorder_electrostat" : 7,
              "mult" : 0,
              "start_mag" : 0.0,
              "norb": 10}

keys = list(keysToWords)
keywords = list(keysToWords.values())
nkeys = len(keys)
for line in lines:
    sysName, charge, mult = line.split()
    charge = float(charge)
    mult = int(mult)
    os.chdir(rootdir + "/" + sysName)
    keysVals = copy.deepcopy(keysDefaultVals)
    if mult != 1:
        keysVals['mult'] = 1
        keysVals['start_mag'] = 0.1
        keysVals['temp'] = 1


    atomicNums = getAtomicNumbers(coordsFilename)
    nelectron = sum(atomicNums) - charge
    atomicNumsSet = set(atomicNums)
    keysVals['natoms'] = len(atomicNums)
    keysVals['ntypes'] = len(atomicNumsSet)
    keysVals['norb'] = int(nelectron*1.2 + 5)

    maxAtomicNum = max(atomicNums)
    meshParams = getMeshParams(maxAtomicNum)
    keysVals['feorder'] = meshParams['feorder']
    keysVals['feorder_electrostat'] = meshParams['feorder_electrostat']
    keysVals['mesh_around_atom'] = meshParams['mesh_around_atom']
    keysVals['mesh_at_atom'] = meshParams['mesh_at_atom']
    keysVals['atom_ball_radius'] = meshParams['atom_ball_radius']


    fparamsNew = open(paramsNewFilename, 'w')
    for paramLine in paramsLines:
        lineToPrint = paramLine.rstrip()
        equalToPos = paramLine.find("=")
        if equalToPos != -1:
            keyword = (paramLine[:equalToPos]).strip()
            for i in range(nkeys):
                if keyword == keywords[i]:
                    lineToPrint = paramLine[:equalToPos] + " = " + str(keysVals[keys[i]])

        print(lineToPrint, file = fparamsNew)


    fparamsNew.close()
