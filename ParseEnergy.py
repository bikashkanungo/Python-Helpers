import sys
import os
from os.path import exists
import filecmp

dirfile = str(sys.argv[1])

rootdir = os.getcwd()
rootOutfilename = "out_NN_GGA_PBEBase_H2_eq_LiH_Ne_Li_N_C_xi2_no_aux_bse_qz_tz_5x80_epoch_16000_lr_1e-3"
stringToFind = "Total internal energy per atom"
stringNAtoms = "set NATOMS"
energyOut = open("energy_" + rootOutfilename, 'w')

fdir = open(dirfile,"r")
fout = open("out_" + dirfile, "w")
print("Parsing file: ", dirfile, file = fout)
print("----------------------------------------", file = fout)

lines = fdir.readlines()
for line in lines:
    systemdir = rootdir + "/" + line.strip()
    outfilename = systemdir + "/" + rootOutfilename
    out = open(outfilename, "r")
    outLines = out.readlines()
    natoms = 0
    foundNAtoms = False
    for outLine in outLines:
        if stringNAtoms in outLine:
            N = outLine.rfind("=")
            natoms = int(outLine[N+1:])
            foundNAtoms = True
            break

    if foundNAtoms == False:
        raise Exception("Unable to find the string " + stringNAtoms + " in \
                        file " + outfilename);
    for outLine in reversed(outLines):
        if stringToFind in outLine:
            energyPerAtom = float(outLine.split()[-1])
            print(line.strip(), energyPerAtom*natoms, file = energyOut)
            break

    out.close()

fdir.close()


