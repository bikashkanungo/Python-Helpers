import sys
import os
from os.path import exists
import filecmp

dirfile = str(sys.argv[1])

rootdir = os.getcwd()
fdir = open(dirfile,"r")
fout = open("out_" + dirfile, "w")
print("Parsing file: ", dirfile, file = fout)
print("----------------------------------------", file = fout)

filesToCheck = ["parameters.prm", "coordinates.inp", "domainVectors.inp"]

lines = fdir.readlines()
for line in lines:
    os.chdir(rootdir + "/" + line.strip())
    print("Entered directory: " + os.getcwd())
    files = os.listdir(os.getcwd())
    for f in filesToCheck:
        if f not in files:
            print(f + " not found in " + line.strip(), file = fout)

fdir.close()
fout.close()


