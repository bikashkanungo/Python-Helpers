import sys

def splitValues(line):
    x = line.split()
    aStr = (x[-2].split('('))[1]
    bStr = (x[-1].split(')'))[0]
    absVal = aStr.split(',')[0]
    relVal = bStr.split(',')[0]
    
    return absVal,relVal

inp = open(str(sys.argv[1]),'r')
lines = inp.readlines()
nlines = len(lines)
nlinesPerModel = 15
nspins = 2

keyword = "Printing VXC Information for model: "
models = ["pw92", "pbe", "scan", "scan0", "b3lyp"]
vxcErrAbs = {}
vxcErrRel = {}
vxcGradErrAbs = {}
vxcGradErrRel = {}


for model in models:
    vxcErrAbs[model] = []
    vxcErrRel[model] = []
    vxcGradErrAbs[model] = []
    vxcGradErrRel[model] = []
    for i in range(nlines):
        if keyword+model == lines[i].strip():
            for j in range(i+1,i+nlinesPerModel): 
                if "VXCExactDiff L2 Norm (Absolute, Relative):" in  lines[j]:
                    a,r = splitValues(lines[j])
                    vxcErrAbs[model].append(a)
                    vxcErrRel[model].append(r)
                if "VXCExactGradDiff L2 Norm (Absolute, Relative):" in  lines[j]:
                    a,r = splitValues(lines[j])
                    vxcGradErrAbs[model].append(a)
                    vxcGradErrRel[model].append(r)

            i = i+nlinesPerModel

outAbs = open("AbsErr","w")
outRel = open("RelErr","w")

offset = 2
for model in models:
    err = vxcErrRel[model]
    gradErr = vxcGradErrRel[model]
    for iSpin in range(nspins):
        print(model+"-"+str(iSpin), "\t",  err[iSpin], "\t",  gradErr[iSpin],\
              "\t", err[offset+iSpin], "\t", gradErr[offset+iSpin], file = outRel)

    err = vxcErrAbs[model]
    gradErr = vxcGradErrAbs[model]
    for iSpin in range(nspins):
        print(model+"-"+str(iSpin), "\t" , err[iSpin], "\t", gradErr[iSpin],\
              "\t", err[offset+iSpin], "\t", gradErr[offset+iSpin], file = outAbs)

outAbs.close()
outRel.close()




