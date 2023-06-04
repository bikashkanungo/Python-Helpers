import sys
import numpy as np


nspin = 2
inp = open(str(sys.argv[1]), 'r')
lines = inp.readlines()
nlines = len(lines)
topline = lines[0]
models = topline.split()
eigs = {}
nmodels = len(models)
for model in models:
    eigs[model] = [[],[]]

for i in range(1,nlines):
    line = lines[i]
    vals = line.split()
    nvals = len(vals)
    #check if both spin-up and spin-down eigenavalues exist
    if int(nvals/nmodels) == 2:
        print('true')
        for j in range(nmodels):
            model = models[j]
            eigs[model][0].append(float(vals[2*j]))
            eigs[model][1].append(float(vals[2*j+1]))
    else:
        print('false')
        for j in range(nmodels):
            model = models[j]
            eigs[model][0].append(float(vals[j]))

print(eigs)
fermiLevelExact = max(eigs['exact'][0] + eigs['exact'][1])
exactEigSum = [0.0,0.0]
exactEigSum[0] = sum([abs(x) for x in eigs['exact'][0]])
exactEigSum[1] = sum([abs(x) for x in eigs['exact'][1]])
print('Fermilevel exact', fermiLevelExact)
diff = {}
for model in models:
    fermiLevelModel = max(eigs[model][0] + eigs[model][1])
    print('Fermilevel', model, fermiLevelModel)
    diff[model] = [0.0]*nspin
    shift = fermiLevelExact - fermiLevelModel
    for ispin in range(nspin):
        neigs = len(eigs[model][ispin])
        for ieig in range(neigs):
            eigExact = eigs['exact'][ispin][ieig]
            eigModel = eigs[model][ispin][ieig] + shift
            diff[model][ispin] += abs(eigExact - eigModel) #/abs(eigExact)


for model in models:
    print('Diff0', model, 'Abs.:', diff[model][0], 'Rel.:', diff[model][0]/len(eigs[model][0]))
    print('Diff1', model, 'Abs.:', diff[model][1], 'Rel.:', diff[model][1]/len(eigs[model][1]))















