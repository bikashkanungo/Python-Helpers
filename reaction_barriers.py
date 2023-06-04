import sys
import os
from os.path import exists
import filecmp
import copy

def parseEnergy(sysNames, rootoutfilename):
    rootdir = os.getcwd()
    stringToFind = "Total internal energy per atom"
    stringNAtoms = "set NATOMS"
    energies = {}

    for sys in sysNames:
        systemdir = rootdir + "/" + sys
        outfilename = systemdir + "/" + rootoutfilename
        print('parsing:', outfilename)
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
                energies[sys] = energyPerAtom*natoms
                break

        out.close()

    return energies


hartreeToKcalPerMol = 627.5096080305927
kcalPerMolToHartree = 0.0015936010974213599
reactionsFile = open(str(sys.argv[1]),'r')
reactionIdsFile = open(str(sys.argv[2]), 'r')
outfilenameToParse = str(sys.argv[3])
reactionslines = reactionsFile.readlines()
reactionIdsLines = reactionIdsFile.readlines()
reactionsFile.close()
reactionIdsFile.close()

words = reactionIdsLines[0].split()
ids = [int(x) for x in words]

nreactionsline = len(reactionslines)
reactions = []
systems = set()

offset = 1 #skip the first line
for index in ids:
    words = reactionslines[index+offset].split()
    # in the reactions file the first column in the reaction id,
    # the next k columns define the k systems in the reaction (note that k varies from one reaction to another),
    # then comes the stoichiometry (1 or -1 for eachs system). To extract the systems, we only need to
    # extract the columns between 2nd and the column where the stoichiometry begins (i.e., occurence of either 1 or -1)
    startStoichiometryColId = words.index('-1')
    sysNames = words[1:startStoichiometryColId]
    nsys = len(sysNames)
    stoch = [float(x) for x in words[startStoichiometryColId:startStoichiometryColId+nsys]]
    ref = float(words[startStoichiometryColId+nsys])*kcalPerMolToHartree
    pbe = float(words[startStoichiometryColId+nsys+1])*kcalPerMolToHartree + ref
    reaction = {'id':index, 'systems':sysNames, 'stochiometry':stoch, 'ref':ref, 'pbe':pbe}
    reactions.append(reaction)
    for s in sysNames:
        systems.add(s)

print(reactions)

energies = parseEnergy(list(systems), outfilenameToParse)
reaction_barriers = {}
for reaction in reactions:
    index = reaction['id']
    sysNames = reaction['systems']
    stoch= reaction['stochiometry']
    nsys = len(sysNames)
    barrier = 0.0
    for i in range(nsys):
        sysName = sysNames[i]
        barrier += stoch[i]*energies[sysName]

    barrier_ref = reaction['ref']
    barrier_pbe = reaction['pbe']
    reaction_barriers[index] = {'model': barrier, 'pbe': barrier_pbe,  'ref': barrier_ref}

mad = 0.0
mad_pbe = 0.0
for b in reaction_barriers:
    mad += abs(reaction_barriers[b]['model']-reaction_barriers[b]['ref'])
    mad_pbe += abs(reaction_barriers[b]['pbe']-reaction_barriers[b]['ref'])

mad = mad/len(reaction_barriers)
mad_pbe = mad_pbe/len(reaction_barriers)

print(reaction_barriers)
print('MAD', mad)
print('MAD PBE', mad_pbe)
