import sys
import os
from os.path import exists
import filecmp
import copy
import random

sysfilename = str(sys.argv[1])
reactionsfilename = str(sys.argv[2])
nselect = int(sys.argv[3])
sysfile = open(sysfilename,"r")
reactionsfile = open(reactionsfilename,"r")
syslines = sysfile.readlines()
reactionslines = reactionsfile.readlines()
sysfile.close()
reactionsfile.close()
sys = {}
for line in syslines:
    sysName, charge, mult = line.split()
    charge = float(charge)
    mult = int(mult)
    sys[sysName] = {'charge':charge, 'mult':mult}

nreactionsline = len(reactionslines)
reactions = []
#skip the first line
for i in range(1,nreactionsline):
    words = reactionslines[i].split()
    print(words)
    # in the reactions file the first column in the reaction id,
    # the next k columns define the k systems in the reaction (note that k varies from one reaction to another),
    # then comes the stoichiometry (1 or -1 for eachs system). To extract the systems, we only need to
    # extract the columns between 2nd and the column where the stoichiometry begins (i.e., occurence of either 1 or -1)
    startStoichiometryColId = words.index('-1')
    sysNames = words[1:startStoichiometryColId]
    nsys = len(sysNames)
    stoch = [float(x) for x in words[startStoichiometryColId:startStoichiometryColId+nsys]]
    reaction = {'id':i-1, 'systems':sysNames, 'stochiometry':stoch, 'ref':float(words[startStoichiometryColId+nsys])}
    reactions.append(reaction)

print(reactions)

nreactions = len(reactions)
reactions_neutral = []
for reaction in reactions:
    sysNames = reaction['systems']
    isNeutral = True
    for s in sysNames:
        if sys[s]['charge'] != 0:
            isNeutral = False
            break

    if isNeutral:
        reactions_neutral.append(reaction)


print('\n\n')
print(reactions_neutral)

nreactions_neutral = len(reactions_neutral)
nselect = min(nselect, nreactions_neutral)
nselect = int(nselect/2)*2
print('nreactions_neutral', nreactions_neutral, 'nselect', nselect)
count = 0
selectedIds = []
count = 0
while count < nselect:
    index = random.randint(0,nreactions_neutral-1)
    if index not in selectedIds:
        if index % 2 == 0:
            index_pair = index+1

        else:
            index_pair = index-1

        selectedIds.append(reactions_neutral[index]['id'])
        selectedIds.append(reactions_neutral[index_pair]['id'])
        count += 2

print('ids', selectedIds)
sys_selected = set()
for index in selectedIds:
    sysNames = reactions_neutral[index]['systems']
    for s in sysNames:
        sys_selected.add(s)


print(sys_selected)
outfile = open('selected_reactions_sys','w')
for index in selectedIds:
    print(index, file = outfile, end = " ")

print('\n',file=outfile)
for s in sys_selected:
    print(s, file = outfile)

outfile.close()

