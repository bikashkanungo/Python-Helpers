import sys


inp = open(str(sys.argv[1]), 'r')
lines = inp.readlines()
nlines = len(lines)
systems = lines[0].split()
errNames = lines[1].split()
print(systems)
print(errNames)
nerrs = len(errNames)
errs = {}
nsys = len(systems)

modelPrintOrder = ['b3lyp', 'scan0', 'scan', 'pbe', 'pw92']
systemPrintOrder = ['Li', 'C', 'N', 'O', 'CN', 'CH2']

for iline in range(2, nlines):
    isys = 0
    icol = 0
    words = lines[iline].split()
    model = (words[0])[:-2]
    errs[model] = {}
    for system in systems:
        errs[model][system] = {}
        icol += 1
        for err in errNames:
            errs[model][system][err] = float(words[icol])
            icol += 1

print(errs)

#print e1 and e2 consequtively

print ('\n\n Printing e3 and e4')
print('-------------------------\n\n')
for model in modelPrintOrder:
    print(model.upper(), end='')
    for system in systemPrintOrder:
        print(' & ', errs[model][system]['e3'], ' & ', errs[model][system]['e4'], end = '')

    print(' \\\\ \\hline')


print ('\n\n Printing e1 and e2')
print('-------------------------\n\n')
for model in modelPrintOrder:
    print(model.upper(), end='')
    for system in systemPrintOrder:
        print(' & ', errs[model][system]['e1'], ' & ', errs[model][system]['e2'], end = '')

    print(' \\\\ \\hline')




