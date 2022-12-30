symbols = ["H",
           "He",
           "Li",
           "Be",
           "B",
           "C",
           "N",
           "O",
           "F",
           "Ne",
           "Na",
           "Mg",
           "Al",
           "Si",
           "P",
           "S",
           "Cl",
           "Ar",
           "K",
           "Ca",
           "Sc",
           "Ti",
           "V",
           "Cr",
           "Mn",
           "Fe",
           "Co",
           "Ni",
           "Cu",
           "Zn",
           "Ga",
           "Ge",
           "As",
           "Se",
           "Br",
           "Kr",
           "Rb",
           "Sr",
           "Y",
           "Zr",
           "Nb",
           "Mo",
           "Tc",
           "Ru",
           "Rh",
           "Pd",
           "Ag",
           "Cd",
           "In",
           "Sn",
           "Sb",
"Te",
"I",
"Xe",
"Cs",
"Ba",
"La",
"Ce",
"Pr",
"Nd",
"Pm",
"Sm",
"Eu",
"Gd",
"Tb",
"Dy",
"Ho",
"Er",
"Tm",
"Yb",
"Lu",
"Hf",
"Ta",
"W",
"Re",
"Os",
"Ir",
"Pt",
"Au",
"Hg",
"Tl",
"Pb",
"Bi",
"Po",
"At",
"Rn",
"Fr",
"Ra",
"Ac",
"Th",
"Pa",
"U",
"Np",
"Pu",
"Am",
"Cm",
"Bk",
"Cf",
"Es",
"Fm",
"Md",
"No",
"Lr",
"Rf",
"Db",
"Sg",
"Bh",
"Hs",
"Mt",
"Ds",
"Rg",
"Cp",
"Uut",
"Uuq",
"Uup",
"Uuh",
"Uus",
"Uuo"]


f1 = open('PeriodicTable1','w')
f2 = open('PeriodicTable2','w')
for i in range(len(symbols)):
    print('{',i+1.0,',',symbols[i],'},', file=f1)
    print('{',symbols[i],',',i+1.0,'},', file=f2)