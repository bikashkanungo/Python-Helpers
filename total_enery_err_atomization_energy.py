import csv
import sys

def getAllChemicalElementSymbols():
  pt = open('PeriodicTable.csv')
  ptcsv = csv.reader(pt)
  allRows = list(ptcsv)
  heading = allRows[0]
  elements = allRows[1:]
  pt.close()
  symbolCol = heading.index('Symbol')
  symbols = []
  for e in elements:
    symbols.append(e[symbolCol])

  return symbols

def getAtoms(names, allAtomicSymbols):
    atoms = []
    for name in names:
        if name in allAtomicSymbols:
            atoms.append(name)

    return atoms

def getMols(names, allAtomicSymbols):
    mols = []
    for name in names:
        if name not in allAtomicSymbols:
            mols.append(name)

    return mols


def getMolConstituents(name, allAtomicSymbols):
    valid = True
    msg = ''
    atoms = []
    numbers = []
    if name[0].isalpha == False or name[0].isupper == False:
        valid = False
        msg = 'Invalid first character in chemical forumala'
        return atoms, numbers, valid, msg

    posUpperChars = []
    for i in range(len(name)):
        if name[i].isupper():
            posUpperChars.append(i)

    numUpperChars = len(posUpperChars)
    for i in range(numUpperChars):
        upperCharPos = posUpperChars[i]
        if i == numUpperChars - 1:
            nextUpperCharPos = len(name)

        else:
            nextUpperCharPos = posUpperChars[i+1]

        s = name[upperCharPos:nextUpperCharPos]
        letters = ''
        digits = ''
        for char in s:
            if char.isalpha():
                letters = letters + char

            elif char.isnumeric():
                digits = digits + char

            else:
                valid = False
                msg = 'Non-alphanumeric character found in the chemical formula'
                return atoms, numbers, valid, msg

        if digits == '':
            digits = '1'

        if letters not in allAtomicSymbols:
            valid = False
            msg = 'Invalid atomic symbol ' + letters + ' found in the chemical formula'
            return atoms, numbers, valid, msg

        atoms.append(letters)
        numbers.append(int(digits))

    return atoms, numbers, valid, msg


def getAtomizationEnergies(molEnergies, singleAtomEnergies):
    allAtomicSymbols = getAllChemicalElementSymbols()
    ae = {}
    for molName in molEnergies:
        atoms, numbers, valid, msg = getMolConstituents(molName, allAtomicSymbols)
        if valid == False:
            raise Exception(msg)

        x = 0.0
        for i in range(len(atoms)):
            x += singleAtomEnergies[atoms[i]]*numbers[i]

        x -= molEnergies[molName]
        ae[molName] = x

    return ae

def getRefTotalEnergies(molNames, refAE, singleAtomEnergies):
    allAtomicSymbols = getAllChemicalElementSymbols()
    te = {}
    for molName in molNames:
        atoms, numbers, valid, msg = getMolConstituents(molName, allAtomicSymbols)
        if valid == False:
            raise Exception(msg)

        x = 0.0
        for i in range(len(atoms)):
            x += singleAtomEnergies[atoms[i]]*numbers[i]

        x -= refAE[molName]
        te[molName] = x

    return te


if __name__=="__main__":
    if len(sys.argv) != 4:
        raise Exception('''Invalid command line arguments passed.\n'''\
                '''Usage: python total_energy_err_atomization_energy.py'''\
                '''total_energies_file_name'''\
                '''reference_single_atom_energies_file_name'''\
                '''reference_atomization_energies_file_name''')

    teFilename = str(sys.argv[1])
    refSAEFilename = str(sys.argv[2])
    refAEFilename = str(sys.argv[3])
    teFile = open(teFilename, 'r')
    refSAEFile = open(refSAEFilename, 'r')
    refAEFile = open(refAEFilename, 'r')
    teLines = teFile.readlines()
    refSAELines = refSAEFile.readlines()
    refAELines = refAEFile.readlines()
    te = {}
    refSAE = {}
    refAE = {}

    for line in teLines:
        words = line.split()
        if len(words) != 2:
            raise Exception("Only two columns expected in the total energies file, namely, the chemical formula and the total energy.")

        te[words[0]] = float(words[1])

    for line in refSAELines:
        words = line.split()
        if len(words) != 2:
            raise Exception("Only two columns expected in the single atom energies file, namely, the chemical formula and the total energy of the atom.")

        refSAE[words[0]] = float(words[1])

    for line in refAELines:
        words = line.split()
        if len(words) != 2:
            raise Exception("Only two columns expected in the reference atomization energies file, namely, the chemical formula and the atomization energy.")

        refAE[words[0]] = float(words[1])

    teFile.close()
    refSAEFile.close()
    refAEFile.close()

    molNamesAll = te.keys()
    molAvailable = {}
    allAtomicSymbols = getAllChemicalElementSymbols()
    for molName in molNamesAll:
        atoms, numbers, valid, msg = getMolConstituents(molName, allAtomicSymbols)
        if valid == False:
            raise Exception(msg)

        isMolNameAvailable = True
        for atom in atoms:
            if atom not in refSAE.keys():
                isMolNameAvailable = False
                break

        if isMolNameAvailable:
            molAvailable[molName]=(atoms,numbers)


    refTE = getRefTotalEnergies(molAvailable.keys(), refAE, refSAE)

    me = 0.0
    mae = 0.0
    err = {}
    for molName in refTE:
        val = te[molName] - refTE[molName]
        atomNums = molAvailable[molName][1]
        Na = sum(atomNums)
        val /= Na
        me += val
        mae += abs(val)
        err[molName] = val

    me /= len(refTE)
    mae /= len(refTE)

    outFile = open('err','w')
    for molName in err:
        print(molName, err[molName], file = outFile)

    outFile.close()
    print("ME:", me, "MAE:", mae)

