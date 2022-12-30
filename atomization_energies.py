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


if __name__=="__main__":
    if len(sys.argv) != 4:
        raise Exception("Invalid command line arguments passed.\n Usage: python atomization_energies.py total_energies_file_name single_atom_energies_file_name reference_atomization_energies_file_name")

    teFilename = str(sys.argv[1])
    saeFilename = str(sys.argv[2])
    refAEFilename = str(sys.argv[3])
    teFile = open(teFilename, 'r')
    saeFile = open(saeFilename, 'r')
    refAEFile = open(refAEFilename, 'r')
    teLines = teFile.readlines()
    saeLines = saeFile.readlines()
    refAELines = refAEFile.readlines()
    te = {}
    sae = {}
    refAE = {}

    for line in teLines:
        words = line.split()
        if len(words) != 2:
            raise Exception("Only two columns expected in the total energies file, namely, the chemical formula and the total energy.")

        te[words[0]] = float(words[1])

    for line in saeLines:
        words = line.split()
        if len(words) != 2:
            raise Exception("Only two columns expected in the single atom energies file, namely, the chemical formula and the total energy of the atom.")

        sae[words[0]] = float(words[1])

    for line in refAELines:
        words = line.split()
        if len(words) != 2:
            raise Exception("Only two columns expected in the reference atomization energies file, namely, the chemical formula and the atomization energy.")

        refAE[words[0]] = float(words[1])

    teFile.close()
    saeFile.close()
    refAEFile.close()

    ae = getAtomizationEnergies(te, sae)
    print(ae)

    me = 0.0
    mae = 0.0
    err = {}
    for molName in ae:
        val = ae[molName] - refAE[molName]
        me += val
        mae += abs(val)
        err[molName] = val

    me /= len(ae)
    mae /= len(ae)

    outFile = open('err','w')
    for molName in err:
        print(molName, err[molName], file = outFile)

    outFile.close()
    print("ME:", me, "MAE:", mae)

