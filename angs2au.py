import sys

ANGS_TO_BOHR = 1.8897259885789

if(len(sys.argv) != 3):
    raise Exception("Invalid number of input arguments passed.\n Usage: python angs2au coordinate_file_name option \n, where option can be angs2au or au2angs depending upon whether to convert from Angstrom to atomic units or vice-versa.")

inp = open(str(sys.argv[1]), 'r')
option = str(sys.argv[2])
out = open(str(sys.argv[1])+"_"+option, 'w')
factor = 1.0
if option == 'angs2au':
  factor = ANGS_TO_BOHR

elif option == 'au2angs':
  factor = 1.0/ANGS_TO_BOHR

else:
  raise Exception("Invalid option passed as input. The option can be either angs2au or au2angs")

lines = inp.readlines()
for line in lines:
  words = line.split()
  symbol = words[0]
  print(symbol, file = out, end=" ")
  for word in words[1:4]:
    print(float(word)*factor, file = out, end=" ")

  print(file = out)

inp.close()
out.close()

