import sys
filenameRoot = str(sys.argv[1])
outfilename = (sys.argv[2])
nFiles = int(sys.argv[3])
out = open(outfilename, 'w')


for i in range(nFiles):
    f = open(filenameRoot + str(i), 'r')
    lines = f.readlines()
    for line in lines:
        print(line.strip(), file = out)

    f.close()

out.close()


