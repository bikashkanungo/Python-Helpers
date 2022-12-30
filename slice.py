import sys

if len(sys.argv) != 3:
    raise Exception("Incorrect usage. Correct usage: python slie.py function-file params-file")

fFilename = str(sys.argv[1])
pFilename = str(sys.argv[2])
f = open(fFilename, 'r')
p = open(pFilename, 'r')

lines = p.readlines()
ranges = []
count = 0
oFilename = fFilename
for line in lines:
    if not line:
        raise Exception("Empty line found in filei: " + pFilename)

    words = line.split()
    if len(words) !=2:
        raise Exception("Only two values per row (hi and low) allowed in the params file")

    if count >=3:
            raise Exception("Only three rows, one for each direction, allowed in the params file")

    x = [0.,0.]
    x[0] = float(words[0])
    x[1] = float(words[1])
    ranges.append(x)
    oFilename += "_" + words[0] + "_" + words[1]
    count += 1

print(ranges)

lines = f.readlines()
data =[]
for line in lines:
    if not line:
        raise Exception("Empty line found in filei: " + fFilename)

    words = line.split()
    rowData = []
    for w in words:
        rowData.append(float(w))

    data.append(rowData)


N = len(data)
o = open(oFilename,'w')
for i in range(N):
    rowData = data[i]
    keep = True
    for j in range(3):
        if data[i][j] < ranges[j][0] or data[i][j] > ranges[j][1]:
            keep = False
            break

    if keep:
      M = len(rowData)
      for j in range(M):
        print(rowData[j], end = " ", file = o)

      print(file = o)


p.close()
f.close()
o.close()
