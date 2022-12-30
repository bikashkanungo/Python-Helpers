##
## Switches and options for preprocessing
##
import numpy as np
import sys

speedLight = 137.03604 # in atomic units


def parseInp(inp):
    keysStringValued = ['Dipole','OutFile', 'DampingType]
    keysFloatValued = ['t0', 't1']
    keysBoolValued = ['DampingFlag', 'PadZeroFlag']
    keysIntValued = ['NumZerosPad', 'PolarIndex']
    keysAll = keysStringValued + keysFloatValued + keysBoolValued + keysIntValued
    inpF = open(inp, 'r')
    lines = inpF.readlines()
    inpdict = {}
    for line in lines:
        line = line.strip()
        if line: 
            #skip comments
            if line[0] == '#':
                continue;

            words = line.split()
            try:
                Id = words.index('=')
            except:
                raise Exception("Invalid input line in input file. The invalid line is: "+ line)

            key = str(words[Id-1])

            if key in keysStringValued:
                val = str(words[Id+1])

            elif key in keysFloatValued:
                val = float(words[Id+1])

            elif key in keysBoolValued:
                if (words[Id+1].lower())[0] == 't':
                    val = True
                
                elif (words[Id+1].lower())[0] == 'f':
                    val = False

                else:
                    raise Exception("Invalid boolean value found in input file. The ivalid line is: "+line)

            elif key in keysIntValued:
                val = int(words[Id+1])

            else:
                raise Exception("Invalid keyword: " + key + " found in input file")

            inpdict[key] = val


    keysFound = list(inpdict.keys())

    for key in keysAll:
        if key not in keysFound:
            raise Exception("Key: " + key + " not found in the input file")

    return inpdict


    

inputFile = str(sys.argv[1])
inpdict= parseInp(inputFile)

print(inpdict)

preprocess_zero = True
preprocess_pad  = inpdict['PadZeroFlag']
preprocess_damp = inpdict['DampingFlag']
preprocess_dampType = inpdict['DampingType']

preprocess_expconst = inpdict['tau']  #if damping: damp by exp(-t/tau) before FFT; same time units as input
preprocess_npad = inpdict['NumZerosPad']    #if padding: add this many points to time signal before FFT
polariId = inpdict['PolarIndex'] ## index of the polarization of the field x=0, y=1, z=2)

##
## Read in raw data t, f(t) from file (1st command line arg)
##
filename = inpdict['Dipole']
dipole = np.loadtxt(fname=filename, dtype=np.float64)

nRows= dipole.shape[0]
t = dipole[0:nRows,0]

t0 = inpdict['t0']
t1 = inpdict['t1']

if preprocess_pad:
    zeros = np.linspace(0.0, 0.0, preprocess_npad)
else:
    zeros = np.linspace(0.0, 0.0, 0)

if preprocess_damp:
    if preprocess_dampType == 'exponential':
        damp = np.exp(-(t-t[0])/preprocess_expconst)

    elif preprocess_dampType == 'gaussian':
        damp = np.exp(-((t-t[0])**2.0)/preprocess_expconst)

    else:
        raise Exception('''Invalid DampingType provided. Valid types: exponential
                        and gaussian'''

else:
    damp = 1.0

n = nRows + preprocess_npad     #this includes padding
dt = t[1] - t[0]   #assumes constant time step; XXX no safety checks
period = (n-1)*dt - t[0]
dw = 2.0*np.pi/period
if n%2 == 0:
    m = int(n/2)
else:
    m = int((n-1)/2)

print("m", m)
wmin = 0.0
wmax = m*dw

w = np.linspace(wmin, wmax, m)  #positive frequency list

dataFFT = np.zeros(m, dtype=np.complex128)
f = dipole[0:nRows,1+polarId]
##
## Optional preprocessing of time signal
##
## (zero at t=0)
if preprocess_zero:
    f0 = f[0]
    f = f - f0

f = f*damp
fpad = np.append(f, zeros)

##
## Do FFT, compute frequencies, and print to stdout: w, Re(fw), Im(fw),
## Abs(fw). Note we only print the positive frequencies (first half of
## the data)--this is fine since time signal is pure real so the
## negative frequencies are the Hermitian conjugate of the positive
## frequencies.
##
fw = np.fft.fft(fpad)
fw_pos = fw[0:m]              #FFT values of positive frequencies (first half of output array)
dataFFT = fw_pos

outf = open(inpdict['OutFile'], 'w')
for i in range(m):
    val = (-(w[i]**2.0)/(t1-t0))*dataFFT[i]
    strength = abs(val)**2.0
    print(w[i], dataFFT[i].real/(t1-t0), dataFFT.imag/(t1-t0), strength, file = outf)

outf.close()
