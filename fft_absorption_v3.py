##
## fft1d.m
##
## Simple GNU Octave script for doing a 1D forward Fourier Transform.
## Tested for Octave 3.2.4 on GNU/Linux, no guarantees will work for
## matlab...
##
## Example usage (note "-q" for silent so you can redirect output):
##
## octave -q fft1d.m ft.dat > fw.dat
##
## where the input (ft.dat) has data: t, f(t) and the output (fw.dat)
## will have w, Re[f(w)], Im[f(w)], Abs[f(w)].
##
## You will have to manually change the flag, damping, padding in this
## file.
##
## XXXTODO command line switches for opts
##
## Kenneth Lopata
##
## Last modified: May 22, 2013
##


##
## Switches and options for preprocessing
##
import numpy as np
import sys

speedLight = 137.03604 # in atomic units


def parseInp(inp):
    keysStringValued = ['DipoleX', 'DipoleY', 'DipoleZ', 'OutFile', 'FieldType']
    keysFloatValued = ['k', 't0', 'w', 'tau']
    keysBoolValued = ['DampingFlag', 'PadZeroFlag']
    keysIntValued = ['NumZerosPad']
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

preprocess_expconst = inpdict['tau']  #if damping: damp by exp(-t/tau) before FFT; same time units as input
preprocess_npad = inpdict['NumZerosPad']    #if padding: add this many points to time signal before FFT


##
## Read in raw data t, f(t) from file (1st command line arg)
##
dipoleIdToKeyMap = {0: 'DipoleX', 1: 'DipoleY', 2: 'DipoleZ'}
data = []
for i in range(3):
    dipoleKey = dipoleIdToKeyMap[i]
    filename = inpdict[dipoleKey]
    X = np.loadtxt(fname=filename, dtype=np.float64)
    data.append(X)

minRows= min([X.shape[0] for X in data])
t = data[0][0:minRows,0]

k = inpdict['k']
t0 = inpdict['t0']
omega = inpdict['w']

if preprocess_pad:
    zeros = np.linspace(0.0, 0.0, preprocess_npad)
else:
    zeros = np.linspace(0.0, 0.0, 0)

if preprocess_damp:
    damp = np.exp(-(t-t[0])/preprocess_expconst)
else:
    damp = 1.0

n = minRows + preprocess_npad     #this includes padding
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

if inpdict['FieldType']== 'gaussian':
    g  = k*np.exp(-(t-t0)**2.0/(2.0*omega**2.0)) 
    g = np.append(g,zeros)
    gw = np.fft.fft(g)
    gw_pos = gw[0:m]              #FFT values of positive frequencies (first half of output array)

elif inpdict['FieldType'] == 'delta':
    gw_pow = np.array([1.0/k]*m)

else:
    raise Exception("Invalid field type: " + inpdict['FieldType'] + " passed\
                     input file")

dataFFT = np.zeros((m,9), dtype=np.complex128)
for i in range(3):
    for j in range(3):
        f = data[i][0:minRows,1+j]
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
        dataFFT[:,i*3+j] = fw_pos/gw_pos


trace = dataFFT[:,0].imag# + dataFFT[:,4].imag + dataFFT[:,8].imag


factor = 4.0*np.pi/(3.0*speedLight)
outf = open(inpdict['OutFile'], 'w')
for i in range(m):
    print(w[i], file = outf, end = " ")
    for j in range(3):
        for k in range(3):
            print(dataFFT[i,j*3+k].real, dataFFT[i,j*3+k].imag, file = outf,
                  end = " ")

    print(factor*w[i]*trace[i], file = outf)

outf.close()
