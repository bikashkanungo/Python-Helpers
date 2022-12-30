import numpy as np
import sys

if (len(sys.argv) ==6 or len(sys.argv)==8) is False:
    raise Exception('''Incorrect command line arguments passed.\n'''\
                    '''Usage: python CombineXRhoVQuad <numspins>'''\
                    '''<rho-data-file-name-space separated list> '''\
                    '''<vxc-data-file-names-space separated list> '''\
                    '''<quad-data-file-name>'''\
                    '''<output-file-name>''')

nspin = int(sys.argv[1])
if (nspin==1 or nspin==2) is False:
    raise Exception("Number of spins can be either 1 or 2")

if nspin==1 and len(sys.argv)==8:
    raise Exception('''Invalid number of commandline arguments passed for a'''\
                    ''' spin unpolarized case''')

if nspin==2 and len(sys.argv)==6:
    raise Exception('''Invalid number of commandline arguments passed for a'''\
                    ''' spin polarized case''')

rhoFilenames = []
vxcFilenames = []
if nspin == 1:
    rhoFilenames.append(str(sys.argv[2]))
    vxcFilenames.append(str(sys.argv[3]))
    qFilename = str(sys.argv[4])
    outFilename =str( sys.argv[5])

else:
    rhoFilenames.append(str(sys.argv[2]))
    rhoFilenames.append(str(sys.argv[3]))
    vxcFilenames.append(str(sys.argv[4]))
    vxcFilenames.append(str(sys.argv[5]))
    qFilename = str(sys.argv[6])
    outFilename = str(sys.argv[7])

q = np.loadtxt(qFilename, dtype=np.float64)
X = q[:,0:3]
qw = q[:,4]
L2X = np.linalg.norm(X,2)
if nspin==1:
    rhoTmp = np.loadtxt(rhoFilenames[0], dtype=np.float64)
    vxcTmp = np.loadtxt(vxcFilenames[0], dtype=np.float64)
    diffL2X = np.linalg.norm(rhoTmp[:,0:3]-X,2)
    if diffL2X/L2X > 1e-12:
        raise Exception('''Mismatch of coordinates in the rho and quad files''')
    
    diffL2X = np.linalg.norm(vxcTmp[:,0:3]-X,2)
    if diffL2X/L2X > 1e-12:
        raise Exception('''Mismatch of coordinates in the vxc and quad files''')
    
    rho = rhoTmp[:,3:]
    vxc = vxcTmp[:,4]

else:
    rhoA = np.loadtxt(rhoFilenames[0], dtype=np.float64)
    rhoB = np.loadtxt(rhoFilenames[1], dtype=np.float64)
    vxcA = np.loadtxt(vxcFilenames[0], dtype=np.float64)
    vxcB = np.loadtxt(vxcFilenames[1], dtype=np.float64)
    diffL2X = np.linalg.norm(rhoA[:,0:3]-X,2)
    if diffL2X/L2X > 1e-12:
        raise Exception('''Mismatch of coordinates in the rho-alpha and quad files''')
    
    diffL2X = np.linalg.norm(rhoB[:,0:3]-X,2)
    if diffL2X/L2X > 1e-12:
        raise Exception('''Mismatch of coordinates in the rho-beta and quad files''')
    
    diffL2X = np.linalg.norm(vxcA[:,0:3]-X,2)
    if diffL2X/L2X > 1e-12:
        raise Exception('''Mismatch of coordinates in the vxc-alpha and quad files''')
    
    diffL2X = np.linalg.norm(vxcB[:,0:3]-X,2)
    if diffL2X/L2X > 1e-12:
        raise Exception('''Mismatch of coordinates in the vxc-beta and quad files''')
    
    rho = np.column_stack((rhoA[:,3:],rhoB[:,3:]))
    vxc = np.column_stack((vxcA[:,4],vxcB[:,4]))

data = np.column_stack((X, rho, vxc, qw))
np.savetxt(outFilename,data)
