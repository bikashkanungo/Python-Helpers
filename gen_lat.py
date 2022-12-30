#!/usr/bin/env python3

from numpy import *

L = 5.869   # lattice constant (Ang.)
N = [2,2,1] # nx, ny, nz for bulk unit cell
cap = True  # capping layers.

# Fill-in In for top unit cell
nTop = ['In']*2
xTop = [
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0] # In
        ]

# 2 x 2 P locations
# These are specified in lattice positions
# with reference to a square 2x2 unit of In

# refs:
# PRL 82(9): 1879, 1999.
# https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.90.126101
zTerm1 = 0.2 # height of term above/below layer of In atoms
zTerm2 = 0.27 # height of term above/below layer of In atoms
yHyd = 0.1 # Hydrogen offset from its P-atom
zHyd = 0.15
a = 0.2
nTerm = ['P', 'P', 'H']*2
xTerm = [
        [0.5,0.5-a,zTerm2], [0.5,0.5+a,zTerm1], [0.5,0.5+a+yHyd,zTerm1+zHyd],
        [1.5,0.5+a,zTerm2], [1.5,0.5-a,zTerm1], [1.5,0.5-a-yHyd,zTerm1+zHyd],
        ] # (2x2)-2D-2H

# Transformation matrix:
# x_{2x2 square lattice} -> dot(x, Trs) = dot(Trs^T, x^T)^T
upT = array([[-1,1,0],[1,1,0],[0,0,1]], float)
# rotate 180 about x-axis
dnT = array([[-1,-1,0],[1,-1,0],[0,0,-1]], float)
upT[:2] *= 0.5 # scale [1,1] to [0.5,0.5] (to align with central fcc In)
dnT[:2] *= 0.5 # scale [1,1] to [0.5,0.5] (to align with central fcc In)

def test2x2(cap, nx, ny):
    """ Returns True when 2x2 reconstruction
         should be overlayed on surface.
         nx == 0 and ny == 0 or nx == 1 and ny == 1
         (circles in the diagram below)
     """
    return cap and (nx%2 == ny%2)

## Actual In locations on a 2x2 lattice
# 7
# 6     .       .
# 5
# 4 .       o
# 3
# 2     .       .
# 1
# 0 o       .
#   0 1 2 3 4 5 6 7


nUnit = ['In']*4 + ['P']*4
xUnit = [
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], # In
        [0.5, 0.0, 0.5], [0.0, 0.5, 0.5], # In
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], # P
        [0.75, 0.25, 0.75], [0.25, 0.75, 0.75] # P
        ]
# xUnit = xIn + xP

class Cut:
    tol = -1e-2
    scale = N[2]

    # List of cutting planes specified by homogeneous coordinates
    #  (unnormalized direction, dot prod. with any point on that plane)
    #
    # inversion symmetry is assumed
    cutplanes = [[1, 1, 0,  2*scale+tol],
            [1, 0, 1,  2*scale+tol],
            [0, 1, 1,  2*scale+tol],
            [1,-1, 0,  2*scale+tol],
            [1, 0,-1,  2*scale+tol],
            [0, 1,-1,  2*scale+tol]]

    class Mol:
        def __init__(self, names, crds):
            self.names = names
        self.crds = array(crds)

    def __str__(self):
        res = "MOL"
        chain = ' '
        lines = []
        for i in range(len(self.names)):
            lines.append(
                    "ATOM  %5d %4s %3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f    %2s"%(
                        i+1, self.names[i], res, chain, 1, self.crds[i][0],
                        self.crds[i][1], self.crds[i][2], 1.0, 0.0, self.names[i][:2]
                        ))
            return '\n'.join(lines)

    def cull(self, plane):
        """ remove atoms outside of a given cutting plane (direction, dot prod.) """
        crds = self.crds.tolist()

        for i in reversed(range(len(self.names))):
            d = crds[i][0]*plane[0] + crds[i][1]*plane[1] + crds[i][2]*plane[2]
            if abs(d) > plane[3]:
                del crds[i]
                del self.names[i]
        self.crds = array(crds)

    def wrap(self, L): # wrap nearest 0
        self.crds[:,2] -= L[2]*floor(self.crds[:,2]/L[2] + 0.5)
        self.crds[:,1] -= L[1]*floor(self.crds[:,1]/L[1] + 0.5)
        self.crds[:,0] -= L[0]*floor(self.crds[:,0]/L[0] + 0.5)

def gen_supercell(n, cap=False):
    if cap:
        assert n[0]%2 == 0 and n[1]%2 == 0, "xy plane must be divisible by 2x2"
    names = []
    crds = []
    tr = zeros(3, float)
    for nx in range(n[0]):
        tr[0] = nx
        for ny in range(n[1]):
            tr[1] = ny
            if test2x2(cap, nx, ny):
                tr[2] = 0.0
                names += nTerm
                crds += (dot(array(xTerm),dnT)+tr).tolist()
            for nz in range(n[2]):
                tr[2] = nz
                names += nUnit
                crds += (array(xUnit)+tr).tolist()
            if cap:
                tr[2] = n[2]
                names += nTop
                crds += (array(xTop)+tr).tolist()
            if test2x2(cap, nx, ny):
                tr[2] = n[2]
                names += nTerm
                crds += (dot(array(xTerm),upT)+tr).tolist()

    return Mol(names, crds)

mol = gen_supercell( N, cap )
mol.crds -= 0.5*array(N, float)
mol.wrap([N[0], N[1], N[2]+10])

#for plane in cutplanes:
#    mol.cull(plane)

mol.crds *= L
print(mol)

