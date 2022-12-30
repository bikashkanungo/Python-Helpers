import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import grad
from torch import optim
import math
import time
import os.path
import shutil
import FastTensorDataLoader as ftl
import pylibxc
import csv
"""Learning hyperparameters"""
# No. of molecules to process at a time, the current implementation
# assumes it to be 1
bsize = 1 

# Batch size of the number of points within a molecule to process at a time
bsize_within_mol = 100000

#learning rate
lr = 1e-3

#number of epochs
epochs = 5000

# filename into which the model should be written
PATH="./model_"

inputFile_train = 'input_H2_LiH_Ne_Li'

# float byte to use for pytorch and numpy
dtype= torch.float32
nptype = np.float32

# tolerance to be added to the density to avoid zero denominator or other
# underflow issues
RHO_TOL = 1e-8

# Lagrange multiplier for the regularizing term
weight_decay = 0.0

# Slater factor -  a physically useful prefactor for an LDA model
C_LDA = -3.0/4.0*math.pow(3.0/math.pi,1.0/3.0)

# power to which the density (rho) should be raised and applied as a 
# weight for the vxc loss term
pow_rho_weight = 1.0

# X and C functional to be used as a base
# We use the strings that are permissible by libxc package
XBaseFunctional = "gga_x_pbe"
CBaseFunctional = "gga_c_pbe"

# whether modeling a spin dependent (polarized) or spin-independent
# (unpolarized) model for the base functional
SpinBase = "polarized"

def getInputFileData(dataFile):
    data = []
    rows = []
    with open(dataFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for row in csv_reader:
            rows.append(row)

    columnNames = [word.strip() for word in rows[0]]
    numRows = len(rows)
    numCols = len(columnNames)
    for i in range(1,numRows):
        x = {}
        for j in range(numCols):
            key = columnNames[j]
            if key in ['DataFileName', 'Spin', 'Remarks']:
                x[columnNames[j]] = rows[i][j].strip()

            else:
                x[columnNames[j]] = float(rows[i][j].strip())
        
        data.append(x)

    return data



class BaseXCHandler():
    def __init__(self, XFunctionalName, CFunctionalName, spin):
        self.hasX = True
        self.hasC = True
        self.spin = spin 
        self.nspin = 1
        if self.spin not in ["polarized", "unpolarized"]:
            raise Exception('''Invalid spin value = ''' + self.spin + '''
                            encountered. Spin value can be either polarized
                            or unpolarized''')
        if self.spin == 'polarized':
            self.nspin = 2
            

        if XFunctionalName == "" or XFunctionalName is None:
            self.hasX = False
        
        if CFunctionalName == "" or CFunctionalName is None:
            self.hasC = False

        if self.hasX:
            self.X = pylibxc.LibXCFunctional(XFunctionalName, spin)

        if self.hasC:
            self.C = pylibxc.LibXCFunctional(CFunctionalName, spin)

    def getg(self,drho,i,alpha):
        if i == 1:
            return drho[:,self.nspin-1-alpha]
        
        elif (i == 0 and alpha == 0) or (i==2 and alpha == 1):
            return 2.0*drho[:,alpha]

        else:
            return np.zeros((drho.shape[0],3))
    
    
    def getDivergenceg(self,ddrho,i,alpha):
        rhoLap = np.trace(ddrho, axis1=2, axis2=3)
        if i == 1:
            return rhoLap[:,self.nspin-1-alpha]
        
        elif (i == 0 and alpha == 0) or (i==2 and alpha == 1):
            return 2.0*rhoLap[:,alpha]

        else:
            return np.zeros(ddrho.shape[0])

    def getGradSigma(self, drho, ddrho, i):
        if i==0:
            return 2.0*np.einsum('ijk,ik->ij', ddrho[:,0], drho[:,0])

        elif i==1:
            return (np.einsum('ijk,ik->ij', ddrho[:,0], drho[:,1])\
                    +\
                    np.einsum('ijk,ik->ij', ddrho[:,1],drho[:,0]))

        elif i==2:
            return 2.0*np.einsum('ijk,ik->ij', ddrho[:,1],drho[:,1])

        else:
            raise Exception("Invalid sigma index passed")

    def getv(self, vrho, vsigma, v2rhosigma, v2sigma2, drho, ddrho):
        v = vrho
        npoints = vrho.shape[0]
        nsigma = 1
        if self.nspin == 2:
            nsigma = 3
        
        v2sigma2IJMap = {}
        for iSigma in range(nsigma):
            for jSigma in range(iSigma,nsigma):
                offset = iSigma*nsigma - int((iSigma-1)*iSigma/2) + (jSigma-iSigma)
                v2sigma2IJMap[(iSigma,jSigma)] = offset
                v2sigma2IJMap[(jSigma,iSigma)] = offset

        gradSigma = np.zeros((npoints,nsigma,3))
        for iSigma in range(nsigma):
            gradSigma[:,iSigma] = self.getGradSigma(drho, ddrho, iSigma)

        for alpha in range(self.nspin):
            for iSigma in range(nsigma):
                g = self.getg(drho, iSigma, alpha)
                for iSpin in range(self.nspin):
                    v[:,alpha] -= v2rhosigma[:,iSpin*nsigma +iSigma]*\
                            np.einsum('ik,ik->i',drho[:,iSpin],g)

                for jSigma in range(nsigma):
                    index = v2sigma2IJMap[(iSigma,jSigma)] 
                    v[:,alpha] -=  v2sigma2[:,index]*\
                            np.einsum('ik,ik->i',gradSigma[:,jSigma], g)

                v[:,alpha] -= vsigma[:,iSigma]*\
                        self.getDivergenceg(ddrho,iSigma,alpha)

        return v
            

    def getexcAndVxc(self, rho, drho, ddrho):
        isSpinPolarized = (self.spin == "polarized")

        if isinstance(rho, np.ndarray):
            if isSpinPolarized is False:
                rho_np = rho
                sigma_np = np.sum(drho*drho, axis=1)

            else:
                rho_np = np.empty(2*rho.shape[0], dtype=nptype)
                sigma_np = np.empty(3*drho.shape[0], dtype=nptype)
                rho_np[0::2] = rho[:,0]
                rho_np[1::2] = rho[:,1]
                sigma_np[0::3]= np.sum(drho[:,0]*drho[:,0], axis=1)
                sigma_np[1::3]= np.sum(drho[:,0]*drho[:,1], axis=1)
                sigma_np[2::3]= np.sum(drho[:,1]*drho[:,1], axis=1)


        elif isinstance(rho, torch.Tensor):
            if isSpinPolarized is False:
                rho_np = (torch.flatten(rho)).numpy()
                sigma_np = ((drho*drho).sum(dim=1)).numpy()

            else: 
                rhoAlpha_np = (torch.flatten(rho[:,0])).numpy()
                rhoBeta_np = (torch.flatten(rho[:,1])).numpy()
                rho_np = np.empty(2*rhoAlpha_np.shape[0], dtype=nptype)
                sigma_np = np.empty(3*drho.shape[0], dtype=nptype)
                rho_np[0::2] = rhoAlpha_np
                rho_np[1::2] = rhoBeta_np
                sigma_np[0::3]= ((drho[:,0]*drho[:,0]).sum(dim=1)).numpy()
                sigma_np[1::3]= ((drho[:,0]*drho[:,1]).sum(dim=1)).numpy()
                sigma_np[2::3]= ((drho[:,1]*drho[:,1]).sum(dim=1)).numpy()
            
        else:
            raise Exception('''Input variable rho can either be a numpy ndarray
                            or pytorch tensor''')
        inp = {}
        inp["rho"] = rho_np
        inp["sigma"] = sigma_np
        ex = np.zeros_like(rho_np)
        vx = np.zeros_like(rho_np)
        ec = np.zeros_like(rho_np)
        vc = np.zeros_like(rho_np)
        
        if self.hasX:
          XVals = self.X.compute(inp, do_exc=True, do_vxc=True, do_fxc=True)
          ex = (XVals["zk"])[:,0]
          vrho = XVals["vrho"]
          vsigma = XVals["vsigma"]
          v2rhosigma = XVals["v2rhosigma"]
          v2sigma2 = XVals["v2sigma2"]
          vx = self.getv(vrho, vsigma, v2rhosigma, v2sigma2, drho, ddrho)
        
        if self.hasC:
          CVals = self.C.compute(inp, do_exc=True, do_vxc=True, do_fxc=True)
          ec = (CVals["zk"])[:,0]
          vrho = CVals["vrho"]
          vsigma = CVals["vsigma"]
          v2rhosigma = CVals["v2rhosigma"]
          v2sigma2 = CVals["v2sigma2"]
          vc = self.getv(vrho, vsigma, v2rhosigma, v2sigma2, drho, ddrho)
       
        if self.spin == "unpolarized":
            exc = rho_np*(ex+ec)
            vxc = vx[:,0] + vc[:,0]
            excAndVxc = np.column_stack((exc,vxc))

        else:
            rhoAlpha = rho_np[0::2]
            rhoBeta = rho_np[1::2]
            rhoTotal = rhoAlpha + rhoBeta
            exc = rhoTotal*(ex+ec)
            vxcAlpha = vx[:,0] + vc[:,0]
            vxcBeta = vx[:,1] + vc[:,1]
            excAndVxc = np.column_stack((exc,vxcAlpha, vxcBeta))

        excAndVxcTensor = torch.from_numpy(excAndVxc)
        return excAndVxcTensor

""" Dataset
It loads the entire molecule (i.e., all the suuplied grid points for the
molecule """
class DatMolecules(Dataset):
    def __init__(self, moleculesMetaData):
        super(DatMolecules).__init__()
        self.rhoInpDataTensor = []
        self.vxcDataTensor = []
        self.quadWeightDataTensor = []
        self.modDRhoTotalDataTensor = []
        self.uDataTensor = []
        self.wDataTensor = []
        self.rhoTotalLapDataTensor = []
        self.baseexcAndVxc = []
        self.moleculesMetaData = moleculesMetaData
        baseXCHandler = BaseXCHandler(XBaseFunctional, CBaseFunctional, SpinBase)
        if SpinBase == 'polarized':
            nSpin = 2

        elif SpinBase == 'unpolarized':
            nSpin = 1

        else:
            raise Exception("Invalid SpinBase provided. It can be either polarized or unpolarized.")
        
        for index in range(len(moleculesMetaData)):
            filename = moleculesMetaData[index]["DataFileName"]
            Exc = moleculesMetaData[index]["Exc"]
            spin = moleculesMetaData[index]["Spin"]
            ExcWeight = moleculesMetaData[index]["ExcWeight"]
            VxcWeight = moleculesMetaData[index]["VxcWeight"]
            print("Storing molecule data: ", filename, "\n")
            ## The storage format is as follows:
            ## Columns 0,1,2 correspond to the x, y, z coordinates of the points
            ## For a spin unpolarized system: 
            ## (i) column 3 represents the total rho (i.e. rho = 2*rho_alpha = 2*rho_beta)
            ## (ii) column 4 represents the vxc (vxc = vxc_alpha = vxc_beta)
            ## (ii) column 5 represents the quadrature weight for the point
            ## 
            ## For a spin polarized system: 
            ## (i) column 3 and 4 represent rho_alpha and rho_beta, respectively
            ## (ii) column 5 and 6 represent vxc_alpha and vxc_beta, respectively
            ## (ii) column 7 represents the quadrature weight for the point
            tmp = np.loadtxt(fname = filename, dtype = nptype)
            npoints = tmp.shape[0]
            rho = np.empty((npoints,nSpin), dtype=nptype)
            drho = np.empty((npoints,nSpin,3), dtype=nptype)
            ddrho = np.empty((npoints,nSpin,3,3), dtype=nptype)
            vxc = np.empty((npoints,nSpin), dtype=nptype)
            quadWeight = np.empty((npoints), dtype=nptype)
            rhoPreFactor = 1.0
            nSpinAvailable = 1
            if spin == "unpolarized":
                nSpinAvailable = 1
                rhoPreFactor = 0.5

            elif spin == "polarized":
                nSpinAvailable = 2
                rhoPreFactor = 1.0
                
            else:
                raise Exception('''Invalid spin value = ''' + spin + '''
                                encountered. Spin value can be either polarized
                                or unpolarized''')

            # rho, grad \rho, \grad\grad\rho (1+3+9)
            nRhoDataColsPerSpin = 13
            
            # rho, \grad rho, \grad\grad rho for each available spin, 
            # vxc for each avaiable spin, quad weighr
            #
            numColsToRead= nSpinAvailable*nRhoDataColsPerSpin + nSpinAvailable + 1
            A = tmp[:,3:3+numColsToRead]
            
            for iSpin in range(nSpin):
                colOffset = (nSpinAvailable-1)*iSpin*nRhoDataColsPerSpin
                rho[:,iSpin] = rhoPreFactor*A[:,colOffset]
                drho[:,iSpin] = rhoPreFactor*A[:,colOffset+1:colOffset+4]
                ddrho[:,iSpin] = rhoPreFactor*((A[:,colOffset+4:colOffset+13]).reshape(npoints,3,3))
                vxc[:,iSpin] = A[:,nSpinAvailable*nRhoDataColsPerSpin + (nSpinAvailable-1)*iSpin]

            quadWeight = A[:,nSpinAvailable*nRhoDataColsPerSpin + nSpinAvailable]
            

            drhoTotal = drho.sum(axis=1)
            ddrhoTotal = ddrho.sum(axis=1)
            modDRhoTotal = ((drhoTotal**2.0).sum(axis=1))**0.5
            # store \grad \rho_alpha dot \grad rhoTotal
            u = np.zeros((npoints,nSpin))
            for iSpin in range(nSpin):
                u[:,iSpin] = (drho[:,iSpin]*drhoTotal).sum(axis=1)

            # store \grad mod (\grad\rhoTotal)
            v = np.einsum('ijk,ik->ij',ddrhoTotal,drhoTotal)
            w = np.einsum('ij,ij->i',v,drhoTotal)/modDRhoTotal
            rhoTotalLap = np.diagonal(ddrhoTotal, dim1=-2, dim2=-1).sum(axis=-1)

            rhoInp = np.column_stack((rho, modDRhoTotal))
            self.rhoInpDataTensor.append(torch.from_numpy(rhoInp).to(device))
            self.vxcDataTensor.append(torch.from_numpy(vxc).to(device))
            self.quadWeightDataTensor.append(torch.from_numpy(quadWeight).to(device))
            self.modDRhoTotalDataTensor.append(torch.from_numpy(modDRhoTotal).to(device))
            self.uDataTensor.append(torch.from_numpy(u).to(device))
            self.wDataTensor.append(torch.from_numpy(w).to(device))
            self.rhoTotalLapDataTensor.append(torch.from_numpy(rhoTotalLap).to(device))
            excAndVxcTensor = (baseXCHandler.getexcAndVxc(rho, drho, ddrho)).to(device)
            self.baseexcAndVxc.append(excAndVxcTensor)
        
        for index in range(len(moleculesMetaData)):
            print("Exc: ", self.moleculesMetaData[index]["Exc"])
            print("ExcWeight: ", self.moleculesMetaData[index]["ExcWeight"])
            print("VxcWeight: ", self.moleculesMetaData[index]["VxcWeight"])


    def __getitem__(self, index):
        return {'rhoInpData': self.rhoInpDataTensor[index], 
                'vxcData': self.vxcDataTensor[index],
                'quadWeightData': self.quadWeightDataTensor[index],
                'modDRhoTotalData': self.modDRhoTotalDataTensor[index],
                'uData': self.uDataTensor[index],
                'wData': self.wDataTensor[index],
                'rhoTotalLapData': self.rhoTotalLapDataTensor[index],
                'metaData': self.moleculesMetaData[index], 
                'baseexcAndVxc': self.baseexcAndVxc[index]}

    def __len__(self):
        return len(self.moleculesMetaData)

def get_train_sets():
    moleculesMetaData_train = getInputFileData(inputFile_train)
    train = DatMolecules(moleculesMetaData_train)
    return train


"""Create data loaders"""
def get_train_loaders(train_set):
    return DataLoader(dataset = train_set, batch_size = bsize, shuffle =
                       False)

def loss_func(vxc, vxc_ref, quad, weight = None):
    if weight is not None:
        return (((weight*(vxc-vxc_ref))**2)*quad).sum(dim=0)
    else:
        return (((vxc-vxc_ref)**2)*quad).sum(dim=0)

def eval_Exc(exc, quad):
    return (exc*quad).sum()

"""A DNN model with softplus activation function
at each layer except the output layer"""
class DNN(Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 80)
        self.lin2 = nn.Linear(80, 80)
        self.lin3 = nn.Linear(80, 80)
        self.lin4 = nn.Linear(80, 80)
        self.lin5 = nn.Linear(80, 1)

    def forward(self, rho_b):
        inp = torch.empty((rho_b.shape[0],3), device=rho_b.device)
        rhoTotal = (rho_b[:,0]+rho_b[:,1])
        rhoUnif = rhoTotal**(1.0/3.0)
        xi = torch.div(rho_b[:,0]-rho_b[:,1],rhoTotal)
        chi = 0.5*((1.0+xi)**(4.0/3.0)+(1.0-xi)**(4.0/3.0))
        modDRhoTotal = rho_b[:,2]
        Cs = 1.0/(2.0*(3.0*math.pi**2.0)**(1.0/3.0))
        s = Cs*modDRhoTotal/(rhoTotal**(4.0/3.0))
        inp[:,0] = torch.log(rhoUnif)
        inp[:,1]=xi**2.0
        inp[:,2]=torch.log(s)
        #logInp = torch.log(inp)
        #act = nn.Softplus()
        #act = nn.Tanh()
        act = nn.ELU()
        exc_b = act(self.lin1(inp))
        exc_b = act(self.lin2(exc_b))
        exc_b = act(self.lin3(exc_b))
        exc_b = act(self.lin4(exc_b))

        #exc_b = F.softplus(self.lin1(logRhoUnif_b))
        #exc_b = F.softplus(self.lin2(exc_b))
        #exc_b = F.softplus(self.lin3(exc_b))
        #exc_b = F.softplus(self.lin4(exc_b))
        #exc_b = F.softplus(self.lin5(exc_b))
        exc_b = self.lin5(exc_b)
        ## The first past: C_LDA*rho^(4/3.0) is a widely used reference model
        exc_b = C_LDA*(((rhoTotal**(4.0/3.0))*chi).view(-1,1))*exc_b
        return exc_b

    def getVxc(self, rho, modDRhoTotal, u, w, rhoTotalLap):
        exc = self.forward(rho)
        nx = rho.shape[0]
        nspin = 2
        vxc = torch.zeros(nx,nspin).to(device)
        
        exc_ones = torch.ones_like(exc).to(device)
        g = grad(exc, rho, grad_outputs = exc_ones,
                   retain_graph=True, create_graph=True)[0]
        f = g[:,nspin]
        t1 = torch.zeros(nx).to(device)
        for iSpin in range(nspin):
            t1 = t1 + (grad(f, rho, u[:,iSpin], retain_graph=True)[0])[:,iSpin]
        
        t1 = t1 + (grad(f, rho, w, retain_graph=True)[0])[:,nspin]
        t1 = -1.0*t1/modDRhoTotal
        t2 = f*w/(modDRhoTotal**2.0)
        t3 = - f*rhoTotalLap/modDRhoTotal
        for iSpin in range(nspin):
            vxc[:,iSpin] = g[:,iSpin] + t1 + t2 + t3
        
        return vxc


"""Get model and optimizer"""
def get_model():
    model = DNN()
    return model, torch.optim.Adam(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)

"""Calculate loss, backpropogate and learn when training (have optimizer argument)"""
def loss_batch(model, loss_func, molecules_b, opt = None):
    timeLossBatch = time.time()

    loss = torch.tensor(0.0).to(device)
    Exc_mols = []
    Exc_loss = []
    Vxc_loss = []

    timeForward = 0.0
    timeForwardCumulative = 0.0
    batch_loss = 0.0
    rhoInpData_b = molecules_b['rhoInpData']
    modDRhoTotalData_b = molecules_b['modDRhoTotalData']
    uData_b = molecules_b['uData']
    wData_b = molecules_b['wData']
    rhoTotalLapData_b = molecules_b['rhoTotalLapData']
    vxcData_b = molecules_b['vxcData']
    quadWeightData_b = molecules_b['quadWeightData']
    metaData_b = molecules_b['metaData']
    baseexcAndVxc_b = molecules_b['baseexcAndVxc']
    numMols = len(rhoInpData_b)
    for Id in range(numMols):
        filename = metaData_b["DataFileName"][Id]
        Exc_ref = metaData_b["Exc"][Id]
        ExcWeight = metaData_b["ExcWeight"][Id]
        VxcWeight = metaData_b["VxcWeight"][Id]
        rhoInp = rhoInpData_b[Id]
        modDRhoTotal = modDRhoTotalData_b[Id]
        u = uData_b[Id]
        w = wData_b[Id]
        rhoTotalLap = rhoTotalLapData_b[Id]
        vxc = vxcData_b[Id]
        quadWeight = quadWeightData_b[Id]
        baseexcAndVxc = baseexcAndVxc_b[Id]
        mol_data_loader = ftl.FastTensorDataLoader(rhoInp,
                                                   modDRhoTotal,
                                                   u,
                                                   w,
                                                   rhoTotalLap,
                                                   vxc,
                                                   quadWeight.view(-1,1), 
                                                   batch_size = bsize_within_mol,
                                                   shuffle=False,
                                                   requires_grad_flags=[True,
                                                                        False,
                                                                        False,
                                                                        False,
                                                                        False,
                                                                        False,
                                                                        False])
        baseexcAndVxc_data_loader = ftl.FastTensorDataLoader(baseexcAndVxc[:,0].view(-1,1),
                                                             baseexcAndVxc[:,1:3],
                                                             batch_size =
                                                             bsize_within_mol,
                                                             shuffle=False,
                                                             requires_grad_flags=[False,
                                                                                  False])       

        Exc = torch.tensor(0.0)#.requires_grad_(True).to(device)
        startForward = time.time()
        baseexcAndVxc_data_loader_iter = iter(baseexcAndVxc_data_loader) 
        batchId = 0
        vxc_loss_mol = 0.0
        vxcL2Square = 0.0 
        for rhoInp_b, modDRhoTotal_b, u_b, w_b, rhoTotalLap_b, vxc_ref_b, quad_b in mol_data_loader:
            rhoInp_b = rhoInp_b + RHO_TOL
            rho_b_copy = (rhoInp_b[:,0:2]).detach().clone() 
            rho_b_weight = rho_b_copy**pow_rho_weight
            vxcL2Square = vxcL2Square + ((((rho_b_weight*vxc_ref_b)**2)*quad_b).sum(dim=0)).sum()
        
        for rhoInp_b, modDRhoTotal_b, u_b, w_b, rhoTotalLap_b, vxc_ref_b, quad_b in mol_data_loader:
            excbase_b, vxcbase_b = next(baseexcAndVxc_data_loader_iter)
            startTime = time.time()
            rhoInp_b = rhoInp_b + RHO_TOL
            exc_b = model(rhoInp_b) + excbase_b
            Exc = Exc + eval_Exc(exc_b,quad_b)
            vxc_b = model.getVxc(rhoInp_b, modDRhoTotal_b, u_b, w_b, rhoTotalLap_b) + vxcbase_b
            rho_b_copy = (rhoInp_b[:,0:2]).detach().clone() 
            rho_b_weight = rho_b_copy**pow_rho_weight #torch.zeros_like(rho_b_copy)
            vxc_loss_b = loss_func(vxc_b, vxc_ref_b, quad_b, weight = rho_b_weight)
            loss = VxcWeight*vxc_loss_b.sum() #/vxcL2Square
            vxc_loss_mol = vxc_loss_mol + vxc_loss_b.detach().clone()#/vxcL2Square
            if opt is not None:
                loss.backward(retain_graph=False)

            timeForwardCumulative = timeForwardCumulative + (time.time()-startTime)
            batchId = batchId + 1

        loss = ExcWeight*(Exc_ref-Exc)**2
        if opt is not None:
            loss.backward(retain_graph=False)

        Exc_loss_mol = ((Exc_ref-Exc)**2).detach().clone()
        Exc_loss.append(Exc_loss_mol)
        Vxc_loss.append(vxc_loss_mol)
        Exc_mols.append(Exc)
        batch_loss = batch_loss + ExcWeight*Exc_loss_mol + VxcWeight*vxc_loss_mol.sum()

        timeForward = time.time() - startForward

    if opt is not None:
        opt.step()
        opt.zero_grad()

    print("Time loss batch: ", time.time() - timeLossBatch)
    return batch_loss/numMols, Exc_mols, Exc_loss, Vxc_loss, numMols


"""Training and Validate, save model when validation loss
improve after each epoch (1 epoch: 1 round of all the data)"""
def fit(epochs, model, loss_func, opt, train_l, valid_l):
    best_val_loss = 1e20
    outfile = open('out_'+PATH[2:],'w')
    print("Num valid mols: ", len(valid_l), file=outfile)
    for epoch in range(epochs):
        model.train()
        startTime = time.time()
        train_losses, Exc_mols_t, Exc_loss_t, Vxc_loss_t, nums_t = zip(*[loss_batch(model, loss_func,
                                                                                    molecules_b,
                                                                                    opt) for
                                                                         molecules_b in train_l]
                                                                      )
        train_loss = sum(train_losses[i].item()*nums_t[i] for i in
                         range(len(nums_t)))/sum(nums_t)
        #np.sum(np.multiply(train_losses.detach().cpu().numpy(), nums_t)) / np.sum(nums_t) #np.mean(train_losses)
        endTime = time.time()
        print(epoch, "\ntraining loss: ", train_loss, "Wall time: ",
              endTime-startTime, "sec", file = outfile)
        Exc_mols = []
        Exc_loss = []
        Vxc_loss = []
        for iBatch in range(len(nums_t)):
            numMols = nums_t[iBatch]
            Exc_mols = Exc_mols + [x.item() for x in Exc_mols_t[iBatch]]
            Exc_loss = Exc_loss + [x.item() for x in Exc_loss_t[iBatch]]
            Vxc_loss = Vxc_loss + [(x[0].item(), x[1].item()) for x in Vxc_loss_t[iBatch]]

        print("Exc training: ", Exc_mols, file = outfile)
        print("Exc loss: ", Exc_loss, file = outfile)
        print("Vxc loss:", Vxc_loss, file = outfile)
        outfile.flush()
        model.eval()

        ## Validation part
        startTime = time.time()
        valid_losses, Exc_mols_v, Exc_loss_v, Vxc_loss_v, nums_v = zip(*[loss_batch(model, loss_func,
                                                                                    molecules_b,
                                                                                    opt= None) for
                                                                         molecules_b in valid_l]
                                                                      )
        valid_loss = sum(valid_losses[i].item()*nums_v[i] for i in
                         range(len(nums_v)))/sum(nums_v)
        #valid_loss = np.sum(np.multiply(valid_losses, nums_v)) / np.sum(nums_v)
        endTime = time.time()
        print(epoch, "\nvalidation loss: ", valid_loss, "Wall time: ",
              endTime-startTime, "sec", file = outfile)
        Exc_mols = []
        Exc_loss = []
        Vxc_loss = []
        for iBatch in range(len(nums_v)):
            numMols = nums_v[iBatch]
            Exc_mols = Exc_mols + [x.item() for x in Exc_mols_v[iBatch]]
            Exc_loss = Exc_loss + [x.item() for x in Exc_loss_v[iBatch]]
            Vxc_loss = Vxc_loss + [(x[0].item(), x[1].item()) for x in
                                   Vxc_loss_v[iBatch]]
        
        print("Exc validation: ", Exc_mols, file = outfile)
        print("Exc loss: ", Exc_loss, file = outfile)
        print("Vxc loss:", Vxc_loss, file = outfile)
        outfile.flush()

        #Storing the model
        if valid_loss < best_val_loss:
            state = {
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
            }
            torch.save(state, PATH)
            best_val_loss = valid_loss
            print("loss: ", valid_loss)
            
        if epoch%1000 == 0:
          shutil.copy2(PATH, PATH+"."+str(epoch))

    outfile.close()

"""Main code"""
def run():
    torch.set_num_threads(num_threads)

    # Load training and validation data
    train_s = get_train_sets()
    train_l = get_train_loaders(train_s)
    valid_l= train_l
    model, opt = get_model()
    lf = loss_func

    #Load saved model to continue training and start testing
    if os.path.exists(PATH):
        print("Loading model from: ", PATH)
        checkpoint = torch.load(PATH, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        opt.load_state_dict(checkpoint['optimizer'])

    else:
        model.to(device)

    """Start training"""
    fit(epochs, model, lf, opt, train_l, valid_l)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_threads = 18
print("Device: ", device)
if __name__ == "__main__":
    run()
