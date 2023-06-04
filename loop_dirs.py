import sys
import os
from os.path import exists
import filecmp

dirfile = str(sys.argv[1])
numProcs = str(sys.argv[2])


rootdir = os.getcwd()
modelDirName = "/scratch/vikramg_root/vikramg/bikash/XC_ML/scf_gga_v2/models/"
#str(input("Enter model directory path: "))
modelFilename = "traced_lda_5x80_ELU_gpu_PBE_base_plus_C_LDA_rho4by3_chi_Exc_log_unifRho_xi2_modGradRho_weight_1.0e6_vxc_weight_1.0_weight_rho_decay_0.0_H2_eq_LiH_Ne_Li_N_C_spin_lr_1e-3_bse_10000_.ptc"
#str(input("Enter model name: "))
outFilename = "out_NN_GGA_PBEBase_H2_eq_LiH_Ne_Li_N_C_xi2_no_aux_bse_qz_tz_5x80_epoch_10000_lr_1e-3"
#str(input("Enter dftfe output file name: "))
modelfile = modelDirName + "/" + modelFilename

fdir = open(dirfile,"r")
lines = fdir.readlines()
for line in lines:
    sysName = line.split()[0]
    os.chdir(rootdir + "/" + sysName)
    print("Entered directory: " + os.getcwd())
    os.system("cp " + modelfile +" .")
    os.system("cp " + modelfile + " traced_spin.ptc")
    s = "Using NN-GGA model: " + modelfile
    s_with_quotes = '"' + s + '"'
    os.system("echo " + s_with_quotes + " >  model_name")
    os.system("mpirun -n " + numProcs + " ./dftfe_non_minimal parameters.prm &> out_")
    #os.system('echo "test" &> out_')
    os.system("cat model_name out_ > " + outFilename)
    os.system("rm model_name out_")

fdir.close()


