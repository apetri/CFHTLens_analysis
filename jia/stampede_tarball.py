import os
#import subprocess
from emcee.utils import MPIPool

#inDIR_arr = ['/scratch/02977/jialiu/KSsim_noiseless/', '/scratch/02977/jialiu/KSsim/']

#outDIR_arr = ['/scratch/02977/jialiu/ranch_archive/KSsim_noiseless/', '/scratch/02977/jialiu/ranch_archive/KSsim/']

inDIR = '/scratch/02977/jialiu/CMB_hopper/CMB_batch_storage/'
outDIR = '/scratch/02977/jialiu/ranch_archive/CMB_batch_storage/'

os.system("lfs setstripe -c 4 %s"%(outDIR))

def create_tarball (FNs):
    inFN, outFN = FNs
    os.system("tar cfv %s.tar %s"%(outFN, inFN))

pool = MPIPool()
#for i in (1,):#range(len(inDIR_arr)):
cosmo_arr = os.listdir(inDIR)
FNs_arr = [['%s%s'%(inDIR, cosmo_arr[j]),'%s%s'%(outDIR, cosmo_arr[j])] for j in range(len(cosmo_arr))]
pool.map(create_tarball, FNs_arr)
    
    