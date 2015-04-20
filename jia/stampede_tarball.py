import os
import subprocess
from emcee.utils import MPIPool

inDIR_arr = ['/scratch/02977/jialiu/KSsim_noiseless/', '/scratch/02977/jialiu/KSsim/']

outDIR_arr = ['/scratch/02977/jialiu/ranch_archive/KSsim_noiseless/', '/scratch/02977/jialiu/ranch_archive/KSsim/']

for outDIR in outDIR_arr:
	subprocess.check_call('lfs setstripe -c 4 %s' %(outDIR))

pool = MPIPool()

def create_tarball (FNs):
	inFN, outFN = FNs
	subprocess.check_call('tar -cfv %s.tar %s'%(outFN, inFN))

for i in range(len(inDIR_arr)):
	cosmo_arr = os.listdir(inDIR_arr[i])
	FNs_arr = [['%s%s'%(inDIR_arr[i], cosmo),'%s%s'%(outDIR_arr[i], cosmo)] for cosmo in cosmo_arr]
	pool.map(create_tarball, FNs_arr)
	
	