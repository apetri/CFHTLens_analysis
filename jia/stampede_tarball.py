import os
#import subprocess
from emcee.utils import MPIPool

inDIR_arr = ['/scratch/02977/jialiu/KSsim_noiseless/', '/scratch/02977/jialiu/KSsim/']

outDIR_arr = ['/scratch/02977/jialiu/ranch_archive/KSsim_noiseless/', '/scratch/02977/jialiu/ranch_archive/KSsim/']

for outDIR in outDIR_arr:
	os.system("lfs setstripe -c 4 %s"%(outDIR))

def create_tarball (FNs):
	inFN, outFN = FNs
	os.system("tar cfv %s.tar %s"%(outFN, inFN))

pool = MPIPool()
for i in (1,):#range(len(inDIR_arr)):
	cosmo_arr = os.listdir(inDIR_arr[i])
	FNs_arr = [['%s%s'%(inDIR_arr[i], cosmo_arr[j]),'%s%s'%(outDIR_arr[i], cosmo_arr[j])] for j in range(45, 59)]
	pool.map(create_tarball, FNs_arr)
	
	