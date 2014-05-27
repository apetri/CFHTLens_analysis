#!~/anaconda/bin/python
# Jia Liu 2014/5/21
# What the code does: create mass maps for 100 cosmologies, for the CFHT emulator project
# Cluster: XSEDE Stampede

import WLanalysis
from emcee.utils import MPIPool
import os
import numpy as np
from scipy import *
import scipy.ndimage as snd
import sys
from multiprocessing import Pool
from scoop import futures

########## define constants ############
print 'start'
sim_dir = '/home1/02977/jialiu/cat/'
KS_dir = '/scratch/02977/jialiu/KSsim/'
cosmo_arr = os.listdir(sim_dir)

i_arr=arange(1,14)
R_arr=arange(1,1001)

def testMPIPool(iiRcosmo):
	i, R, cosmo = iiRcosmo
	savetxt('/home1/02977/jialiu/test%i'%(i),zeros(5))

R=1
cosmo=cosmo_arr[1]
iRcosmo = [[i, R, cosmo] for i in i_arr]
#pool = MPIPool()
#pool = Pool(len(iRcosmo))
#pool.map(testMPIPool, iRcosmo)
#pool.close()
futures.map(testMPIPool, iRcosmo)

print 'KSKSKS-DONE-DONE-DONE'
#savetxt('/home1/02977/jialiu/done_KS.ls',zeros(5))

##### power spectrum, peaks ############

