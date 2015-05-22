#!~/anaconda/bin/python
# Jia Liu 2014/5/21
# What the code does: create mass maps for 100 cosmologies, for the CFHT emulator project
# Cluster: XSEDE Stampede

from emcee.utils import MPIPool
import numpy as np
from scipy import *
from multiprocessing import Pool
#from scoop import futures

print 'start'

i_arr=arange(1,14)

def testMPIPool(i):
	savetxt('/home1/02977/jialiu/test%i'%(i),zeros(5))

pool = MPIPool()
#pool = Pool(len(i_arr))
pool.map(testMPIPool, i_arr)
#pool.close()
#futures.map(testMPIPool, i_arr)

print 'DONE-DONE-DONE'

