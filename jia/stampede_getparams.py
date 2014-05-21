import numpy as np
from scipy import *
import os
#import WLanalysis
#from emcee.utils import MPIPool
#import scipy.ndimage as snd
#import sys

cat_dir = '/home1/02977/jialiu/cat/'
cosmo_all = os.listdir(cat_dir)
params = zeros(shape=(len(cosmo_all),3))
i = 0
for cosmo in cosmo_all:
	params[i, 0] =  float(cosmo[15:20])
	params[i, 1] = -float(cosmo[-21:-16]) # bcoz 0.000, so can't simply take [30:36]
	params[i, 2] =  float(cosmo[-5:])
	i += 1
savetxt('/home1/02977/jialiu/KSsim/cosmo_params.txt',params)