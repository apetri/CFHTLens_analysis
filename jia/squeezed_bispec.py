### Jia Liu 2016/12/13
### works on Stampede
### compute the bispectrum in the squeezed limit kappa^2 x kappa
### Gaussian smoothing 1,2,5 arcmin

import WLanalysis
import glob, os, sys
import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack
from emcee.utils import MPIPool

from scipy import interpolate
import random

main_dir = '/work/02977/jialiu/squeeze/'
file_dir = '/work/02977/jialiu/CMBL_maps_46cosmo/'

all_points = genfromtxt(file_dir+'model_point.txt')
cosmo_arr = array(['Om%.3f_Ol%.3f_w-1.000_si%.3f'%(cosmo[0],1-cosmo[0], cosmo[1]) for cosmo in all_points])

fidu_cosmo = 'Om0.296_Ol0.704_w-1.000_si0.786'
kmapGen = lambda cosmo, r: WLanalysis.readFits(file_dir+'%s/WLconv_z1100.00_%04dr.fits'%(cosmo, r))
PPA=2048.0/3.5/60.0

def BispecGen (r, cosmo = fidu_cosmo, R_arr = [1.0, 2.0, 5.0]):
    '''Generate the squeezed bispectrum, with smoothing scales R_arr
    '''
    print cosmo, r
    ikmap = kmapGen(cosmo, r)
    ikmap -= mean(ikmap) ## set mean to 0
    bs_arr = zeros(shape=(3,50))
    
    for i in range(len(R_arr)):
        ikmap_smooth = WLanalysis.smooth(ikmap,  R_arr[i]*PPA)
        bs_arr[i] = CrossCorrelate(ikmap**2, ikmap, sizedeg = 12.25, PPA=PPA)[1]
    return bs_arr


pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)
a=pool.map(BispecGen, arange(1,1025))
save(main_dir+'%s_BS.npy'%(fidu_cosmo), a)
pool.close()
print '---DONE---DONE---'


#### on stampede
#### idev -m 60
#### ibrun python squeezed_bispec.py


