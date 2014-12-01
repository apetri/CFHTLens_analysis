# Jia Liu 11/13/2014
# after found bug in power spectrum computation, this code is to 
# (1) compute the ps for all 13 subfields for 0.5 smoothing

import numpy as np
from scipy import *
import os,sys
import WLanalysis
from emcee.utils import MPIPool

BG = str(sys.argv[1])

fsky_pass = array([0.800968170166,0.639133453369,0.686164855957,0.553855895996,
		0.600227355957,0.527587890625,0.671237945557,0.494361877441,
		0.565235137939,0.592998504639,0.584747314453,0.530345916748,
		0.417697906494])
fsky_all = array([0.839298248291,0.865875244141,0.809467315674,
		  0.864688873291,0.679264068604,0.756385803223,
		  0.765892028809,0.747268676758,0.77250289917,
		  0.761451721191,0.691867828369,0.711254119873,
		  0.745429992676])

if BG == 'pass':
	fsky = fsky_pass
	Mask_fcn = lambda i: WLanalysis.readFits('/scratch/02977/jialiu/KSsim/mask/BAD_CFHT_mask_ngal5_sigma05_subfield%02d.fits'%(i))
	fn = '/home1/02977/jialiu/KSsim/cfhtcov-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/BAD_ps_fidu39'
if BG == 'all':
	fsky = fsky_all
	Mask_fcn = lambda i: WLanalysis.readFits('/scratch/02977/jialiu/KSsim/mask/CFHT_mask_ngal5_sigma05_subfield%02d.fits'%(i))
	fn = '/home1/02977/jialiu/KSsim/cfhtcov-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/ALL_ps_fidu39'

fsky_sum = sum(fsky)

mask_arr = map(Mask_fcn, range(1,14))

kmap_fcn = lambda i, r: WLanalysis.readFits('/home1/02977/jialiu/KSsim/cfhtcov-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/subfield%i/sigma05/SIM_KS_sigma05_subfield%i_WL-only_cfhtcov-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800_4096xy_%04dr.fit'%(i,i,r))*mask_arr[i-1]


ps_gen = lambda i, r: WLanalysis.PowerSpectrum(kmap_fcn(i, r), sizedeg=12.0)[-1][11:]
##ps_allfield_gen = lambda r: sum(array([[ps_gen(i, r)] for i in range(1,14)]).squeeze(),axis=0)/fsky_sum
def ps_allfield_gen(r):
	print r
	return sum(array([[ps_gen(i, r)] for i in range(1,14)]).squeeze(),axis=0)/fsky_sum

pool = MPIPool()
all_ps = pool.map(ps_allfield_gen, range(1,1001))
np.save(fn,all_ps)