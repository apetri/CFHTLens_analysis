# Jia Liu 09/25/2014
# This code tests MF covariance and interpolation
# using results run by A. Petri

import numpy as np
from scipy import *
import scipy.optimize as op
from scipy import interpolate#,stats
import os
import WLanalysis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import matplotlib.gridspec as gridspec 
import scipy.ndimage as snd

params = genfromtxt('/scratch/02977/jialiu/KSsim/cosmo_params.txt')

loadMF = lambda Om, w, si8, i: np.load('/scratch/02918/apetri/Storage/wl/features/CFHT/cfht_masked_BAD/Om%.3f_Ol%.3f_w%.3f_ns0.960_si%.3f/subfield%i/sigma10/minkowski_all.npy'%(Om,1-Om,w,si8,i))

loadMF_cov = lambda i: np.load('/scratch/02918/apetri/Storage/wl/features/CFHT/cfht_masked_BAD/Om0.260_Ol0.740_w-1.000_ns0.960_si0.800_cov/subfield%i/sigma10/minkowski_all.npy'%(i))

fsky = array([0.800968170166,0.639133453369,0.686164855957,0.553855895996,
		0.600227355957,0.527587890625,0.671237945557,0.494361877441,
		0.565235137939,0.592998504639,0.584747314453,0.530345916748,
		0.417697906494])
fsky_all = array([0.839298248291,0.865875244141,0.809467315674,
		  0.864688873291,0.679264068604,0.756385803223,
		  0.765892028809,0.747268676758,0.77250289917,
		  0.761451721191,0.691867828369,0.711254119873,
		  0.745429992676])

### 91 params
def sumMF (iparam, loadcov=False):
	Om, w, si8 = iparam
	if loadcov:
		iMF = array([loadMF_cov(i) for i in range(1,14)])
	else:
		iMF = array([loadMF(Om, w, si8, i) for i in range(1,14)])
	sumMF = sum(fsky.reshape(13,1,1)*iMF,axis=0)/float(sum(fsky))
	return sumMF

### covariance matrix

'/home1/02977/jialiu/KSsim/MF_sum'