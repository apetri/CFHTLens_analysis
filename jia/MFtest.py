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