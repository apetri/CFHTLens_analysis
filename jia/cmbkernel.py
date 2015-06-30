#!python
# compute CMB lensing, zmax where 99% kernel is covered

from scipy import *
import numpy as np
from scipy.integrate import quad

cmbNG_dir = '/Users/jia/weaklensing/CMBnonGaussian/'
H0 = 72.0
h = H0/100.0
c = 299792.458#km/s
z_ls = 1100
Om_arr = genfromtxt('/Users/jia/weaklensing/CFHTLenS/emulator/cosmo_params.txt').T[0]

############### test with omega radiation ###############
#OmegaM = 0.3
#OmegaR = 0#8.24e-5
#OmegaV = 1-OmegaM-OmegaR

#H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV+OmegaR*(1+z)**4))#H^-1
#DC = lambda z: c*quad(H_inv, 0, z)[0] # comoving distance Mpc

#Omega_radiation = 8.24e-5
#r(z=38)=11316.94
#r(z=165)=12557.87
#r(z=1100)=13241.61

#Omega_radiation = 0
#r(z=38)=11327.04
#r(z=165)=12581.56
#r(z=1100)=13303.41
###########################################

def W_cmbD_fcn (OmegaM):
	OmegaV = 1.0-OmegaM
	H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV))#H^-1
	DC = lambda z: c*quad(H_inv, 0, z)[0] # comoving distance Mpc
	W_cmb = lambda z: 1.5*OmegaM*H0**2*(1+z)*H_inv(z)*DC(z)/c*(1-DC(z)/DC(z_ls))
	W_cmbD = lambda z: W_cmb(z)/(1.0+z)
	return W_cmbD

z_arr = logspace(0, 3, 50)
def find_ratio(OmegaM):
	W_cmbD = W_cmbD_fcn(OmegaM)
	norm = quad(W_cmbD, 0, z_ls)[0]
	ratio_arr = array([quad(W_cmbD, 0, z)[0] for z in z_arr])/norm
	#find_fn = lambda z, ratio: quad(W_cmbD, 0, z)[0]/norm - ratio
	#z90 = float(bisect(find_fn, 5, 300, args=(0.9,)))
	#z99 = float(bisect(find_fn, 5, 300, args=(0.99,)))
	#return OmegaM, norm, z90, z99
	ratio_arr = array([quad(W_cmbD, 0, z)[0] for z in z_arr])/norm
	return ratio_arr
ratio_arr = find_ratio(0.3)
ratio26_arr = find_ratio(0.26)

################ junk ########################
#norms = genfromtxt(cmbNG_dir+'kernel_norm_growth.txt')

#all_norms = map(kernel_norm, Om_arr)
#savetxt(cmbNG_dir+'kernel_norm_growth.txt', all_norms, header='OmegaM\tD*kernel_norm')

#from scipy.optimize import bisect, newton, brenth, ridder
#all_zmax = genfromtxt(cmbNG_dir+'kernel_zmax.txt')

#all_zmax = map(find_99z, Om_arr)
#savetxt(cmbNG_dir+'kernelD_zmax.txt', all_zmax, header='OmegaM\tD*kernel_norm\tz(90%)\tz(99%)')

