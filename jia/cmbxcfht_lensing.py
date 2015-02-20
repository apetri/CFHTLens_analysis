#!python
# Jia Liu 2015/02/19
# This code calculates the model for cmb lensing x weak lensing

import WLanalysis
import os
import numpy as np
from scipy import *
import sys
from scipy.integrate import quad
import scipy.optimize as op
from scipy import interpolate

cmb_dir = '/Users/jia/weaklensing/cmblensing/'
#######################################
####### planck 2015 TT, TE, EE + lowP
#######################################
H0 = 67.27
h = H0/100.0
OmegaM = 0.3156
OmegaV = 1.0-OmegaM
H0_cgs = H0*1e5/3.08567758e24
# assume 0 radiation

#######################################
####### constants & derived params
#######################################
c = 299792.458#km/s
Gnewton = 6.674e-8#cgs cm^3/g/s
rho_c0 = 3.0/8.0/pi/Gnewton*H0_cgs**2#9.9e-30

H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV))
# luminosity distance Mpc
DL = lambda z: (1+z)*c*quad(H_inv, 0, z)[0] 
# comoving distance Mpc
DC = lambda z: c*quad(H_inv, 0, z)[0] 
z_ls = 1090 #last scattering

#######################################
##### find my own curve for dn/dz
#######################################
###### Hand+2014: a, b, c, A = 0.531, 7.810, 0.517, 0.688
dndz_Hand = lambda z: 0.688*(z**0.531+z**(0.531*7.810))/(z**7.810+0.517)
########### van Waerbeke fitting
dndz_VW = lambda z: 1.5*exp(-(z-0.7)**2/0.32**2)+0.2*exp(-(z-1.2)**2/0.46**2)
########### my dndz interpolation 

z_arr, W_wl0 = load(cmb_dir+'W_wl_interp.npy')
W_wl = interpolate.interp1d(z_arr, W_wl0)

Ptable = genfromtxt(cmb_dir+'P_delta_smith03_revised')[::5,]
aa = array([1/1.05**i for i in arange(33)])
zz = 1.0/aa-1
kk = Ptable.T[0]
iZ, iK = meshgrid(zz,kk)
Z, K = Z.flatten(), K.flatten()
P_deltas = Ptable[:,1:].flatten()

Pmatter_interp = interpolate.Rbf(Z, K, P_deltas)
Pmatter = lambda z, k: Pmatter_interp (z, k)

Ckk_integrand = lambda z, ell: 1.0/(H_inv(z)*c*DC(z)**2)*W_wl(z)*W_cmb(z)*Pmatter(ell/DC(z), z)


################### various tests ########
## (1) test dndz_interp - pass #########
#z_arr = linspace(1e-2, 4, 100)
#plot(z_arr, dndz_Hand(z_arr),label='Hand+2014')
#plot(z_arr, dndz_VW(z_arr),label='van Waerbeke+2013')
#plot(z_hist[0],z_hist[1]/0.05,label='CFHT',drawstyle='steps-post')
#plot(z_arr, dndz_interp(z_arr), label='JL interpolation')
#legend()
#xlabel('z')
#ylabel('dn/dz')
#xlim(0,2.5)
#show()

########### (2) my attempt to fit to dndz, aborted
#def chisq_dndz_JL (abcA):
	#a, b, c, A = abcA
	#dndz = A*(z0**a+z0**(a*b))/(z0**b+c)
	#diff = sum(abs(dndz - z1))
	#return diff
#abcA_guess = (0.531, 7.810, 0.517, 0.688)
#out = op.minimize(chisq_dndz_JL, abcA_guess)r
#abcA_JL = [ 0.28391205,  6.79531638,  0.73779074,  0.65322337]
####### do interpolation directly

########## (3) calculate  W_wl ##################
########## prepare for interpolation ############

#z_hist = genfromtxt(cmb_dir+'nz_sumpdf.hist').T
#z0, z1 = z_hist[0,:-1]+0.025, z_hist[1,:-1]/0.05
#z0 = concatenate([[0,], z0, linspace(z0[-1]*1.2, 1200,100)])
#z1 = concatenate([[z1[0],], z1, zeros(100)])
#dndz_interp = interpolate.interp1d(z0, z1)

#W_cmb = lambda z: 1.5*OmegaM*H0**2*(1+z)*H_inv(z)*DC(z)/c*(1-DC(z)/DC(z_ls))

#integrand = lambda zs, z: dndz_interp(zs)*(1-DC(z)/DC(zs))
#W_wl = lambda z: 1.5*OmegaM*H0**2*(1+z)*H_inv(z)*DC(z)/c*quad(integrand, z, 1199.0, args=(z,))[0]
#z_arr200 = linspace(1e-5, 4, 200)
#W_wl_arr = array(map(W_wl, z_arr200))
#save(cmb_dir+'W_wl_interp.npy',array([z_arr200, W_wl_arr]))


### (4) W_cmb & W_wl - test pass #############
#z_arr = linspace(1e-2, 4, 100)
#W_cmb_arr = array(map(W_cmb, z_arr))
#plot(z_arr,W_cmb_arr, label='Planck')
#plot(z_arr,W_wl(z_arr)*amax(W_cmb_arr)/amax(W_wl0), label='CFHT')
#legend(loc=0)
#xlabel('z')
#ylabel('W')
#show()

#### (5) nicaea matter powspec #######
#for i in range(len(zz)):
    #loglog(kk, Ptable[:,i+1])
#xlabel('k [Mpc/h]')
#ylabel('P_delta')
#show()

