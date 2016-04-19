##########################################################
### This code is for Jia's project B - 2nd try (4/18/2016)
### try to find
### observational evidence of over density at peak location
### as discovered by Yang 2011.
### It does the following:
### 1) create kappa_proj map, using L12 M_star->M_halo relation
### 2) relation between kappa_proj and kappa_lens
### 3) for each peak in kappa_proj, find all contributing halos, 
### relation N_halo (kappa_peak) for # of halos contributing to half kappa
### 4) same, but for random direction

import numpy as np
from scipy import *
from pylab import *
import os
import WLanalysis
from scipy import interpolate,stats
from scipy.integrate import quad
import scipy.optimize as op
import sys, os

make_kappa_predict = 0
if make_kappa_predict:
        ######## for stampede #####
        from emcee.utils import MPIPool
        obsPK_dir = '/home1/02977/jialiu/obsPK/'
else:
        ######## for laptop #####
        obsPK_dir = '/Users/jia/weaklensing/obsPK/'
        plot_dir = obsPK_dir+'plot/'

########### constants ######################

sizes = (1330, 800, 1120, 950)
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
PPR512=8468.416479647716
PPA512=2.4633625
c = 299792.458#km/s
Gnewton = 6.674e-8#cgs cm^3/g/s^2
H0 = 70.0
h = 0.7
OmegaM = 0.3#0.25#
OmegaV = 1.0-OmegaM
#rho_c0 = 9.9e-30#g/cm^3
M_sun = 1.989e33#gram
sigmaG_arr = (0.5, 1.0, 1.8, 3.5, 5.3, 8.9)

############################################
############ functions #####################
############################################

########### generate maps ##################

maskGen = lambda Wx: load(obsPK_dir+'mask/Mask_W%i_0.5_sigmaG10.npy'%(Wx))

kmapGen = lambda Wx, sigmaG: WLanalysis.readFits(obsPK_dir+'kappa_lens/W%i_KS_1.3_lo_sigmaG%02d.fit'%(Wx,sigmaG*10))*maskGen(Wx)

cat_gen = lambda Wx: np.load(obsPK_dir+'CFHTLens_2016-03-16T13-41-52_W%i.npy'%(Wx))
# columns: (0)ALPHA_J2000 (1)DELTA_J2000     
# (2)e1 (3)e2 (4)weight (5)MASK (6)Z_B (7)m (8)c2 
# (9)LP_Mr (10)LP_Mi (11)LP_Mz (12)LP_log10_SM_MED 
# (13)MAG_r (14)MAG_i (15)MAG_y (16)MAG_z

##############################################
########## cosmology #########################
##############################################

# growth factor
Hcgs = lambda z: H0*sqrt(OmegaM*(1+z)**3+OmegaV)*3.24e-20
H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV))
# luminosity distance Mpc
DC_integral = lambda z: c*quad(H_inv, 0, z)[0]
z_arr = linspace(0.1, 1.4, 1000)
DC_arr0 = array([DC_integral(z) for z in z_arr])
DC = interpolate.interp1d(z_arr, DC_arr0)
DA = lambda z: DC(z)/(1.0+z)
DL = lambda z: DC(z)*(1.0+z)
# find the rest magnitude at the galaxy, from observed magnitude cut
#M_rest_fcn = lambda M_obs, z: M_obs - 5.0*log10(DL_interp(z)) - 25.0
##rho_cz = lambda z: rho_c0*(OmegaM*(1+z)**3+(1-OmegaM))#critical density
rho_cz = lambda z: 0.375*Hcgs(z)**2/pi/Gnewton
