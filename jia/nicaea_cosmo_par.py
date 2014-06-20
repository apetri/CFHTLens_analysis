import numpy as np
from scipy import *
import WLanalysis
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from pylab import *

plot_dir = '/Users/jia/weaklensing/CFHTLenS/plot/'

def gen_cosmopar(om, w, si8, n):
	print n
	fn = '/Users/jia/CFHTLenS/emulator/nicaea_params/cosmo%i.par'%(n)
	text ='''### Cosmological parameters ###

Omega_m		%.3f		# Matter density, cold dark matter + baryons
Omega_de	%.3f		# Dark-energy density
w0_de		%.3f		# Dark-energy equation of state parameter (constant term)
w1_de		 0.0	  	# Dark-energy equation of state parameter (linear term)
h_100		0.70		# Dimensionless Hubble parameter
Omega_b		0.0227		# Baryon density
Omega_nu_mass	0.0		# Massive neutrino density (so far only for CMB)
Neff_nu_mass	0.0		# Effective number of massive neutrinos (only CMB)
normalization	%.3f		# This is sigma_8 if normmode=0 below
n_spec		0.96		# Scalar power-spectrum index

### Flags ###

# Power spectrum prescription
#  linear	Linear power spectrum
#  pd96		Peacock&Dodds (1996)
#  smith03	Smith et al. (2003)
#  smith03_de	Smith et al. (2003) + dark-energy correction from icosmo.org
#  coyote10	Coyote emulator, Heiman, Lawrence et al. 2009, 2010
snonlinear	smith03_de

# Transfer function
#  bbks		Bardeen, Bond, Kaiser & Szalay (1986)
#  eisenhu	Eisenstein & Hu (1998) 'shape fit'
#  eisenhu_osc  Eisenstein & Hu (1998) with baryon wiggles
stransfer	eisenhu_osc

# Linear growth factor
#  heath	Heath (1977) fitting formula
#  growth_de	Numerical integration of density ODE (recommended)
sgrowth		growth_de

# Dark-energy parametrisation
#  jassal	w(a) = w_0 + w_1*a*(1-a)
#  linder	w(a) = w_0 + w_1*(1-a)
sde_param	linder

#sde_param	poly_DE
#N_poly_de	4
#w_poly_de	-0.6 -0.2 0.2 0.1

normmode	0		# Normalization mode. 0: normalization=sigma_8

# Minimum scale factor
a_min		0.1		# For late Universe stuff'''%(om, 1-om, w, si8)
	f = open(fn, 'w')
	f.write(text)
	f.close
	

#cosmo_params =  genfromtxt('/Users/jia/CFHTLenS/emulator/cosmo_params.txt')
#om_arr = arange (0.12, 1.02, 0.04)
#w_arr = arange(-3, 0.1, 0.1)
#si8_arr = arange(0.1,1.6,0.1)
#cosmo_params_grid = array([[om, w, si8] for om in om_arr for w in w_arr for si8 in si8_arr])
#cc = concatenate([cosmo_params, cosmo_params_grid])
#savetxt('/Users/jia/CFHTLenS/emulator/nicaea_params/cosmo_params_grid.txt',cc)

cc = genfromtxt('/Users/jia/CFHTLenS/emulator/nicaea_params/cosmo_params_grid.txt')

#for n in range(len(cc)):
	#om, w, si8 = cc[n]
	#gen_cosmopar(om, w, si8, n)

##### manually add noise
#noise = np.random.normal(0, 0.29,size=(20,512,512))
#def smooth2(kmap):
	#return WLanalysis.smooth(kmap,0.5*2.4633625)

#smooth_arr = array(map(smooth2, noise))
#noise_ps_arr = array(map(WLanalysis.PowerSpectrum, smooth_arr))
#noise_ps = mean(noise_ps_arr,axis=0)
#savetxt('/Users/jia/CFHTLenS/emulator/nicaea_params/noise_ps',noise_ps)

noise_ps = genfromtxt('/Users/jia/CFHTLenS/emulator/nicaea_params/noise_ps')

#get_P_ell = lambda n: genfromtxt('/Users/jia/CFHTLenS/emulator/nicaea_params/P_kappa%i'%(n)).T[1]

#P_ell_arr = array(map(get_P_ell, range(len(cc))))
#WLanalysis.writeFits(P_ell_arr,'/Users/jia/CFHTLenS/emulator/nicaea_params/P_kappa_arr')

P_ell_arr = WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/nicaea_params/P_kappa_arr')

P_ell_noise_arr = P_ell_arr+noise_ps[1]

#WLanalysis.writeFits(P_ell_noise_arr[:91],'/Users/jia/CFHTLenS/emulator/nicaea_params/P_kappa_noise_arr91.fit')

ell_arr = logspace(log10(110.01746692),log10(25207.90813028),50)#[11:]

for i in arange(91):
	loglog(ell_arr,P_ell_noise_arr[i])
xlim(ell_arr[11],ell_arr[-1])
title('Noise: Gaussian width=0.29')
savefig(plot_dir+'emu_91cosmo_nicaea_noise.jpg')

#title('No Noise')
#savefig(plot_dir+'emu_91cosmo_nicaea_nonoise.jpg')
close()