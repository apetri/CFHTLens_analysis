import WLanalysis
import glob, os, sys
import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack

cosmo_arr = genfromtxt('/Users/jia/weaklensing/CMBnonGaussian/cosmo_arr.txt',dtype='string')
## e.g.Om0.145_Ol0.855_w-2.211_si1.303

def create_Nicaea_ps (cosmo):
	param0 = float(cosmo[2:7])
	param1 = 1-param0
	param2 = float(cosmo[17:23])
	param3 = float(cosmo[-5:])
	content='''### Cosmological parameters Planck 2015###

Omega_m         {0}            # 0.3156Matter density, cold dark matter + baryons
Omega_de        {1}            # Dark-energy density
w0_de           {2}            # Dark-energy equation of state parameter (constant term)
w1_de            0.0            # Dark-energy equation of state parameter (linear term)
h_100           0.72            # Dimensionless Hubble parameter
Omega_b         046          # Baryon density # 0.046, 0.0492
Omega_nu_mass   0.0             # Massive neutrino density (so far only for CMB)
Neff_nu_mass    0.0             # Effective number of massive neutrinos (only CMB)
normalization   {3}             # 0.831This is sigma_8 if normmode=0 below
n_spec          0.96            # Scalar power-spectrum index


### Flags ###

# Power spectrum prescription
#  linear       Linear power spectrum
#  pd96         Peacock&Dodds (1996)
#  smith03      Smith et al. (2003)
#  smith03_de   Smith et al. (2003) + dark-energy correction from icosmo.org
#  smith03_revised
#               Takahashi et al. 2012, revised halofit parameters
#  coyote10     Coyote emulator v1, Heitmann, Lawrence et al. 2009, 2010
#  coyote13     Coyote emulator v2, Heitmann et al. 2013
snonlinear smith03_revised

# Transfer function
#  bbks         Bardeen, Bond, Kaiser & Szalay (1986)
#  eisenhu      Eisenstein & Hu (1998) 'shape fit'
#  eisenhu_osc  Eisenstein & Hu (1998) with baryon wiggles
stransfer       eisenhu_osc

# Linear growth factor
#  heath        Heath (1977) fitting formula
#  growth_de    Numerical integration of density ODE (recommended)
sgrowth         growth_de

# Dark-energy parametrisation
#  jassal       w(a) = w_0 + w_1*a*(1-a)
#  linder       w(a) = w_0 + w_1*(1-a)
sde_param       linder

normmode        0               # Normalization mode. 0: normalization=sigma_8

# Minimum scale factor
a_min           0.1             # For late Universe stuff
'''.format(param0, param1, param2, param3)
	f=open('/Users/jia/Documents/code/nicaea_2.5/par_files/cosmo.par','w')
	f.write(content)
	f.close()
	os.system('cd /Users/jia/Documents/code/nicaea_2.5/Demo; ./lensingdemo; cp P_kappa /Users/jia/weaklensing/CMBnonGaussian/Pkappa_nicaea/Pkappa_nicaea25_%s_1100; cp P_delta /Users/jia/weaklensing/CMBnonGaussian/Pkappa_nicaea/Pdelta_nicaea25_%s_1100'%(cosmo,cosmo))

#map(create_Nicaea_ps, cosmo_arr)
plot_dir='/Users/jia/weaklensing/CMBnonGaussian/plot/'
ell_gadget=WLanalysis.PowerSpectrum(rand(2048,2048))[0][:34]
van_dir = '/Users/jia/weaklensing/CMBnonGaussian/to_vanessa/'
#import matplotlib.pyplot as plt
create_Nicaea_ps(cosmo_arr[12])
from pylab import *
for cosmo in (cosmo_arr[12],):#cosmo_arr:#(cosmo_arr[18],):
	#create_Nicaea_ps(cosmo)
	pspkPDFgadget=load('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_gadget/{0}_ps_PDF_pk_600b.npy'.format(cosmo))
	ps_gadget=array([pspkPDFgadget[i][0][:34] for i in range(len(pspkPDFgadget))])
	##savetxt(van_dir+'Pkappa_gadget_{}'.format(cosmo),array([ell_gadget, mean(ps_gadget,axis=0),std(ps_gadget,axis=0)]).T, header='ell\tell(ell+1)/2pi*Pkappa\tsigma_(ell(ell+1)/2pi*Pkappa)')
	###ps_gadget[331]=ps_gadget[332]
	ell_nicaea, ps_nicaea=genfromtxt('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_nicaea/Pkappa_nicaea25_{0}_1100'.format(cosmo))[33:-5].T
	ell_nicaea38, ps_nicaea38=genfromtxt('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_nicaea/Pkappa_nicaea25_{0}'.format(cosmo))[33:-5].T

	f=figure(figsize=(6,4.5))
	ax=f.add_subplot(111)
	ax.errorbar(ell_gadget, mean(ps_gadget,axis=0),std(ps_gadget,axis=0),label='gadget (z=38)')
	ax.plot(ell_nicaea, ps_nicaea,'--',lw=1,label='nicaea2.5 (z=1100)')
	ax.plot(ell_nicaea38, ps_nicaea38,'--',lw=1,label='nicaea2.5 (z=38)')
	        
        ########### colin camb comparison ######
        #ell_camb, ps_camb =genfromtxt( '/Users/jia/Desktop/Clkappa_camb_Jia_Om0.296_Ol0.704_w-1.000_si0.786_Ascorrectedbyme.dat').T
        #ax.plot(ell_camb, ps_camb*ell_camb*(ell_camb+1)/2/pi, lw=1,label='camb')
        ########################################
        
        ######## 2 power spectrum traced at z=1100 #####
        ellxx,ps1,ps2=load('/Users/jia/Desktop/test_z1100.npy')
        ax.plot(ellxx,ps1,label='gadget (z=1100)')
        ax.plot(ellxx,ps2,label='gadget (z=1100)')
        
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_title(cosmo)
	ax.set_xlabel(r'$\ell$')
	ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$')
	ax.set_xlim(100,1e4)
	legend(fontsize=10,loc=0)
	ax.set_ylim(6e-5,1e-2)
	plt.tight_layout()
	show()
	#savefig(plot_dir+'ps_{}.jpg'.format(cosmo))
	#close()
	


