#!python
# Jia Liu 2014/5/21
# What the code does: create mass maps for 100 cosmologies, for the CFHT emulator project
# Cluster: XSEDE Stampede

import WLanalysis
from emcee.utils import MPIPool
import os
import numpy as np
from scipy import *
import scipy.ndimage as snd
import sys
#from multiprocessing import Pool

########## define constants ############
print 'start'
sim_dir = '/home1/02977/jialiu/cat/'
KS_dir = '/scratch/02977/jialiu/KSsim/'
cosmo_arr = os.listdir(sim_dir)
params = genfromtxt(KS_dir+'cosmo_params.txt')

kmin = -0.04 # lower bound of kappa bin = -2 SNR
kmax = 0.12 # higher bound of kappa bin = 6 SNR
bins = 600 # for peak counts
sigmaG_arr = (0.5, 1, 1.8, 3.5, 5.3, 8.9)
i_arr = arange(1,14)
R_arr = arange(1,1001)

# constants not used
# zmax = 1.3
# zmin = 0.2
# ngal_arcmin = 5.0
# ngal_cut = ngal_arcmin*(60**2*12)/512**2# = 0.82, cut = 5 / arcmin^2
#PPR512 = 8468.416479647716#pixels per radians
#PPA512 = 2.4633625
#rad2pix = lambda x: around(512/2.0-0.5 + x*PPR512).astype(int) #from radians to pixel location

SIMfn = lambda i, cosmo, R: sim_dir+'%s/emulator_subfield%i_WL-only_%s_4096xy_%04dr.fit'%(cosmo, i, cosmo, R)

KSfn = lambda i, cosmo, R, sigmaG: KS_dir+'%s/subfield%i/sigma%02d/SIM_KS_sigma%02d_subfield%i_%s_%04dr.fit'%(cosmo, i, sigmaG*10, sigmaG*10, i, cosmo,R)

Mask_fn = lambda i, sigmaG: KS_dir+'mask/CFHT_mask_ngal5_sigma%02d_subfield%02d.fits'%(sigmaG*10, i)

peaks_fn = lambda i, cosmo, sigmaG, bins: KS_dir+'peaks/%s/subfield%i/sigma%02d/SIM_peaks_sigma%02d_subfield%i_%s_%03dbins.fit'%(cosmo, i, sigmaG*10, sigmaG*10, i, cosmo, bins)

powspec_fn = lambda i, cosmo, sigmaG: KS_dir+'powspec/%s/subfield%i/sigma%02d/SIM_powspec_sigma%02d_subfield%i_%s.fit'%(cosmo, i, sigmaG*10, sigmaG*10, i, cosmo)


######### functions ######################

### read in MW and yxewm first
#Mw_fn = KS_dir+'SIM_Mw_subfield%i.fit'%(i) # same for all R
#Mw = WLanalysis.readFits(Mw_fn)
Mw_fcn = lambda i: WLanalysis.readFits(KS_dir+'SIM_Mw_subfield%i.fit'%(i))
Mw_arr = map(Mw_fcn, i_arr) # Mw = w (1+m) in a grid

yxewm_fcn = lambda i: WLanalysis.readFits(KS_dir+'yxewm_subfield%i_zcut0213.fit'%(i))
yxewm_arr = map(yxewm_fcn, i_arr)

def fileGen(i, R, cosmo):
	'''
	Input:
	i: subfield range from (1, 2..13)
	R: realization range from (1..1000)
	cosmo: one of the 100 cosmos
	Return:
	Me1 = e1*w
	Me2 = e2*w
	
	'''
	y, x, e1, e2, w, m = yxewm_arr[i].T
	k, s1, s2 = (WLanalysis.readFits(SIMfn(i,cosmo,R)).T)[[0,1,2]]
	s1 *= (1+m)
	s2 *= (1+m)
	eint1, eint2 = WLanalysis.rndrot(e1, e2, iseed=R)#random rotation	
	## get reduced shear
	e1, e2 = WLanalysis.eobs_fun(s1, s2, k, eint1, eint2)

	print 'coords2grid', i, R, cosmo
	A, galn = WLanalysis.coords2grid(x, y, array([k, e1*w, e2*w]))
	Mk, Ms1, Ms2 = A

	### add Mk just for comparison ###
	Mk_fn = KS_dir+'SIM_Mk/%s/SIM_Mk_subfield%i_%s_%04dr.fit'%(cosmo, i, cosmo, R)
	try:
		WLanalysis.writeFits(Mk, Mk_fn)
	except Exception:
		pass
	# galn_fn = KS_dir+'galn_subfield%i.fit'%(i) # same for all R
	return Ms1, Ms2

def KSmap(iiRcosmo):
	'''Input:
	i: subfield range from (1, 2..13)
	R: realization range from (1..1000)
	cosmo: one of the 1000 cosmos
	Return:
	KS inverted map
	Power spectrum
	Peak counts
	'''
	i, R, cosmo = iiRcosmo
	## check if power spectrum and peaks are created already ##
	create_ps_pk = 0
	for sigmaG in sigmaG_arr:
		ps_fn = powspec_fn(i, cosmo, sigmaG)
		pk_fn = peaks_fn(i, cosmo, sigmaG, bins)
		if not os.path.isfile(ps_fn) or not os.path.isfile(ps_fn):
			creat_ps_pk = 1
			break
	if create_ps_pk:
		print 'creating KSmap i, R, cosmo', i, R, cosmo
		Me1, Me2 = fileGen(i, R, cosmo)
		Mw = Mw_arr[i]
		for sigmaG in sigmaG_arr:
			ps_fn = powspec_fn(i, cosmo, sigmaG)
			pk_fn = peaks_fn(i, cosmo, sigmaG, bins)
			
			KS_fn = KSfn(i, cosmo, R, sigmaG)
			create_kmap = 1
			if os.path.isfile(KS_fn):
				try: # make sure it's not a broken file
					kmap = WLanalysis.readFits(KS_fn)
					create_kmap = 0
				except Exception:
					pass
					create_kmap = 1	
			if create_kmap:	
				Me1_smooth = WLanalysis.weighted_smooth(Me1, Mw, PPA=PPA512, sigmaG=sigmaG)
				Me2_smooth = WLanalysis.weighted_smooth(Me2, Mw, PPA=PPA512, sigmaG=sigmaG)
				kmap = WLanalysis.KSvw(Me1_smooth, Me2_smooth)
				try:
					WLanalysis.writeFits(kmap, KS_fn)
				except Exception: #prob don't need try here.
					os.remove(KS_fn)
					WLanalysis.writeFits(kmap, KS_fn)
					pass
			############# power spectrum and peaks ####
			powspec = WLanalysis.PowerSpectrum(kmap, sizedeg=12.0)[-1]
			try:
				WLanalysis.writeFits(powspec, ps_fn)
			except Exception:
				os.remove(ps_fn)
				WLanalysis.writeFits(powspec, ps_fn)
				pass
			mask = WLanalysis.readFits(Mask_fn(i, sigmaG))
			peaks_hist = WLanalysis.peaks_mask_hist(kmap, mask, bins, kmin = kmin, kmax = kmax)
			try:
				WLanalysis.writeFits(peaks_hist,pk_fn)
			except Exception:
				os.remove(pk_fn)
				WLanalysis.writeFits(peaks_hist,pk_fn)
				pass
			

# full set
pool = MPIPool()
iRcosmo = [[i, R, cosmo] for i in i_arr for R in R_arr for cosmo in cosmo_arr]
#for R in R_arr:
	#iRcosmo = [[i, R, cosmo] for i in i_arr for cosmo in cosmo_arr]
pool.map(KSmap, iRcosmo)
pool.close()


#### development test
### 1.pass
#for i in i_arr:
	#print i
	#iRcosmo = [[i, R, cosmo] for R in R_arr[:2] for cosmo in cosmo_arr[:8]]
	#pool = MPIPool()
	#pool.map(KSmap, iRcosmo)
	#pool.close()
	
### 2.pass
#iRcosmo = [[i, R, cosmo] for i in i_arr for R in R_arr[2:4] for cosmo in cosmo_arr[:8]]
#pool = MPIPool()
#pool.map(KSmap, iRcosmo)
#pool.close()

### 3.pass
#pool = MPIPool()
#for i in i_arr:
	#print i
	#iRcosmo = [[i, R, cosmo] for R in R_arr[4:6] for cosmo in cosmo_arr[:8]]
	#pool.map(KSmap, iRcosmo)
#pool.close()

### 4. include ps & pk #pass
#iRcosmo = [[i, R, cosmo] for i in i_arr[-3:] for R in R_arr[30:32] for cosmo in cosmo_arr[:2]]
#pool = MPIPool()
#pool.map(KSmap, iRcosmo)
#pool.close()

print 'DONE-DONE-DONE', len(iRcosmo)
#savetxt('/home1/02977/jialiu/done_KS.ls',zeros(5))


