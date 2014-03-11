#! /afs/rhic.bnl.gov/@sys/opt/astro/SL64/anaconda/bin
# yeti: /vega/astro/users/jl3509/tarball/anacondaa/bin/python
# Jia Liu 2014/3/10
# Overview: this code calculate peaks and power spectrum using KS maps 
# from both simulation and CFHT observation, and fit cosmology to CFHT

import WLanalysis
from emcee.utils import MPIPool
import os
import numpy as np
from scipy import *
import sys
from multiprocessing import Pool # has bug on my laptop, but fine with astro
#[Errno 35] Resource temporarily unavailable
#import scipy.ndimage as snd

################ steps #####################
#1) count peaks, with mask
#2) power spectrum, without mask, but note need to use good fields (to be implemented)
#3) construct model, using rz1
#4) fit model to CFHT
#5) fit model to simulation pz, rz2

########## beging: define constants ############
kmin = -0.04 # lower bound of kappa bin = -2 SNR
kmax = 0.12 # higher bound of kappa bin = 6 SNR
ngal_arcmin = 5.0
zmax=1.3
zmin=0.2

ngal_cut = ngal_arcmin*(60**2*12)/512**2# = 0.82, cut = 5 / arcmin^2
PPR512=8468.416479647716#pixels per radians
PPA512=2.4633625

full_dir = '/direct/astro+astronfs01/workarea/jia/CFHT/full_subfields/'
KSCFHT_dir = '/direct/astro+astronfs03/workarea/jia/CFHT/KSCFHT/'
KSsim_dir = '/direct/astro+astronfs03/workarea/jia/CFHT/KSsim/'
fit_dir = '/direct/astro+astronfs03/workarea/jia/CFHT/KSsim/fit/'
fidu='mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800'
hi_w='mQ3-512b240_Om0.260_Ol0.740_w-0.800_ns0.960_si0.800'
hi_s='mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.850'
hi_m='mQ3-512b240_Om0.290_Ol0.710_w-1.000_ns0.960_si0.800'

####### maps to process #########
processmaps = 0

zg_arr = ('pz','rz1','rz2')
bins_arr = arange(10, 110, 15)
sigmaG_arr = (0.5, 1, 1.8, 3.5, 5.3, 8.9)
i_arr=[1,2]
R_arr=arange(1,129)
cosmo_arr=(fidu,hi_m,hi_w,hi_s)
Rtol=len(R_arr)
dp = array([0.03, 0.2, 0.05])
####### cosmology model configuration ######

config_2_21 = array([[1.8, 25, 3, 17],
		     [3.5, 25, 5, 12]])#2 smoothing scale, tot 24 bins
config = config_2_21
####### end: define constant ####

########## beging: functions ###########
KSCFHT_fn = lambda i, sigmaG: KSCFHT_dir+'CFHT_KS_sigma%02d_subfield%02d.fits'%(sigmaG*10, i)

Mask_fn = lambda i, sigmaG: KSCFHT_dir+'CFHT_mask_ngal5_sigma%02d_subfield%02d.fits'%(sigmaG*10, i)

KSsim_fn = lambda i, cosmo, R, sigmaG, zg: KSsim_dir+'%s/SIM_KS_sigma%02d_subfield%i_%s_%s_%04dr.fit'%(cosmo, sigmaG*10, i, zg, cosmo,R)#i=subfield, cosmo, R=realization, sigmaG=smoothing, zg=zgroup=(pz, rz, rz2)

# this is one matrix for Rtol realizations, where Rtol is the total number of realizations
peaks_fn = lambda i, cosmo, Rtol, sigmaG, zg, bins: KSsim_dir+'peaks/SIM_peaks_sigma%02d_subfield%i_%s_%s_%04dR_%03dbins.fit'%(sigmaG*10, i, zg, cosmo, Rtol, bins)

powspec_fn = lambda i, cosmo, Rtol, sigmaG, zg: KSsim_dir+'powspec/SIM_powspec_sigma%02d_subfield%i_%s_%s_%04dR.fit'%(sigmaG*10, i, zg, cosmo, Rtol)

def Psingle (i, sigmaG, zg, bins, cosmo, kmin=kmin, kmax=kmax, pk=True):
	'''return a function that:
	takes in i, sigmaG, zg, bins, cosmo. 
	return ipeaks_list (R)'''
	def ips_pk_single (R):#, sigmaG, zg, bins):
		kmap = WLanalysis.readFits(KSsim_fn(i, cosmo, R, sigmaG, zg))
		if pk:#peaks
			mask = WLanalysis.readFits(Mask_fn(i, sigmaG))
			peaks_hist = WLanalysis.peaks_mask_hist(kmap, mask, bins, kmin=kmin, kmax=kmax)
			return peaks_hist
		else:#powspec
			ell_arr, powspec = WLanalysis.PowerSpectrum(kmap, sizedeg=12.0)
			return powspec
	return ips_pk_single

def Pmat (iRcosmo, Rtol=Rtol, R0 = 1):
	'''
	Input:
	iRcosmo = (i, bins, sigmaG, cosmo, zg)
	Rtol: total number of realizations, and count peaks for realizations #(R0, R0+1, .. R0+Rtotl-1)
	R0: the first realization, if not starting from 1
	if bins = 0 return power spectrum, else return peak counts.
	Return:
	A maxtrix of shape=(Rtol x bins)
	'''
	i, sigmaG, zg, bins, cosmo = iRcosmo
	if bins == 0:#powspec
		fn = powspec_fn(i, cosmo, Rtol, sigmaG, zg)
	else:#peaks
		fn = peaks_fn(i, cosmo, Rtol, sigmaG, zg, bins)
	if os.path.isfile(fn):
		mat = WLanalysis.readFits(fn)
	else:
		print 'i, bins, sigmaG', i, bins, sigmaG
		map_fcn = Psingle (i, sigmaG, zg, bins, cosmo, pk=bins)
		#p = Pool(Rtol/4)#use multiprocessing on 1 single core
		mat = array(map(map_fcn,R_arr))
		WLanalysis.writeFits(mat, fn)
	return mat
############ end: functions ##############

############ begein: calculate ############

#1. process maps
if processmaps:
## Make sure the thread we're running on is the master
#if not pool.is_master():
	#pool.wait()
	#sys.exit(0)
	## logger.debug("Running with MPI...")
	iRcosmo_pk = [[i, sigmaG, zg, bins, cosmo] for i in i_arr for sigmaG in sigmaG_arr for zg in zg_arr for bins in bins_arr for cosmo in cosmo_arr]
	iRcosmo_ps = [[i, 0.5, zg, 0, cosmo] for i in i_arr for zg in zg_arr for cosmo in cosmo_arr]
	pool = MPIPool()
	pool.map(Pmat, iRcosmo_ps+iRcosmo_pk)
	pool.close()

#2. cosmology model
#2.1 covariance matrix
# (1.8 arcmin, 25bins)[3:17] # 14 bins
# (3.5 arcmin, 25bins)[5:12] # 7 bins
## build array of sigmaG, bins, start, end, to prepare for cov, fisher mat
#config_2_21 = array([[1.8, 25, 3, 17],
		     #[3.5, 25, 5, 12]])

bintol = int(sum(config[:,-1]-config[:,-2])) # total bins used in cosmo model
cosmo_mat = zeros((4, Rtol, bintol))
obs_rz2_mat = zeros((Rtol, bintol))
obs_pz_mat = zeros((Rtol, bintol))

j = 0
for cosmo in cosmo_arr:
	for i in i_arr:
		k = 0
		for iconfig in config:
			sigmaG, bins, x0, x1 = iconfig
			l = x1-x0
			imat = Pmat((i, sigmaG, 'rz1', bins, cosmo))[:,x0:x1]
			cosmo_mat[j, :, k:k+l] += imat
			if j == 0: #fidu cosmo, then append to obs_mats
				imat_pz = Pmat((i, sigmaG, 'pz', bins, cosmo))[:,x0:x1]
				imat_rz2 = Pmat((i, sigmaG, 'rz2', bins, cosmo))[:,x0:x1]
				obs_pz_mat[:,k:k+l] += imat_pz
				obs_rz2_mat[:,k:k+l] += imat_rz2
			k += x1-x0
	j+=1

fidu_params = array([0.26, -1.0, 0.8])
cov_mat = cov(cosmo_mat[0], rowvar = 0)#rowvar is the row contaning observations, aka 128R
cov_inv = np.mat(cov_mat).I
# cosmo_arr=(fidu,hi_m,hi_w,hi_s)
fidu_avg = mean(cosmo_mat[0], axis = 0)
him_avg, hiw_avg, his_avg = mean(cosmo_mat[1:], axis = 1)
dNdm = (him_avg - fidu_avg)/dp[0]
dNdw =(hiw_avg - fidu_avg)/dp[1] 
dNds = (his_avg - fidu_avg)/dp[2]
X=np.mat([dNdm, dNdw, dNds])

# unbiased estimator for covariance matrix (R-N_bins-2)/(R-1)
def cosmo_fit (obs):
	Y = np.mat(obs-fidu_avg)
	del_p = ((X*cov_inv*X.T).I)*(X*cov_inv*Y.T)
	m, w, s = np.squeeze(np.array(del_p.T))+fidu_params
	del_N = Y-del_p.T*X
	chisq = (Rtol-bintol-2.0)/(Rtol-1.0)*del_N*cov_inv*del_N.T
	return chisq, m, w, s

fit_rz2_fn = fit_dir+'fit_rz2_config_%isubfields_%04dR_%03dbins'%(len(i_arr), Rtol, bintol)
fit_pz_fn = fit_dir+'fit_pz_config_%isubfields_%04dR_%03dbins'%(len(i_arr), Rtol, bintol)

p = Pool(Rtol)
fits_rz2 = array(p.map(cosmo_fit, obs_rz2_mat))
fits_pz = array(p.map(cosmo_fit, obs_pz_mat))

savetxt(fit_rz2_fn, fits_rz2)
savetxt(fit_pz_fn, fits_pz)

############# end: calculate ###############
savetxt(KSsim_dir+'done.ls',zeros(5))
print 'done-done-done!'
sys.exit(0)