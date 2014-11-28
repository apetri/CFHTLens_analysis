#!python
# Jia Liu 2014/5/21
# What the code does: create mass maps for 100 cosmologies, for the CFHT emulator project
# Cluster: XSEDE Stampede
# 2014/11/27: clean up codes, remove useless lines..separate KS creation and ps/pk calc.

import WLanalysis
from emcee.utils import MPIPool
import os
import numpy as np
from scipy import *
import scipy.ndimage as snd
import sys
#from multiprocessing import Pool

##### note 2014/10/18 for recovering maps due to scratch failure
##### look for "(cov 1)" for covariance cosmology generation (3? places)

########################################
########## define constants ############
########################################
i = int(sys.argv[1]) # subfield count, to reduce computing memory burdun
KS_dir = '/scratch/02977/jialiu/KSsim/'
sim_dir = '/home1/02977/jialiu/cat/'
cosmo_arr = os.listdir(sim_dir)
params = genfromtxt(KS_dir+'cosmo_params.txt')
kmin = -0.04 # lower bound of kappa bin = -2 SNR
kmax = 0.12 # higher bound of kappa bin = 6 SNR
bins = 25#600 # for peak counts
sigmaG_arr = array([0.5, 1, 1.8, 3.5, 5.3, 8.9])
R_arr = arange(1,1001)
PPA512 = 2.4633625
i_arr = range(1,14)
# constants not used
# zmax = 1.3
# zmin = 0.2
# ngal_arcmin = 5.0
# ngal_cut = ngal_arcmin*(60**2*12)/512**2# = 0.82, cut = 5 / arcmin^2
#PPR512 = 8468.416479647716#pixels per radians
#rad2pix = lambda x: around(512/2.0-0.5 + x*PPR512).astype(int) #from radians to pixel location

#########################################################################
#######(cov 1) next 2 lines are for cov mat KS creation 09/17/2014 #############
#########################################################################

#SIMfn = lambda i, cosmo, R: '/home1/02977/jialiu/cov_cat/emulator_subfield%i_WL-only_cfhtcov-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800_4096xy_%04dr.fit'%(i, R)

#KSfn = lambda i, cosmo, R, sigmaG: '/home1/02977/jialiu/KSsim/cfhtcov-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/subfield%i/sigma%02d/SIM_KS_sigma%02d_subfield%i_WL-only_cfhtcov-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800_4096xy_%04dr.fit'%(i, sigmaG*10, sigmaG*10, i, R)

########## file names ##################################################
########## (cov 1) comment this block for cov matrix simulations #####
########################################################################

peaks_sum_sf_fn = lambda cosmo, sigmaG, i, BG: KS_dir+'peaks_sf/SIM_peaks_sigma%02d_%s_%03dbins_subfield%02d_%s.npy'%(sigmaG*10, cosmo, bins, i, BG)

powspec_sum_sf_fn = lambda cosmo, sigmaG, i, BG: KS_dir+'powspec_sf/SIM_powspec_sigma%02d_%s_subfield%02d_%s.npy'%(sigmaG*10, cosmo, i, BG)

peaks_sum_fn = lambda cosmo, sigmaG, BG: KS_dir+'peaks_sum/SIM_peaks_sigma%02d_%s_%03dbins_%s.npy'%(sigmaG*10, cosmo, bins, BG)

powspec_sum_fn = lambda cosmo, sigmaG, BG: KS_dir+'powspec_sum/SIM_powspec_sigma%02d_%s_%s.npy'%(sigmaG*10, cosmo, BG)

########## maps, fits file
SIMfn0 = lambda i, cosmo, R: sim_dir+'%s/emulator_subfield%i_WL-only_%s_4096xy_%04dr.fit'%(cosmo, i, cosmo, R)

############ this is to fix the corrupted files ##################
def SIMfn (i, cosmo, R):
	if cosmo == 'emu1-512b240_Om0.380_Ol0.620_w-2.424_ns0.960_si0.199' and R in (216, 285, 299, 294) and i == 7:
		return '/home1/02977/jialiu/KSsim/corruptfiles/emulator_subfield%i_WL-only_%s_4096xy_%04dr.fit'%(cosmo, i, cosmo, R)
	else:
		return SIMfn0(i, cosmo, R)
##################################################################

KSfn = lambda i, cosmo, R, sigmaG: KS_dir+'%s/subfield%i/sigma%02d/SIM_KS_sigma%02d_subfield%i_%s_%04dr.fit'%(cosmo, i, sigmaG*10, sigmaG*10, i, cosmo,R)
	

##########################################
######### functions ######################
##########################################

Mask_bad_fn = lambda i, sigmaG: WLanalysis.readFits(KS_dir+'mask/BAD_CFHT_mask_ngal5_sigma%02d_subfield%02d.fits'%(sigmaG*10, i))
Mask_all_fn = lambda i, sigmaG: WLanalysis.readFits(KS_dir+'mask/CFHT_mask_ngal5_sigma%02d_subfield%02d.fits'%(sigmaG*10, i))
Mw_fcn = lambda i: WLanalysis.readFits(KS_dir+'SIM_Mw_subfield%i.fit'%(i))
yxewm_fcn = lambda i: WLanalysis.readFits(KS_dir+'yxewm_subfield%i_zcut0213.fit'%(i))
######### next 4 lines for 1 subfield only
Mw = Mw_fcn(i)
y, x, e1, e2, w, m = yxewm_fcn(i).T
mask_bad_arr = array([Mask_bad_fn(i, sigmaG) for sigmaG in sigmaG_arr])
mask_all_arr = array([Mask_all_fn(i, sigmaG) for sigmaG in sigmaG_arr])
######### next 2 lines are for 13 fields, also need to change lines inside the code
#Mw_arr = map(Mw_fcn, i_arr) # Mw = w (1+m) in a grid
#yxewm_arr = map(yxewm_fcn, i_arr)
###############################################################

print 'got yxewm_arr'
def fileGen(i, R, cosmo):
	'''
	Put catalogue to grid, with (1+m)w correction. Mw is already done.
	also add randomly rotated noise
	Input:
	i: subfield range from (1, 2..13)
	R: realization range from (1..1000)
	cosmo: one of the 100 cosmos
	Return:
	Me1 = e1*w
	Me2 = e2*w
	
	'''
	#y, x, e1, e2, w, m = yxewm_arr[i-1].T
	s1, s2 = (WLanalysis.readFits(SIMfn(i,cosmo,R)).T)[[1,2]]
	s1 *= (1+m)
	s2 *= (1+m)
	eint1, eint2 = WLanalysis.rndrot(e1, e2, iseed=R)#random rotation	
	e1red, e2red = s1+eint1, s2+eint2
	A, galn = WLanalysis.coords2grid(x, y, array([e1red*w, e2red*w]))
	Ms1, Ms2 = A
	return Ms1, Ms2

def KSmap_massproduce(iiRcosmo):
	'''Input:
	iiRcosmo = [i, R, cosmo]
	i: subfield range from (1, 2..13)
	R: realization range from (1..1000)
	cosmo: one of the 1000 cosmos
	Return:
	KS inverted map
	'''
	i, R, cosmo = iiRcosmo
	create_kmap = 0
	for sigmaG in sigmaG_arr:
		KS_fn = KSfn(i, cosmo, R, sigmaG)
		if not WLanalysis.TestFitsComplete(KS_fn):
			create_kmap = 1
			break
	if create_kmap:
		print 'Mass Produce, creating KS: ', i, R, cosmo
		try:
			Me1, Me2 = fileGen(i, R, cosmo)
			for sigmaG in sigmaG_arr:
				KS_fn = KSfn(i, cosmo, R, sigmaG)
				if not WLanalysis.TestFitsComplete(KS_fn):
				
					Me1_smooth = WLanalysis.weighted_smooth(Me1, Mw, PPA=PPA512, sigmaG=sigmaG)
					Me2_smooth = WLanalysis.weighted_smooth(Me2, Mw, PPA=PPA512, sigmaG=sigmaG)
					kmap = WLanalysis.KSvw(Me1_smooth, Me2_smooth)
					try:
						WLanalysis.writeFits(kmap, KS_fn)
					except Exception: #prob don't need try here.
						os.remove(KS_fn)
						WLanalysis.writeFits(kmap, KS_fn)
						pass
		except Exception:
			print 'corrupted file', iiRcosmo
			pass
		

def KSmap_single(i, R, cosmo, sigmaG):
	'''Input:
	i: subfield range from (1, 2..13)
	R: realization range from (1..1000)
	cosmo: one of the 1000 cosmos
	sigmaG: smoothing scale
	Return:
	KS inverted map
	'''
	KS_fn = KSfn(i, cosmo, R, sigmaG)
	isfile, kmap = WLanalysis.TestFitsComplete(KS_fn, return_file = True)
	if isfile == False:
		Me1, Me2 = fileGen(i, R, cosmo)
		#Mw = Mw_fcn(i)
		Me1_smooth = WLanalysis.weighted_smooth(Me1, Mw, PPA=PPA512, sigmaG=sigmaG)
		Me2_smooth = WLanalysis.weighted_smooth(Me2, Mw, PPA=PPA512, sigmaG=sigmaG)
		kmap = WLanalysis.KSvw(Me1_smooth, Me2_smooth)
		WLanalysis.writeFits(kmap, KS_fn)
	return kmap

def create_ps (iiRcosmoSigma):
	'''grab KSmap, create 0.5 arcmin powspec, and 1, 1.8, 3.5, 5.7 peaks
	each time create 1000 realizations for specifice i, cosmo, sigmaG
	'''
	i, R, cosmo, sigmaG = iiRcosmoSigma
	kmap = KSmap_single(i, R, cosmo, sigmaG)
	idx = int(where(sigmaG_arr==sigmaG)[0])
	mask_all = mask_all_arr[idx]
	mask_bad = mask_bad_arr[idx]
	pspk_all = WLanalysis.PowerSpectrum(kmap*mask_all, sizedeg=12.0)[-1]
	pspk_bad = WLanalysis.PowerSpectrum(kmap*mask_bad, sizedeg=12.0)[-1]
	return pspk_all, pspk_bad

def create_pk (iiRcosmoSigma):
	'''grab KSmap, create 1, 1.8, 3.5, 5.7 peaks
	each time create 1000 realizations for specifice i, cosmo, sigmaG
	'''
	i, R, cosmo, sigmaG = iiRcosmoSigma
	kmap = KSmap_single(i, R, cosmo, sigmaG)
	idx = int(where(sigmaG_arr==sigmaG)[0])
	mask_all = mask_all_arr[idx]
	mask_bad = mask_bad_arr[idx]
	pspk_all = WLanalysis.peaks_mask_hist(kmap, mask_all, bins, kmin = kmin, kmax = kmax)
	pspk_bad = WLanalysis.peaks_mask_hist(kmap, mask_bad, bins, kmin = kmin, kmax = kmax)
	return pspk_all, pspk_bad

###############################################################
######## operations ###########################################
###############################################################
pool = MPIPool()

####### (0)test corrupted SIM file ###########
#def test_corrupte (iRcosmo):
	#cosmo, R = iRcosmo
	#if WLanalysis.TestFitsComplete(SIMfn(i, cosmo, R))==False:
		#print SIMfn(i, cosmo, R)
		#return 1
	#else:
		#return 0
#Rcosmo = [[ cosmo, R] for R in R_arr for cosmo in cosmo_arr]
#badfiles = array(pool.map(test_corrupte, Rcosmo))
#save(KS_dir+'badfiles.npy',badfiles)

######################################################
### (1)create KS map, uncomment next 4 lines #########
######################################################
#iRcosmo = [[i, R, cosmo] for R in R_arr for cosmo in cosmo_arr]
#pool.map(KSmap_massproduce, iRcosmo)

### (cov 1) this block is for covariance cosmology 
###cosmo='WL-only_cfhtcov-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800'
###iRcosmo = [[i, R, cosmo] for i in range(1,14)[::-1] for R in R_arr]# for cosmo in cosmo_arr]

######################################################
### (2) power spectrum for 0.5 smoothing scale only###
######################################################
for cosmo in cosmo_arr:
	ps_all_fn = powspec_sum_sf_fn(cosmo, 0.5, i, 'ALL')
	ps_pass_fn = powspec_sum_sf_fn(cosmo, 0.5, i, 'PASS')
	if os.path.isfile(ps_all_fn)==False or os.path.isfile(ps_pass_fn)==False:
		print 'ps',cosmo
		iRcosmoSigma = [[i, R, cosmo, 0.5] for R in R_arr]
		ps_arr = array(pool.map(create_ps, iRcosmoSigma))
		#ps_arr.shape = [1000, 2, 39]
		save(ps_all_fn, ps_arr[:,0,:])
		save(ps_pass_fn, ps_arr[:,1,:])
	else:
		print 'already exist - ps',cosmo


######################################################
##### (3) peak counts for 4 smoothing ################
######################################################

for cosmo in cosmo_arr:	
	for sigmaG in sigmaG_arr[1:-1]:
		pk_all_fn = peaks_sum_sf_fn(cosmo, sigmaG, i, 'ALL')
		pk_pass_fn = peaks_sum_sf_fn(cosmo, sigmaG, i, 'PASS')
		if os.path.isfile(pk_all_fn)==False or os.path.isfile(pk_pass_fn)==False:
			print 'pk',cosmo
			iRcosmoSigma = [[i, R, cosmo, sigmaG] for R in R_arr]
			pk_arr = array(pool.map(create_pk, iRcosmoSigma))
			#pk_arr.shape = [1000, 2, 25]
			save(pk_all_fn, pk_arr[:,0,:])
			save(pk_pass_fn, pk_arr[:,1,:])
		else:
			print 'already exist - pk',cosmo

###############################################################
### (4)sum over 13 sf for peaks and powspectrum
### !!!will only work if the previous step is done!!!
###############################################################

for cosmo in cosmo_arr:
	# power spectrum only has 0.5 arcmin
	psAll_gen = lambda i: np.load(powspec_sum_sf_fn(cosmo, 0.5, i, 'ALL'))
	psPass_gen = lambda i: np.load(powspec_sum_sf_fn(cosmo, 0.5, i, 'PASS'))
	sum_ps_all = sum(array(map(psAll_gen, i_arr)), axis=0)
	sum_ps_pass = sum(array(map(psPass_gen, i_arr)), axis=0)
	save(powspec_sum_fn(cosmo, 0.5, 'ALL'), sum_ps_all)
	save(powspec_sum_fn(cosmo, 0.5, 'PASS'), sum_ps_pass)

	for sigmaG in sigmaG_arr[1:-1]:
		pkAll_gen = lambda i: peaks_sum_sf_fn(cosmo, sigmaG, i, 'ALL')
		pkPass_gen = lambda i: peaks_sum_sf_fn(cosmo, sigmaG, i, 'PASS')
		sum_pk_all = sum(array(map(pkAll_gen, i_arr)), axis=0)
		sum_pk_pass = sum(array(map(pkPass_gen, i_arr)), axis=0)
		save(peaks_sum_fn(cosmo, sigmaG, 'ALL'), sum_pk_all)
		save(peaks_sum_fn(cosmo, sigmaG, 'PASS'), sum_pk_pass)

print 'DONE SUM SUM SUM'
################################################################
## (5)average over 1000 realizations ###########################
################################################################

for BG in ('ALL', 'PASS'):
	print BG
	all_ps = array([[powspec_sum_fn(cosmo, 0.5, BG)] for cosmo in cosmo_arr]).squeeze()
	# array size (91, 1000, 50)
	avg_ps = mean(all_ps, axis=1)
	save(KS_dir+'%s_ps_avg.npy'%(BG), avg_ps)
	for sigmaG in sigmaG_arr:
		all_pk = array([[peaks_sum_fn(cosmo, sigmaG, BG)] for cosmo in cosmo_arr]).squeeze()
		avg_pk = mean(all_pk, axis=1)
		save(KS_dir+'%s_pk_avg_sigmaG%02d.npy'%(BG, sigmaG*10), avg_ps)
	
pool.close()
print 'DONE DONE DONE'