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
from multiprocessing import Pool

########## define constants ############
i = int(sys.argv[1])

print 'start'
sim_dir = '/home1/02977/jialiu/cat/'
KS_dir = '/scratch/02977/jialiu/KSsim/'
cosmo_arr = os.listdir(sim_dir)
params = genfromtxt(KS_dir+'cosmo_params.txt')

kmin = -0.04 # lower bound of kappa bin = -2 SNR
kmax = 0.12 # higher bound of kappa bin = 6 SNR
bins = 600 # for peak counts
sigmaG_arr = (0.5, 1, 1.8, 3.5, 5.3, 8.9)
#i_arr = arange(1,14)
R_arr = arange(1,1001)
PPA512 = 2.4633625
i_arr = (i,)
# constants not used
# zmax = 1.3
# zmin = 0.2
# ngal_arcmin = 5.0
# ngal_cut = ngal_arcmin*(60**2*12)/512**2# = 0.82, cut = 5 / arcmin^2
#PPR512 = 8468.416479647716#pixels per radians
#rad2pix = lambda x: around(512/2.0-0.5 + x*PPR512).astype(int) #from radians to pixel location

SIMfn = lambda i, cosmo, R: sim_dir+'%s/emulator_subfield%i_WL-only_%s_4096xy_%04dr.fit'%(cosmo, i, cosmo, R)

KSfn = lambda i, cosmo, R, sigmaG: KS_dir+'%s/subfield%i/sigma%02d/SIM_KS_sigma%02d_subfield%i_%s_%04dr.fit'%(cosmo, i, sigmaG*10, sigmaG*10, i, cosmo,R)

Mask_fn = lambda i, sigmaG: KS_dir+'mask/CFHT_mask_ngal5_sigma%02d_subfield%02d.fits'%(sigmaG*10, i)

peaks_fn = lambda i, cosmo, sigmaG, bins, R: KS_dir+'peaks/%s/subfield%i/sigma%02d/SIM_peaks_sigma%02d_subfield%i_%s_%03dbins_%04dr.fit'%(cosmo, i, sigmaG*10, sigmaG*10, i, cosmo, bins, R)

powspec_fn = lambda i, cosmo, sigmaG, R: KS_dir+'powspec/%s/subfield%i/sigma%02d/SIM_powspec_sigma%02d_subfield%i_%s_%04dr.fit'%(cosmo, i, sigmaG*10, sigmaG*10, i, cosmo, R)


######### functions ######################

### read in MW and yxewm first
#Mw_fn = KS_dir+'SIM_Mw_subfield%i.fit'%(i) # same for all R
#Mw = WLanalysis.readFits(Mw_fn)

Mw_fcn = lambda i: WLanalysis.readFits(KS_dir+'SIM_Mw_subfield%i.fit'%(i))
#Mw_arr = map(Mw_fcn, i_arr) # Mw = w (1+m) in a grid

yxewm_fcn = lambda i: WLanalysis.readFits(KS_dir+'yxewm_subfield%i_zcut0213.fit'%(i))
#yxewm_arr = map(yxewm_fcn, i_arr)

## this is customized to one subfield at a time
#Mw = Mw_fcn(i)
#y, x, e1, e2, w, m = yxewm_fcn(i).T

print 'got yxewm_arr'
def fileGen(i, R, cosmo):
	'''
	Put catalogue to grid, with (1+m)w correction. Mw is already done.
	Input:
	i: subfield range from (1, 2..13)
	R: realization range from (1..1000)
	cosmo: one of the 100 cosmos
	Return:
	Me1 = e1*w
	Me2 = e2*w
	
	'''
	#y, x, e1, e2, w, m = yxewm_arr[i-1].T
	k, s1, s2 = (WLanalysis.readFits(SIMfn(i,cosmo,R)).T)[[0,1,2]]
	s1 *= (1+m)
	s2 *= (1+m)
	eint1, eint2 = WLanalysis.rndrot(e1, e2, iseed=R)#random rotation	
	## get reduced shear 
	## 06/26/2014 change to WL approximation, since reduced shear can 
	## blow up gamma when kappa ~= 1
	#e1red, e2red = WLanalysis.eobs_fun(s1, s2, k, eint1, eint2)
	e1red, e2red = s1+eint1, s2+eint2

	print 'coords2grid', i, R, cosmo
	A, galn = WLanalysis.coords2grid(x, y, array([k, e1red*w, e2red*w]))
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
		ps_fn = powspec_fn(i, cosmo, sigmaG, R)
		pk_fn = peaks_fn(i, cosmo, sigmaG, bins, R)
		if (WLanalysis.TestFitsComplete (ps_fn) & WLanalysis.TestFitsComplete (ps_fn)) == False:
			create_ps_pk = 1
			break
	if create_ps_pk:
		print 'creating KSmap i, R, cosmo', i, R, cosmo
		Me1, Me2 = fileGen(i, R, cosmo)
		#Mw = Mw_arr[i-1]
		for sigmaG in sigmaG_arr:
			ps_fn = powspec_fn(i, cosmo, sigmaG, R)
			pk_fn = peaks_fn(i, cosmo, sigmaG, bins, R)
			
			KS_fn = KSfn(i, cosmo, R, sigmaG)
			create_kmap = 1
			if os.path.isfile(KS_fn):
				try: # make sure it's not a broken file
					kmap = WLanalysis.readFits(KS_fn)
					create_kmap = 0
				except Exception:
					create_kmap = 1
					pass	
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
				print 'file exist', ps_fn
				pass
			
			mask = WLanalysis.readFits(Mask_fn(i, sigmaG))
			peaks_hist = WLanalysis.peaks_mask_hist(kmap, mask, bins, kmin = kmin, kmax = kmax)
			
			try:
				WLanalysis.writeFits(peaks_hist,pk_fn)
			except Exception:
				print 'file exist', pk_fn
				pass
	else:
		print 'already done KSmap i, R, cosmo', i, R, cosmo


# full set
#pool = MPIPool()
#iRcosmo = [[i, R, cosmo] for R in R_arr for cosmo in cosmo_arr]
#pool.map(KSmap, iRcosmo)
#pool.close()


### collect all the ps and pk single file to matrix
### ps is weighted over # galaxies, pk is sum of all subfield
### will only work if the previous step is done

peaks_sum_fn = lambda cosmo, sigmaG, bins: KS_dir+'peaks_sum/SIM_peaks_sigma%02d_%s_%03dbins.fit'%(sigmaG*10, cosmo, bins)

powspec_sum_fn = lambda cosmo, sigmaG: KS_dir+'powspec_sum/SIM_powspec_sigma%02d_%s.fit'%(sigmaG*10, cosmo)

galcount = array([342966,365597,322606,380838,
		  263748,317088,344887,309647,
		  333731,310101,273951,291234,
		  308864]).astype(float) # galaxy counts for subfields, prepare for weighte sum powspec
galcount /= sum(galcount)

#p = Pool(1000)
def gen_mat (i, cosmo, sigmaG, ispk = True):
	'''Generate a matrix of peaks or powspec, where rows are realizations, columns are bins.'''
	def get_pkps (R):
		if ispk:
			fn = peaks_fn (i, cosmo, sigmaG, bins, R)
		else: #ps
			fn = powspec_fn (i, cosmo, sigmaG, R)
		if WLanalysis.TestFitsComplete(fn):
			pkps = WLanalysis.readFits(fn)
		else: #fits file broken or not exist, create file
			KS_fn = KSfn(i, cosmo, R, sigmaG)
			kmap = WLanalysis.readFits(KS_fn)
			if ispk:
				mask = WLanalysis.readFits(Mask_fn(i, sigmaG))
				pkps = WLanalysis.peaks_mask_hist(kmap, mask, bins, kmin = kmin, kmax = kmax)
				
			else:
				pkps = WLanalysis.PowerSpectrum(kmap, sizedeg=12.0)[-1]

			WLanalysis.writeFits(pkps, fn)
		return pkps
	return get_pkps
	
def sum_matrix (cosmosigmaG):
	cosmo, sigmaG = cosmosigmaG
	print sigmaG, cosmo
	psfn = powspec_sum_fn(cosmo, sigmaG)
	pkfn = peaks_sum_fn(cosmo, sigmaG, bins)
	
	if WLanalysis.TestFitsComplete(psfn) == False:
		print 'gen', psfn
		powspec_mat = zeros(shape=(len(R_arr), 50))
		for i in range(1,14):
			print 'ps', i
			powspec_mat += galcount[i-1] * np.array(map(gen_mat(i, cosmo, sigmaG, ispk = False), R_arr))
		WLanalysis.writeFits(powspec_mat, psfn)
		
	if WLanalysis.TestFitsComplete(pkfn) == False:
		print 'gen', pkfn
		peaks_mat = zeros(shape=(len(R_arr), bins))
		for i in range(1,14):
			print 'pk', i
			peaks_mat += np.array(map(gen_mat(i, cosmo, sigmaG, ispk = True), R_arr))	
		WLanalysis.writeFits(peaks_mat, pkfn)	

cosmosigmaG_arr = [[cosmo, sigmaG] for cosmo in cosmo_arr for sigmaG in sigmaG_arr]
pool = MPIPool()
pool.map(sum_matrix, cosmosigmaG_arr)
pool.close()
print 'SUM-SUM-SUM'