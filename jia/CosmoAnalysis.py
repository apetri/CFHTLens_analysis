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
#import scipy.ndimage as snd

################ steps #####################
#1) count peaks, with mask
#2) power spectrum, with mask, but note need to use good fields (to be implemented)
#3) construct model, using rz1
#4) fit model to CFHT
#5) fit model to simulation pz, rz2

########## define constants ############
kmin = -0.04 # lower bound of kappa bin = -2 SNR
kmax = 0.12 # higher bound of kappa bin = 6 SNR
bins_arr = arange(20, 150, 10)
ngal_arcmin = 5.0
zmax=1.3
zmin=0.2

ngal_cut = ngal_arcmin*(60**2*12)/512**2# = 0.82, cut = 5 / arcmin^2
PPR512=8468.416479647716#pixels per radians
PPA512=2.4633625
sigmaG_arr = (0.5, 1, 1.8, 3.5, 5.3, 8.9)
zg_arr = ('pz','rz1','rz2')

full_dir = '/direct/astro+astronfs01/workarea/jia/CFHT/full_subfields/'
KSCFHT_dir = '/direct/astro+astronfs03/workarea/jia/CFHT/KSCFHT/'
KSsim_dir = '/direct/astro+astronfs03/workarea/jia/CFHT/KSsim/'
fidu='mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800'
hi_w='mQ3-512b240_Om0.260_Ol0.740_w-0.800_ns0.960_si0.800'
hi_s='mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.850'
hi_m='mQ3-512b240_Om0.290_Ol0.710_w-1.000_ns0.960_si0.800'

####### maps to process #########
i_arr=[1,2]
R_arr=arange(1,129)
cosmo_arr=(fidu,hi_m,hi_w,hi_s)
Rtol=len(R_arr)
####### end: define constant ####

########## functions ###########
KSCFHT_fn = lambda i, sigmaG: KSCFHT_dir+'CFHT_KS_sigma%02d_subfield%02d.fits'%(sigmaG*10, i)

Mask_fn = lambda i, sigmaG: KSCFHT_dir+'CFHT_mask_ngal5_sigma%02d_subfield%02d.fits'%(sigmaG*10, i)

KSsim_fn = lambda i, cosmo, R, sigmaG, zg: KSsim_dir+'%s/SIM_KS_sigma%02d_subfield%i_%s_%s_%04dr.fit'%(cosmo, sigmaG*10, i, zg, cosmo,R)#i=subfield, cosmo, R=realization, sigmaG=smoothing, zg=zgroup=(pz, rz, rz2)

# this is one matrix for Rtol realizations, where Rtol is the total number of realizations
peaks_fn = lambda i, cosmo, Rtol, sigmaG, zg, bins: KSsim_dir+'peaks/SIM_peaks_sigma%02d_subfield%i_%s_%s_%04dR_%03dbins.fit'%(sigmaG*10, i, zg, cosmo, Rtol, bins)

powspec_fn = lambda i, cosmo, Rtol, sigmaG, zg: KSsim_dir+'powspec/SIM_powspec_sigma%02d_subfield%i_%s_%s_%04dR.fit'%(sigmaG*10, i, zg, cosmo, Rtol)

def peaks_list (i, sigmaG, zg, bins, cosmo, kmin=kmin, kmax=kmax, ps=False):
	'''need test.'''
	def ipeaks_list (R):#, sigmaG, zg, bins):
		kmap = WLanalysis.readFits(KSsim_fn(i, cosmo, R, sigmaG, zg))
		if ps:#powspec
			ell_arr, powspec = WLanalysis.PowerSpectrum(kmap, sizedeg=12.0)
			return powspec
		else:#peaks
			mask = WLanalysis.readFits(Mask_fn(i, sigmaG))
			peaks_hist = WLanalysis.peaks_mask_hist(kmap, mask, bins, kmin=kmin, kmax=kmax)
			return peaks_hist
	return ipeaks_list

def peaksmat(i, cosmo, zg, Rtol, bins = False, sigmaG = False, R0 = 1):
	'''Return peak histogram, for bins in bins_arr.
	Input:
	i: subfield
	cosmo: one of the 4 cosmology
	Rtol: total number of realizations, and count peaks for realizations #(R0, R0+1, .. R0+Rtotl-1)
	R0: the first realization, if not starting from 1
	zg: = (pz, rz1, rz2), one of the 3 z group
	Return:
	If bins and sigmaG are not False, return a maxtrix of shape=(Rtol x bins)
	otherwise, returns nothing.
	'''
	if bins and sigmaG:
		# assume the mass run is already complete
		peaksfn = peaks_fn(i, cosmo, Rtol, sigmaG, zg, bins)
		if os.path.isfile(peaksfn):
			peaks_mat = WLanalysis.readFits(peaksfn)
			return peaks_mat
		else:
			print 'Need to run peaksmat first with (bins = False, sigmaG = False).'
	elif bins == False and sigmaG == False:
		for bins in bins_arr:
			for sigmaG in sigmaG_arr:
				peaksfn = peaks_fn(i, cosmo, Rtol, sigmaG, zg, bins)
				if os.path.isfile(peaksfn):
					continue
				else:
					map_fcn = peaks_list (i, sigmaG, zg, bins, cosmo)
					pool = MPIPool()
					peaks_mat = pool.map(map_fcn, R_arr)
					peaks_mat = array(peaks_mat)
					WLanalysis.writeFits(peaks_mat,peaksfn)


def psmat(i, cosmo, zg, Rtol, sigmaG = False, R0 = 1):#bins = False
	'''see peaksmat?'''
	if sigmaG:
		# assume the mass run is already complete
		psfn = powspec_fn(i, cosmo, Rtol, sigmaG, zg)
		if os.path.isfile(peaksfn):
			ps_mat = WLanalysis.readFits(psfn)
			return ps_mat
		else:
			print 'Need to run psmat first with (sigmaG = False).'
	elif sigmaG == False:
		for sigmaG in sigmaG_arr:
			psfn = powspec_fn(i, cosmo, Rtol, sigmaG, zg)
			if os.path.isfile(peaksfn):
				continue
			else:
				map_fcn = peaks_list (i, sigmaG, zg, 10, cosmo, ps=True)#bins=10, no real use
				pool = MPIPool()
				ps_mat = pool.map(map_fcn, R_arr)
				ps_mat = array(peaks_mat)
				WLanalysis.writeFits(ps_mat,psfn)	
	
for i in i_arr:
	for cosmo in cosmo_arr:
		for zg in zg_arr:
			peaksmat(i, cosmo, zg, Rtol, bins = False, sigmaG = False, R0 = 1)
			psmat(i, cosmo, zg, Rtol, sigmaG = False, R0 = 1)

sys.exit(0)