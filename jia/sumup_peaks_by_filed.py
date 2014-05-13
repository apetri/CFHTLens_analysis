import WLanalysis
from emcee.utils import MPIPool
import os
import numpy as np
from scipy import *
import sys

zg_arr = ('pz','rz1','rz2')
bins_arr = arange(10, 110, 15)
sigmaG_arr = (0.5, 1, 1.8, 3.5, 5.3, 8.9)
i_arr=arange(1,14)
R_arr=arange(1,1001)

KSCFHT_dir = '/direct/astro+astronfs03/workarea/jia/CFHT/KSCFHT/'
KSsim_dir = '/direct/astro+astronfs03/workarea/jia/CFHT/KSsim/'
fidu='mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800'
hi_w='mQ3-512b240_Om0.260_Ol0.740_w-0.800_ns0.960_si0.800'
hi_s='mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.850'
hi_m='mQ3-512b240_Om0.290_Ol0.710_w-1.000_ns0.960_si0.800'
cosmo_arr=(fidu,hi_m,hi_w,hi_s)
Rtol=len(R_arr)

peaks_CFHT_fn = lambda i, sigmaG, bins: KSCFHT_dir+'CFHT_peaks_sigma%02d_subfield%02d_%03dbins.fits'%(sigmaG*10, i, bins)

powspec_CFHT_fn = lambda i, sigmaG: KSCFHT_dir+'CFHT_powspec_sigma%02d_subfield%02d.fits'%(sigmaG*10, i)

peaks_fn = lambda i, cosmo, Rtol, sigmaG, zg, bins: KSsim_dir+'peaks/SIM_peaks_sigma%02d_subfield%i_%s_%s_%04dR_%03dbins.fit'%(sigmaG*10, i, zg, cosmo, Rtol, bins)

peaks_sum_fn = lambda cosmo, Rtol, sigmaG, zg, bins: KSsim_dir+'peaks_sum13fields/SIM_peaks_sigma%02d_%s_%s_%04dR_%03dbins.fit'%(sigmaG*10, zg, cosmo, Rtol, bins)

for bins in bins_arr:
	for sigmaG in sigmaG_arr:
		print bins, sigmaG
		
		fn_CFHT = KSsim_dir+'peaks_sum13fields/CFHT_peaks_sigma%02d_%03dbins.fits'%(sigmaG*10, bins)
		CFHT_peaks = zeros(shape=(Rtol,bins))
		for i in i_arr:
			CFHT_peaks += WLanalysis.readFits(peaks_CFHT_fn(i, sigmaG, bins))
		try:
			WLanalysis.writeFits(CFHT_peaks, fn_CFHT)
		except Exception:
			print fn_CFHT,'already exist, but no worries'
			pass
				
		
		for zg in zg_arr:
			for cosmo in cosmo_arr:
				fn = peaks_sum_fn (cosmo, Rtol, sigmaG, zg, bins)
				peaks = zeros(shape=(Rtol,bins))
				for i in i_arr:
					peaks += WLanalysis.readFits(peaks_fn(i, cosmo, Rtol, sigmaG, zg, bins))
				try:
					WLanalysis.writeFits(peaks, fn)
				except Exception:
					fn, ' already exist, but no worries'
					pass
				