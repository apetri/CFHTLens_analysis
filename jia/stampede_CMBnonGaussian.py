### python code to compute non-Gaussianity of CMB lensing maps
### Jia Liu 2015/06/12
### works on Stampede
### 2015/06/30: update, write function that takes one directory, 
### for each file inside that dir, do:
### (1) compute ps
### (2) generate a GRF from that ps - maybe generate from avg ps instead
### (3) compute PDF for file and GRF, for 5 smoothings
### (4) compute peaks for file and GRF, for 5 smoothings

import WLanalysis
import glob, os, sys
import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack
from emcee.utils import MPIPool

from scipy import interpolate
import random

doGRF=int(sys.argv[2])
start=int(sys.argv[3])

print 'GRF=',doGRF

CMBlensing_dir = '/work/02977/jialiu/CMBnonGaussian/'
#CMBlensing_dir ='/Users/jia/weaklensing/CMBnonGaussian/'
############ for 31 cosmos #########
iii = int(sys.argv[1])
cosmo_arr = genfromtxt(CMBlensing_dir+'cosmo_arr.txt',dtype='string')
cosmo = cosmo_arr[iii]
#kmapGen = lambda r: WLanalysis.readFits('/work/02977/jialiu/CMBL_maps_46cosmo/%s/WLconv_z1100.00_%04dr.fits'%(cosmo, r))
kmapGen = lambda r: WLanalysis.readFits('/scratch/02977/jialiu/CMB_hopper/CMB_batch_storage/%s/1024b600/Maps/WLconv_z1100.00_%04dr.fits'%(cosmo, r))

####################################

########### fiducial cosmos #########
#cosmo='Om0.260_Ol0.740_Ob0.046_w-1.000_ns0.960_si0.800'
#kmapGen = lambda r: WLanalysis.readFits(b600_dir+'WLconv_z38.00_%04dr.fits'%(r))
#####################################


ends = [0.5, 0.22, 0.18, 0.1, 0.08]
PDFbin_arr = [linspace(-end, end, 101) for end in ends]
kmap_stds = [0.06, 0.05, 0.04, 0.03, 0.02] #[0.014, 0.011, 0.009, 0.006, 0.005]
peak_bins_arr = [linspace(-3*istd, 6*istd, 26) for istd in kmap_stds]

sizedeg = 3.5**2
PPA = 2048.0/(sqrt(sizedeg)*60.0) #pixels per arcmin
sigmaG_arr = array([0.5, 1.0, 2.0, 5.0, 8.0])
sigmaP_arr = sigmaG_arr*PPA #smoothing scale in pixels

#b600_dir =  '/work/02918/apetri/kappaCMB/Om0.260_Ol0.740_Ob0.046_w-1.000_ns0.960_si0.800/1024b600/Maps/'
#Pixels on a side: 2048
#Pixel size: 6.15234375 arcsec
#Total angular size: 3.5 deg
#lmin=1.0e+02 ; lmax=1.5e+05


def PDFGen (kmap, PDF_bins):
	all_kappa = kmap[~isnan(kmap)]
	PDF = histogram(all_kappa, bins=PDF_bins)[0]
	PDF_normed = PDF/float(len(all_kappa))
	return PDF_normed

def peaksGen (kmap, peak_bins):
	peaks = WLanalysis.peaks_list(kmap)
	peaks_hist = histogram(peaks, bins=peak_bins)[0]
	return peaks_hist

def compute_GRF_PDF_ps_pk (r):
	'''for a convergence map with filename fn, compute the PDF and the power spectrum. sizedeg = 3.5**2, or 1.7**2'''
	print cosmo, r
	kmap = kmapGen(r)
	#kmap = load(CMBlensing_dir+'GRF_fidu/'+'GRF_fidu_%04dr.npy'%(r))
	
	i_arr = arange(len(sigmaP_arr))
	
	if not doGRF:
            kmap_smoothed = [WLanalysis.smooth(kmap, sigmaP) for sigmaP in sigmaP_arr]
            ps = WLanalysis.PowerSpectrum(kmap_smoothed[0])[1]

            PDF = [PDFGen(kmap_smoothed[i], PDFbin_arr[i]) for i in i_arr]
            peaks = [peaksGen(kmap_smoothed[i], peak_bins_arr[i]) for i in i_arr]

	###### generate GRF
	else:
            ps=0
            random.seed(r)
            GRF = (WLanalysis.GRF_Gen(kmap)).newGRF()
            #save(CMBlensing_dir+'GRF_fidu/'+'GRF_fidu_%04dr.npy'%(r), GRF)		
            #GRF = load(CMBlensing_dir+'GRF_fidu/'+'GRF_fidu_%04dr.npy'%(r))
            GRF_smoothed = [WLanalysis.smooth(GRF, sigmaP) for sigmaP in sigmaP_arr]
            PDF = [PDFGen(GRF_smoothed[i], PDFbin_arr[i]) for i in i_arr]
            peaks = [peaksGen(GRF_smoothed[i], peak_bins_arr[i]) for i in i_arr]
	#############

	return [ps,], PDF, peaks#, PDF_GRF, peaks_GRF
		
pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)
a=pool.map(compute_GRF_PDF_ps_pk, range(start*1024+1, (start+1)*1024+1))
##save(CMBlensing_dir+'%s_PDF_pk_600b_GRF'%(cosmo), a)
if doGRF:
    save(CMBlensing_dir+'GRF_%s_ps_PDF_pk_z1100_%i.npy'%(cosmo), a, start)
else:
    save(CMBlensing_dir+'kappa_%s_ps_PDF_pk_z1100_%i.npy'%(cosmo), a, start)
pool.close()
print '---DONE---DONE---'



	