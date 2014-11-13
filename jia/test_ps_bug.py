# Jia Liu 11/13/2014
# after found bus in power spectrum computation, this code is to test
# (1) how does this change average ps
# (2) dP/P for the signal, and pure noise
# (3) final goal is to find a best solution for fixing the paper

import numpy as np
from scipy import *
from scipy import interpolate#,stats
import os
import WLanalysis
#import matplotlib.pyplot as plt
#from pylab import *
import matplotlib.gridspec as gridspec 
import scipy.ndimage as snd
from emcee.utils import MPIPool
from scipy.fftpack import fftfreq, fftshift

def azimuthalAverage(image, center = None, edges = None, logbins = True, bins = 50, bug = False):
	"""
	Calculate the azimuthally averaged radial profile.
	Input:
	image = The 2D image
	center = The [x,y] pixel coordinates used as the center. The default is None, which then uses the center of the image (including fracitonal pixels).
	Output:
	ell_arr = the ell's, lower edge
	tbin = power spectrum
	"""
	# Calculate the indices from the image
	y, x = np.indices(image.shape)
	if not center:
		center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
	if image.shape[0]%2 == 0 and bug == True:## added oct/31/2014, since nyquist freqnecy is not centered for even # mapsize
		center+=0.5
	r = np.hypot(x - center[0], y - center[1])#distance to center pixel, for each pixel

	# Get sorted radii
	ind = np.argsort(r.flat)
	r_sorted = r.flat[ind] # the index to sort by r
	i_sorted = image.flat[ind] # the index of the images sorted by r

	# find index that's corresponding to the lower edge of each bin
	kmin=1.0
	kmax=image.shape[0]/2.0
	if edges == None:
		if logbins:
			edges = logspace(log10(kmin),log10(kmax),bins+1)
		else:
			#edges = linspace(kmin,kmax+0.001,bins+1)	
			edges = linspace(kmin,kmax,bins+1)
	if edges[0] > 0:
		edges = append([0],edges)
		
	hist_ind = np.histogram(r_sorted,bins = edges)[0] # hist_ind: the number in each ell bins, sum them up is the index of lower edge of each bin, first bin spans from 0 to left of first bin edge.	
	hist_sum = np.cumsum(hist_ind)
	csim = np.cumsum(i_sorted, dtype=float)
	tbin = csim[hist_sum[1:]] - csim[hist_sum[:-1]]
	radial_prof = tbin/hist_ind[1:]
	
	return edges[1:], radial_prof

edge2center = lambda x: x[:-1]+0.5*(x[1:]-x[:-1])

def PowerSpectrum(img, sizedeg = 12.0, edges = None, logbins = True, bug = False):#edges should be pixels
	'''Calculate the power spectrum for a square image, with normalization.
	Input:
	img = input square image in numpy array.
	sizedeg = image real size in deg^2
	edges = ell bin edges, length = nbin + 1, if not provided, then do 1000 bins.
	Output:
	powspec = the power at the bins
	ell_arr = lower bound of the binedges
	'''
	img = img.astype(float)
	size = img.shape[0]
	#F = fftpack.fftshift(fftpack.fft2(img))
	F = fftshift(fftpack.fft2(img))
	psd2D = np.abs(F)**2
	ell_arr, psd1D = azimuthalAverage(psd2D, center=None, edges = edges,logbins = logbins, bug = bug)
	ell_arr = edge2center(ell_arr)
	ell_arr *= 360./sqrt(sizedeg)# normalized to our current map size
	norm = ((2*pi*sqrt(sizedeg)/360.0)**2)/(size**2)**2
	powspec = ell_arr*(ell_arr+1)/(2*pi) * norm * psd1D
	return ell_arr, powspec

PPA512 = 2.4633625

cat_fcn = lambda r: (WLanalysis.readFits('/home1/02977/jialiu/cov_cat/emulator_subfield1_WL-only_cfhtcov-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800_4096xy_%04dr.fit'%(r)).T)[[0,1,2]]

y, x, e1, e2, w, m = WLanalysis.readFits('/scratch/02977/jialiu/KSsim/yxewm_subfield1_zcut0213.fit').T

mask = WLanalysis.readFits('/scratch/02977/jialiu/KSsim/mask/CFHT_mask_ngal5_sigma05_subfield01.fits')

def createKS (r=None, sigmaG = 0.5*PPA512):
	if r == None:
		eint1, eint2 = WLanalysis.rndrot(e1, e2)
	else:
		k, s1, s2 = cat_fcn(r)
		eint1, eint2 = s1, s2
	A, galn = WLanalysis.coords2grid(x, y, array([eint1, eint2]))
	Me1, Me2 = A
	Me1_smooth = WLanalysis.smooth(Me1, sigmaG)
	Me2_smooth = WLanalysis.smooth(Me2, sigmaG)
	kmap = WLanalysis.KSvw(Me1_smooth, Me2_smooth)
	kmap *= mask
	ps_bug = PowerSpectrum(kmap, bug = True)[-1]
	ps_correct = PowerSpectrum(kmap, bug = False)[-1]
	return ps_bug, ps_correct
	
pool = MPIPool()
#noise_ps = pool.map(createKS, [None,]*1000)
#signal_ps = pool.map(createKS, range(1,1001))

