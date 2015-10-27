import WLanalysis
import glob, os, sys
import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack

from scipy import interpolate
import random

CMBlensing_dir ='/Users/jia/weaklensing/CMBnonGaussian/'

#kmapGen0 = lambda r: load(CMBlensing_dir+'noisy_maps_blake/sampleKappaMap0-34/sampleKappaMap%03d.npy'%(r))
kmapGen = lambda r: WLanalysis.smooth(kmapGen0(r), 3.2)

#sigmaG_arr = array([0.5, 1.0, 2.0, 5.0, 8.0])
PDFbins = linspace(-0.08, 0.08, 101)
peak_bins = linspace(-3*0.02,6*0.02,26)
sizedeg = 3.5**2

bins=25
ell_arr2048=WLanalysis.PowerSpectrum(kmapGen(0), bins=50)[0]
ell_arr77 = WLanalysis.PowerSpectrum(zeros(shape=(77,77)), bins=bins)[0]
##kmap0_original=WLanalysis.readFits(CMBlensing_dir+'test_maps/WLconv_z38.00_0001r.fits')
##kmap0_smooth10=WLanalysis.smooth(kmap0_original,100)
#kmap0_smooth10=load(CMBlensing_dir+'test_maps/kmap0_smooth10arcmin.npy')
import pickle
#FTmapGen = lambda r: pickle.load(open('/Users/jia/Desktop/kappaMap_TT_Jia/kappaMap%03dTT_3.pkl'%(r)))

FTmapGen = lambda r: pickle.load(open(CMBlensing_dir+'kappaOutput_gaus/kappaMap%03dTT_3.pkl'%(r)))

FTmapGen = lambda r: pickle.load(open(CMBlensing_dir+'kappaOutput_TT/kappaMap%03dTT_3.pkl'%(r)))


def FT_PowerSpectrum (r, bins=10, return_ell_arr=0):
	a=FTmapGen(r)
	PS2D=np.abs(fftpack.fftshift(a))**2
	ell_arr,psd1D=WLanalysis.azimuthalAverage(PS2D, bins=bins)	
	if return_ell_arr:
		ell_arr = WLanalysis.edge2center(ell_arr)* 360./3.5
		return ell_arr
	else:
		return psd1D

def FT2real (r):
	a=FTmapGen(r)
	areal = real(fftpack.ifft2(a))
	inorm = (2*pi*3.5/360.0)/(77.0**2)
	areal /= inorm
	#areal = WLanalysis.smooth(areal, 2.93)#8/((3.5*60)/77)
	return areal
	
def PDFGen (kmap, PDF_bins):
	all_kappa = kmap[~isnan(kmap)]
	PDF = histogram(all_kappa, bins=PDF_bins)[0]
	PDF_normed = PDF/float(len(all_kappa))
	return PDF_normed

def peaksGen (kmap, peak_bins):
	peaks = WLanalysis.peaks_list(kmap)
	peaks_hist = histogram(peaks, bins=peak_bins)[0]
	return peaks_hist

def compute_GRF_PDF_ps_pk (r, IFT=0):
	if IFT:
		kmap = FT2real(r)
	else:
		kmap = kmapGen(r)
	ps = WLanalysis.PowerSpectrum(WLanalysis.smooth(kmap, 0.18), bins=bins)[1]#*2.0*pi/ell_arr**2
	kmapsmooth8 = WLanalysis.smooth(kmap, 2.93)
	PDF = PDFGen(kmapsmooth8, PDFbins)
	peaks = peaksGen(kmapsmooth8, peak_bins)
	return concatenate([ps, PDF, peaks])

#pspkPDF_fidu_mat=load(CMBlensing_dir+'pspkPDF_fidu_mat12.npy')
all_stats = load(CMBlensing_dir+'pspkPDF_fidu_mat12.npy')[:35]
ps_all = all_stats[:,:50]
PDF_all = all_stats[:, 450:550]
peaks_all = all_stats[:, -25:]

all_stats77 = array([compute_GRF_PDF_ps_pk(r, IFT=1) for r in range(1024)])
ps_all77 = all_stats77[:,:bins]
PDF_all77 = all_stats77[:, bins:bins+len(PDFbins)-1]
peaks_all77 = all_stats77[:, bins+len(PDFbins)-1:]


#ps_all = array(map(FT_PowerSpectrum, range(35)))*1e7
#ell_arr77 = FT_PowerSpectrum(1, return_ell_arr=1)

from pylab import *
f=figure(figsize=(6,8))
ax=f.add_subplot(311)
ax.errorbar(ell_arr2048, mean(ps_all,axis=0), std(ps_all, axis=0),label='noiseless')
ax.errorbar(ell_arr77, mean(ps_all77,axis=0), std(ps_all77, axis=0),label='noisy')

ax.set_xlabel(r'$\ell$')
#ax.set_ylabel(r'$C \times 10^7$')
ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(ell_arr77[0],ell_arr77[-1])
#ax.set_ylim(-1, 3)
ax.plot([0, 1e4],[0,0],'--')
leg=ax.legend(prop={'size':10},loc=0)
leg.get_frame().set_visible(False)
		
ax2=f.add_subplot(312)
ax2.errorbar(WLanalysis.edge2center(PDFbins), mean(PDF_all,axis=0), std(PDF_all, axis=0))
ax2.errorbar(WLanalysis.edge2center(PDFbins), mean(PDF_all77,axis=0), std(PDF_all77, axis=0))

ax2.set_xlabel(r'$\kappa$')
ax2.set_ylabel('PDF')
ax.set_xscale('log')
ax.set_yscale('log')

ax3=f.add_subplot(313)
ax3.errorbar(WLanalysis.edge2center(peak_bins), mean(peaks_all,axis=0), std(peaks_all, axis=0))
ax3.errorbar(WLanalysis.edge2center(peak_bins), mean(peaks_all77,axis=0), std(peaks_all77, axis=0))

ax3.set_xlabel(r'$\kappa$')
ax3.set_ylabel('N_peaks')

show()
#savefig('/Users/jia/Desktop/test_noisy_noiseless_maps.jpg')
#close()
