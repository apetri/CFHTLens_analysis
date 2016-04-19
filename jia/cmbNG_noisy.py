import WLanalysis
import glob, os, sys
import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack
from scipy import interpolate
import random
import pickle


CMBlensing_dir ='/Users/jia/weaklensing/CMBnonGaussian/'

#sigmaG_arr = array([0.5, 1.0, 2.0, 5.0, 8.0])
PDFbins = linspace(-0.12, 0.12, 101)
peak_bins = linspace(-0.06,0.14,26)#linspace(-3*0.02,6*0.02,26)
sizedeg = 3.5**2

bins=25
#ell_arr2048=WLanalysis.PowerSpectrum(kmapGen(0), bins=50)[0]
#ell_arr77 = WLanalysis.PowerSpectrum(zeros(shape=(77,77)), bins=bins)[0]

FTmapGen_Gaus = lambda r: pickle.load(open(CMBlensing_dir+'kappaMapTT_10000sims/kappaMap%03dTT_3.pkl'%(r)))
#FTmapGen = lambda r: pickle.load(open(CMBlensing_dir+'kappaMapTT_Gauss_10000sims/kappaMap%03dTT_3.pkl'%(r)))
FTmapGen = lambda r: pickle.load(open(CMBlensing_dir+'reconMaps_Om0.406_Ol0.594_w-1.000_si0.847/kappaMap%04dTT_3.pkl'%(r)))

def FT_PowerSpectrum (r, bins=10, return_ell_arr=0, Gaus=0):
    if Gaus:
        a = FTmapGen_Gaus(r)
    else:
        a = FTmapGen(r)
    PS2D=np.abs(fftpack.fftshift(a))**2
    ell_arr,psd1D=WLanalysis.azimuthalAverage(PS2D, bins=bins)    
    if return_ell_arr:
        ell_arr = WLanalysis.edge2center(ell_arr)* 360./3.5
        return ell_arr
    else:
        return psd1D
    
def FT2real (r, Gaus=0):    
    if Gaus:
        a = FTmapGen_Gaus(r)
    else:
        a = FTmapGen(r)
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

def compute_GRF_PDF_ps_pk (r, Gaus=0):
    kmap = FT2real(r, Gaus=Gaus)
    ps = WLanalysis.PowerSpectrum(WLanalysis.smooth(kmap, 0.18), bins=bins)[1]#*2.0*pi/ell_arr**2
    kmapsmooth8 = WLanalysis.smooth(kmap, 2.93)
    PDF = PDFGen(kmapsmooth8, PDFbins)
    peaks = peaksGen(kmapsmooth8, peak_bins)
    return concatenate([ps, PDF, peaks])

#### noiseless TT
#all_stats = load(CMBlensing_dir+'pspkPDF_fidu_mat12.npy')#[:35]
#ps_all = all_stats[:,:50]
#PDF_all = all_stats[:, 450:550]
#peaks_all = all_stats[:, -25:]

#### noiseless GRF
#mat_GRF=load(CMBlensing_dir+'plot/PDF_pk_600b_GRF.npy')
#iPDF_GRF = array([mat_GRF[x][0][4] for x in range(1024)])
#ipeak_GRF = array([mat_GRF[x][1][4] for x in range(1024)])

### noisy TT
all_stats77 = array([compute_GRF_PDF_ps_pk(r,Gaus=0) for r in range(1000)])#1024
save(CMBlensing_dir+'Pkappa_gadget/noisy_z1100_stats77_kappa_Om0.406_Ol0.594_w-1.000_si0.847.npy', all_stats77)
#all_stats77 = load (CMBlensing_dir+'Pkappa_gadget/noisy_z1100_stats77_kappa.npy')
#ps_all77 = all_stats77[:,:bins]
#PDF_all77 = all_stats77[:, bins:bins+len(PDFbins)-1]
#peaks_all77 = all_stats77[:, bins+len(PDFbins)-1:]

### noisy GRF
#all_stats77_GRF = array([compute_GRF_PDF_ps_pk(r,Gaus=1) for r in range(1024)])
#save(CMBlensing_dir+'Pkappa_gadget/noisy_z1100_stats77_GRF.npy', all_stats77_GRF)
#all_stats77_GRF = load(CMBlensing_dir+'Pkappa_gadget/noisy_z1100_stats77_GRF.npy')
#ps_all77_GRF = all_stats77_GRF[:,:bins]
#PDF_all77_GRF = all_stats77_GRF[:, bins:bins+len(PDFbins)-1]
#peaks_all77_GRF = all_stats77_GRF[:, bins+len(PDFbins)-1:]


##### plots (1) noiseless: GRF vs kappa
##### (2) noiseless GRF vs noisy GRF
##### (3) noisy GRF vs kappa

#from pylab import *

#def plot_3comparisons(ell_arr2048, ell_arr77, ps_all, ps_all77, PDF_all, PDF_all77, peaks_all, peaks_all77, label0, label77, fn, title):
    #'''make a plot 3 panels [ps, pdf, peaks] top to bottom, comparing 2 sets of data.
    #'''
    #errscale=sqrt(12/30000.)
    #f=figure(figsize=(6,10))
    #ax=f.add_subplot(311)
    #ax.errorbar(ell_arr2048, mean(ps_all,axis=0), errscale*std(ps_all, axis=0),label=label0)
    #ax.errorbar(ell_arr77, mean(ps_all77,axis=0), errscale*std(ps_all77, axis=0),label=label77)

    #ax.set_xlabel(r'$\ell$',fontsize=20)
    ##ax.set_ylabel(r'$C \times 10^7$')
    #ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$', fontsize=20)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #ax.set_xlim(ell_arr77[0],ell_arr77[-1])
    ##ax.set_ylim(-1, 3)
    #ax.plot([0, 1e4],[0,0],'--')
    #leg=ax.legend(prop={'size':10},loc=0, fontsize=20)
    #leg.get_frame().set_visible(False)
            
    #ax2=f.add_subplot(312)
    #ax2.errorbar(WLanalysis.edge2center(PDFbins), mean(PDF_all,axis=0), errscale*std(PDF_all, axis=0))
    #ax2.errorbar(WLanalysis.edge2center(PDFbins), mean(PDF_all77,axis=0), errscale*std(PDF_all77, axis=0))

    #ax2.set_xlabel(r'$\kappa$', fontsize=20)
    #ax2.set_ylabel('PDF', fontsize=20)
    ##ax.set_xscale('log')
    #ax2.set_yscale('log')

    #ax3=f.add_subplot(313)
    #ax3.errorbar(WLanalysis.edge2center(peak_bins), mean(peaks_all,axis=0), errscale*std(peaks_all, axis=0))
    #ax3.errorbar(WLanalysis.edge2center(peak_bins), mean(peaks_all77,axis=0), errscale*std(peaks_all77, axis=0))

    #ax3.set_xlabel(r'$\kappa$', fontsize=20)
    #ax3.set_ylabel('N_peaks', fontsize=20)
    #ax3.set_yscale('log')
    ##show()
    
    ##covI_PDF = mat(cov(PDF_all, rowvar=0)*errscale**2).I
    ##covI_peak = mat(cov(peaks_all[:,2:-1], rowvar=0)*errscale**2).I#[:,2:-1]
    ##dN_PDF = mat(mean(PDF_all,axis=0)- mean(PDF_all77,axis=0))
    ##dN_pk =  mat((mean(peaks_all,axis=0)- mean(peaks_all77,axis=0))[2:-1])#[2:-1]
    ##chisq_PDF = dN_PDF*covI_PDF*dN_PDF.T
    ##chisq_peak = dN_pk*covI_peak*dN_pk.T
    ##print '%s SNR(PDF) = %.2f, SNR(peaks) = %.2f' % (fn, sqrt(chisq_PDF), sqrt(chisq_peak))
    
    #ax.set_title(title, fontsize=24)
    #plt.subplots_adjust(hspace=0.25,left=0.18, right=0.95, bottom=0.06, top=0.95)
    #savefig(CMBlensing_dir+'plot/%s.jpg'%(fn))
    #close()

#plot_3comparisons(ell_arr2048, ell_arr2048, ps_all, ps_all, PDF_all, iPDF_GRF, peaks_all, ipeak_GRF, 'noiseless TT', 'noiseless GRF', 'noiseless_TT_GRF','non-Gaussianity (noiseless)')

#plot_3comparisons(ell_arr2048, ell_arr77, ps_all, ps_all77_GRF, iPDF_GRF, PDF_all77_GRF, ipeak_GRF, peaks_all77_GRF, 'noiseless GRF', 'noisy GRF', 'noiseless_noisy_GRF','effect of reconstruction')

#plot_3comparisons(ell_arr77, ell_arr77, ps_all77, ps_all77_GRF, PDF_all77, PDF_all77_GRF, peaks_all77, peaks_all77_GRF, 'noisy TT', 'noisy GRF', 'noisy_TT_GRF','non-Gaussianity (noisy)')
