import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack
import pickle
import WLanalysis
import sys

noiseless=0#0=noisy
Gaus=0

cosmo='Om0.260_Ol0.740_Ob0.046_w-1.000_ns0.960_si0.800'
if noiseless:
    from emcee.utils import MPIPool
    CMBlensing_dir = '/work/02977/jialiu/CMBnonGaussian/'
    thetaG_pix=2048.0/(sqrt(12.25)*60.0)*40 # 40 arcmin
    def kmapGen(r):
        kmap = WLanalysis.readFits('/work/02977/jialiu/CMBL_maps_46cosmo/Om0.296_Ol0.704_w-1.000_si0.786/WLconv_z1100.00_%04dr.fits'%(r))
        kmap_smoothed = WLanalysis.smooth(kmap, thetaG_pix)
        return kmap_smoothed
else:
    CMBlensing_dir ='/Users/jia/weaklensing/CMBnonGaussian/'
    
    FTmapGen_Gaus = lambda r: pickle.load(open(CMBlensing_dir+'colin_noisy/kappaMapTT_Gauss_10000sims/kappaMap%03dTT_3.pkl'%(r)))
    
    FTmapGen_fidu = lambda r: pickle.load(open(CMBlensing_dir+'colin_noisy/kappaMapTT_10000sims/kappaMap%03dTT_3.pkl'%(r)))


    def kmapGen(r):
        if Gaus:
            a = FTmapGen_Gaus(r)
        else:
            a = FTmapGen_fidu(r)
        areal = real(fftpack.ifft2(a))
        inorm = (2*pi*3.5/360.0)/(77.0**2)
        areal /= inorm    
        kmap = areal
        return WLanalysis.smooth(kmap, 2.93*5)

PDF_bins = linspace(-0.04, 0.04, 101)
def PDFGen(r):
    print r
    kmap = kmapGen(r)
    all_kappa = kmap[~isnan(kmap)]
    PDF = histogram(all_kappa, bins=PDF_bins)[0]
    PDF_normed = PDF/float(len(all_kappa))
    return PDF_normed

#if noiseless:
    #pool=MPIPool()
    #if not pool.is_master():
        #pool.wait()
        #sys.exit(0)
    #a=pool.map(PDFGen, range(1,1024))
#else:
    #a=map(PDFGen, range(1,1024))

#save(CMBlensing_dir+'PDF0.04_40arcmin_fidu_%s_%s.npy'%(['noisy','noiseless'][noiseless],['kappa','GRF'][Gaus]), a)
#if noiseless:
    #pool.close()
#print 'DONE DONE'

PDFbincenter=WLanalysis.edge2center(PDF_bins)

PDF_noisy_GRF=load(CMBlensing_dir+'PDF0.04_40arcmin_fidu_noisy_GRF.npy')

PDF_noisy_kappa=load(CMBlensing_dir+'PDF0.04_40arcmin_fidu_noisy_kappa.npy')

PDF_noiseless_kappa=load(CMBlensing_dir+'PDF0.04_40arcmin_fidu_noiseless_kappa.npy')

N=float(PDF_noiseless_kappa.shape[0])
from pylab import *
f=figure()
ax=f.add_subplot(111)

ax.errorbar(PDFbincenter, mean(PDF_noisy_GRF,axis=0), std(PDF_noisy_GRF,axis=0)/sqrt(N),label='noisy GRF')

ax.errorbar(PDFbincenter, mean(PDF_noisy_kappa,axis=0), std(PDF_noisy_kappa,axis=0)/sqrt(N),label='noisy kappa')

ax.errorbar(PDFbincenter, mean(PDF_noiseless_kappa,axis=0), std(PDF_noiseless_kappa,axis=0)/sqrt(N),label='noiseless kappa')

legend(frameon=0, fontsize=12)
ax.set_xlabel('kappa')
ax.set_ylabel('PDF')
ax.set_yscale('log')
show()

#delta_kappa = (PDFbincenter[1]-PDFbincenter[0])
i=0
for iPDF in (mean(PDF_noisy_GRF,axis=0), mean(PDF_noisy_kappa,axis=0), mean(PDF_noiseless_kappa,axis=0)):
    #iPDF = iPDF/delta_kappa
    
    x0=sum(iPDF*(PDFbincenter-mean(PDFbincenter)))
    xx=sum(iPDF*(PDFbincenter-mean(PDFbincenter))**2)/sum(iPDF)
    print ['noisy_GRF','noisy_kappa','noiseless_kappa',][i], x0,xx
    i+=1



