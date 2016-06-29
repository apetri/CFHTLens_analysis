import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack
import pickle
import WLanalysis

noiseless=1#0=noisy
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

if noiseless:
    pool=MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    a=pool.map(PDFGen, range(1,1024))
else:
    a=map(PDFGen, range(1,1024))

save(CMBlensing_dir+'PDF0.04_40arcmin_fidu_%s_%s.npy'%(['noisy','noiseless'][noiseless],['kappa','GRF'][Gaus]), a)
if noiseless:
    pool.close()
print 'DONE DONE'