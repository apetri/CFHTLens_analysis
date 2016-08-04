import WLanalysis
from scipy import *
from scipy.stats import skew
from emcee.utils import MPIPool

pool=MPIPool()
kmapNL = lambda i: WLanalysis.readFits('/work/02977/jialiu/CMBL_maps_46cosmo/Om0.296_Ol0.704_w-1.000_si0.786/WLconv_z1100.00_%04dr.fits'%(i))
kmapNOISY = lambda i: load('/work/02977/jialiu/CMBL_maps_46cosmo/noisy/reconMaps_Om0.296_Ol0.704_w-1.000_si0.786/recon_Om0.296_Ol0.704_w-1.000_si0.786_r%04d.npy'%(i))

sigmaG_arr=array([0.5, 1.0, 2.0, 5.0, 8.0])
PPA_NL = 2048.0/(sqrt(12.25)*60.0)
PPA_NOISY = 77.0/(sqrt(12.25)*60.0)
def iskew (i):
    print i
    ikmap_NL = kmapNL(i)
    ikmap_NOISY = kmapNOISY(i)
    skewness_NL = [skew(WLanalysis.smooth(ikmap_NL, ismooth).flatten() ) for ismooth in sigmaG_arr*PPA_NL] 
    skewness_NOISY = [skew(WLanalysis.smooth(ikmap_NOISY, ismooth).flatten() ) for ismooth in sigmaG_arr*PPA_NOISY]
    return [skewness_NL, skewness_NOISY]

skew_arr = array(pool.map(iskew, range(1,1000)))
save('/work/02977/jialiu/CMBL_skewness.npy',skew_arr)
