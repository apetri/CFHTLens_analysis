#!python
import WLanalysis
import numpy as np
from scipy import *
from emcee.utils import MPIPool
import sys
sf = int(sys.argv[1])
PPA512 = 2.4633625
edges = linspace(-0.05,0.15,51)
#cosmo = 'emu1-512b240_Om0.483_Ol0.517_w-1.515_ns0.960_si0.680'
cosmo = 'cfhtcov-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800'

#KSgen = lambda r: WLanalysis.readFits('/home1/02977/jialiu/cov_cat/emulator_subfield%i_WL-only_cfhtcov-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800_4096xy_%04dr.fit'%(sf, r))

KSgen = lambda r, sf: WLanalysis.readFits('/scratch/02977/jialiu/cat/%s/emulator_subfield%i_WL-only_%s_4096xy_%04dr.fit'%(cosmo, sf, cosmo, r))

yxewm_arr = array([ WLanalysis.readFits('/home1/02977/jialiu/KSsim/yxewm_subfield%i_zcut0213.fit'%(sf)).T] for sf in range(1,14))

mask_arr =  array([WLanalysis.readFits('/home1/02977/jialiu/KSsim/mask/CFHT_mask_ngal5_sigma10_subfield%02d.fits'%(sf))] for sf in range(1,14))

def createPDF(sf, r):
	print sf, r
	y, x, e1, e2, w, m = yxewm_arr [Wx-1]
	kappa = KSgen(r, sf).T[0]
	kmap, galn = WLanalysis.coords2grid(x, y, array([kappa,]))
	kmap_smooth = WLanalysis.weighted_smooth(kmap, galn, sigmaG=1.0)
	all_kappa = kmap_smooth[where(mask_arr[sf-1]>0)]
	all_kappa -= mean(all_kappa)
	PDF = histogram(all_kappa, bins=edges)[0]
	PDF_normed = PDF/float(len(all_kappa))
	return PDF_normed

pool = MPIPool()
PDF_arr = pool.map(createPDF, [[sf, r] for sf in arange(1,14) for r in arange(1,1001)])
#np.save('/home1/02977/jialiu/KSsim/PDF_noiseless/PDF_noiseless_50bins_sf%i_%s.npy'%(sf, cosmo), PDF_arr)
PDF_reshaped = PDF_arr.reshape(13, 1000, 50)
fsky_all = array([0.839298248291,0.865875244141,0.809467315674,
		  0.864688873291,0.679264068604,0.756385803223,
		  0.765892028809,0.747268676758,0.77250289917,
		  0.761451721191,0.691867828369,0.711254119873,
		  0.745429992676])

PDF_sum = sum(fsky_all.reshape(-1,1,1)*all_PDF,axis=0)/sum(fsky_all)

save('PDF_noiseless_%s.npy'%(cosmo), PDF_sum)
print 'done done done'
