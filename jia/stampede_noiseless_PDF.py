#!python
import WLanalysis
import numpy as np
from scipy import *
from emcee.utils import MPIPool

PPA512 = 2.4633625
edges = linspace(-0.05,0.15,51)
cosmo = 'emu1-512b240_Om0.483_Ol0.517_w-1.515_ns0.960_si0.680'
for Wx in range(2,14):
	#KSgen = lambda r: WLanalysis.readFits('/home1/02977/jialiu/cov_cat/emulator_subfield%i_WL-only_cfhtcov-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800_4096xy_%04dr.fit'%(Wx, r))
	KSgen = lambda r: WLanalysis.readFits('/scratch/02977/jialiu/cat/%s/emulator_subfield%i_WL-only_%s_4096xy_%04dr.fit'%(cosmo, Wx, cosmo, r))

	y, x, e1, e2, w, m =  WLanalysis.readFits('/home1/02977/jialiu/KSsim/yxewm_subfield%i_zcut0213.fit'%(Wx)).T
	mask =  WLanalysis.readFits('/home1/02977/jialiu/KSsim/mask/CFHT_mask_ngal5_sigma10_subfield%02d.fits'%(Wx))

	def createPDF(r):
		print Wx, r
		kappa = KSgen(r).T[0]
		kmap, galn = WLanalysis.coords2grid(x, y, array([kappa,]))
		kmap_smooth = WLanalysis.weighted_smooth(kmap, galn, sigmaG=1.0)
		all_kappa = kmap_smooth[where(mask>0)]
		all_kappa -= mean(all_kappa)
		PDF = histogram(all_kappa, bins=edges)[0]
		PDF_normed = PDF/float(len(all_kappa))
		return PDF_normed
	
	pool = MPIPool()
	PDF_arr = pool.map(createPDF, arange(1,1001))
	np.save('/home1/02977/jialiu/KSsim/PDF_noiseless/PDF_noiseless_50bins_sf%i_%s.npy'%(Wx, cosmo), PDF_arr)