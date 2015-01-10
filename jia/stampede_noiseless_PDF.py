#!python
import WLanalysis
import numpy as np
from scipy import *


PPA512 = 2.4633625
edges = linspace(-0.05,0.15,51)

KSgen = lambda r: WLanalysis.readFits('/home1/02977/jialiu/cov_cat/emulator_subfield1_WL-only_cfhtcov-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800_4096xy_%04dr.fit'%(r))

y, x, e1, e2, w, m =  WLanalysis.readFits('/home1/02977/jialiu/KSsim/yxewm_subfield1_zcut0213.fit').T
mask =  WLanalysis.readFits('/home1/02977/jialiu/KSsim/mask/CFHT_mask_ngal5_sigma10_subfield01.fits')

def createPDF(r):
	print r
	kappa = KSgen(r).T[0]
	kmap, galn = WLanalysis.coords2grid(x, y, array([k,])
	kmap_smooth = WLanalysis.weighted_smooth(kmap, galn, sigmaG=1.0)
	all_kappa = kmap_smooth[where(mask>0)]
	all_kappa -= mean(all_kappa)
	PDF = histogram(all_kappa, bins=edges)[0]
	PDF /= float(len(all_kapp))
	return PDF

from emcee.utils import MPIPool
pool = MPIPool()
PDF_arr = pool.map(createPDF, arange(1,1001))