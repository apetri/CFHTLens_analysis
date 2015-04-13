#!python
import WLanalysis
import numpy as np
from scipy import *
from emcee.utils import MPIPool
import sys, os

sigmaG = 1.0
PPA512 = 2.4633625
edges = linspace(-0.05,0.15,51)
sim_dir = '/home1/02977/jialiu/cat/'
cosmo_arr = os.listdir(sim_dir)

KS_dir = '/work/02977/jialiu/KSsim_noiseless/'

kmapGen = lambda i, cosmo, R: np.load(KS_dir+'%s/subfield%i/sigma%02d/SIM_KS_sigma%02d_subfield%i_%s_%04dr.npy'%(cosmo, i, sigmaG*10, sigmaG*10, i, cosmo,R))

maskGen = lambda i: WLanalysis.readFits(backup_dir+'mask/CFHT_mask_ngal5_sigma%02d_subfield%02d.fits'%(sigmaG*10, i))

mask_arr = array(map(maskGen, range(1,14)))

def createPDF(r):
	print sf, r
	kappa = KSgen(r).T[0]
	kmap, galn = WLanalysis.coords2grid(x, y, array([kappa,]))
	kmap_smooth = WLanalysis.weighted_smooth(kmap, galn, sigmaG=1.0)
	all_kappa = kmap_smooth[where(mask>0)]
	all_kappa -= mean(all_kappa)
	PDF = histogram(all_kappa, bins=edges)[0]
	PDF_normed = PDF/float(len(all_kappa))
	return PDF_normed

fn = lambda sf: '/home1/02977/jialiu/KSsim/PDF_noiseless/PDF_noiseless_50bins_sf%i_%s.npy'%(sf, cosmo)
if not os.path.isfile(fn(sf)):
	pool = MPIPool()
	PDF_arr = array(pool.map(createPDF, arange(1,1001)))
	np.save(fn(sf), PDF_arr)

PDF_reshaped = array([load(fn(sf)) for sf in arange(1,14)])
fsky_all = array([0.839298248291,0.865875244141,0.809467315674,
		  0.864688873291,0.679264068604,0.756385803223,
		  0.765892028809,0.747268676758,0.77250289917,
		  0.761451721191,0.691867828369,0.711254119873,
		  0.745429992676])

PDF_sum = sum(fsky_all.reshape(-1,1,1)*PDF_reshaped,axis=0)/sum(fsky_all)

save('PDF_noiseless_%s.npy'%(cosmo), PDF_sum)
print 'done done done'
