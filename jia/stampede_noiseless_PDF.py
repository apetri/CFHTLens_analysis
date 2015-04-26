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
backup_dir = '/work/02977/jialiu/backup/'
KS_dir = '/work/02977/jialiu/KSsim_noiseless/'
fn = lambda cosmo: '/work/02977/jialiu/kappaPDF/PDF_noiseless_%s.npy'%(cosmo)

#KS_dir = '/work/02977/jialiu/KSsim/'
#fn = lambda cosmo: '/work/02977/jialiu/kappaPDF/PDF_noisy_%s.npy'%(cosmo)
	
kmapGen = lambda i, cosmo, R: np.load(KS_dir+'%s/subfield%i/sigma%02d/SIM_KS_sigma%02d_subfield%i_%s_%04dr.npy'%(cosmo, i, sigmaG*10, sigmaG*10, i, cosmo,R))

maskGen = lambda i: WLanalysis.readFits(backup_dir+'mask/CFHT_mask_ngal5_sigma%02d_subfield%02d.fits'%(sigmaG*10, i))

mask_arr = array(map(maskGen, range(1,14)))
mask_arr[mask_arr==0] = nan

def PDFGen(cosmoR):
	cosmo, R = cosmoR
	print cosmo, R
	kmaps = array([(kmapGen(i, cosmo, R)*mask_arr[i-1]) for i in range(1,14)])
	all_kappa = kmaps[~isnan(kmaps)]
	PDF = histogram(all_kappa, bins=edges)[0]
	PDF_normed = PDF/float(len(all_kappa))
	return PDF_normed

pool = MPIPool()
for cosmo in cosmo_arr:
	print cosmo
	if not os.path.isfile(fn(cosmo)):
		cosmoR_arr = [(cosmo, R) for R in range(1,1001)]
		PDF_arr = array(pool.map(PDFGen, cosmoR_arr))
		np.save(fn(cosmo), PDF_arr)

#fsky_all = array([0.839298248291,0.865875244141,0.809467315674,
		  #0.864688873291,0.679264068604,0.756385803223,
		  #0.765892028809,0.747268676758,0.77250289917,
		  #0.761451721191,0.691867828369,0.711254119873,
		  #0.745429992676])

print 'done done done'
