#!python
# Jia Liu 2014/09/12
# What the code does: create noiseless mass maps for fiducial cosmology, and calculate the power spectrum
# Cluster: XSEDE Stampede

import WLanalysis
from emcee.utils import MPIPool
import os
import numpy as np
from scipy import *
import scipy.ndimage as snd
import sys
sigmaG = 0.5
PPA512 = 2.4633625
KS_dir = '/scratch/02977/jialiu/KSsim/'
sim_dir = '/home1/02977/jialiu/cat/'

kappaGen = lambda r: WLanalysis.readFits( sim_dir+'emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765/emulator_subfield1_WL-only_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_4096xy_%04dr.fit'%(r)).T#[0]
#k, s1, s2

y, x = WLanalysis.readFits(KS_dir+'yxewm_subfield1_zcut0213.fit').T[:2]

def kmapPs (r):
	print r
	k, s1, s2 = kappaGen(r)[:3]
	Ms1,Ms2, galn = WLanalysis.coords2grid(x, y, array([s1,s2 ]))
	s1_smooth = WLanalysis.weighted_smooth(Ms1, galn, PPA=PPA512, sigmaG=sigmaG)
	s2_smooth = WLanalysis.weighted_smooth(Ms1, galn, PPA=PPA512, sigmaG=sigmaG)
	kmap_smooth = WLanalysis.KSvw(s1_smooth, s2_smooth)
	ps = WLanalysis.PowerSpectrum(kmap,sizedeg=12.0)[-1]
	return ps

pool = MPIPool()
ps_mat = pool.map(kmapPs, range(1,1001))
WLanalysis.writeFits(ps_mat,KS_dir+'ps_mat_sf1_shear_noiseless.fit')
print 'Done'