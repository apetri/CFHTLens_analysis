#! python
# 2015/01/21
# this code computes kSZxCFHT cross correlation on stampede
# (1) create 500 noise kappa maps by randomly rotate galaxies
# (2) create pixelized kSZ maps from catalogue send to me by Colin H.
# (3) clean the dusts from kSZ maps
# (4) cross correlate kSZ x CFHT convergence maps
# (5) cross correlate kSZ x noise kappa maps

import numpy as np
from scipy import *
from pylab import *
import os, sys
import WLanalysis
from scipy import interpolate
from emcee.utils import MPIPool

kSZ_dir = '/home1/02977/jialiu/kSZ/'
bmap_fn = lambda Wx, iseed: kSZ_dir+'CFHT/Noise/W%i_Noise_sigmaG10_%04d.npy'%(Wx, iseed)

create_noise_KS = 1

#####################################################
############### create noise kappa maps #############
#####################################################
if create_noise_KS:
	Wx=int(sys.argv(1))
	p = MPIPool()
	Mexw = lambda Wx, txt: WLanalysis.readFits(kSZ_dir+'CFHT/Me_Mw_galn/W%i_M%s_1.3_lo.fit'%(Wx,txt))
	Me1, Me2, Mwm = Mexw(Wx, 'e1w'), Mexw(Wx, 'e2w'), Mexw(Wx, 'wm')
	def randmap (iseed, Wx=Wx):	
		Me1rnd, Me2rnd = WLanalysis.rndrot(Me1, Me2, iseed=iseed)
		Me1smooth = WLanalysis.weighted_smooth(Me1rnd, Mwm)
		Me2smooth = WLanalysis.weighted_smooth(Me2rnd, Mwm)
		kmap_rand = WLanalysis.KSvw(Me1smooth, Me2smooth)
		print Wx, iseed, kmap_rand.shape
		np.save(bmap_fn(Wx, iseed), kmap_rand)
	p.map(randmap, arange(500))
	print 'done creating 500 noise KS maps'

