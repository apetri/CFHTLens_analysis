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
freq = 'dusty'

create_noise_KS = 0
cross_correlate_kSZ_noise = 0

#####################################################
###################### constants ####################
#####################################################
sizes = (1330, 800, 1120, 950)
PPR512=8468.416479647716
PPA512=2.4633625
edgesGen = lambda Wx: linspace(5,75,7)*sizes[Wx-1]/1330.0

maskGen = lambda Wx: np.load(kSZ_dir+'mask/W%i_mask.npy'%(Wx))
nosqkSZmapGen = lambda Wx: np.load(kSZ_dir+'Planck/LGMCA_W%s_flipper8192_kSZfilt_NOTsquared_toJia.npy'%(i))

kSZmapGen = nosqkSZmapGen(Wx)**2

bmapGen = lambda Wx, iseed: np.load( kSZ_dir+'CFHT/Noise/W%i_Noise_sigmaG10_%04d.npy'%(Wx, iseed))

#####################################################
############### operations ##########################
#####################################################
if create_noise_KS:
	'''
	create 500 noise maps by randomly rotate galaxies, also include weights and (1+m) correction. all maps are smoothed over 1 arcmin.
	'''
	Wx=int(sys.argv[1])
	p = MPIPool()
	
	bmap_fn = lambda Wx, iseed: kSZ_dir+'CFHT/Noise/W%i_Noise_sigmaG10_%04d.npy'%(Wx, iseed)
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

if cross_correlate_kSZ_noise:		
	kSZmap_arr = map(kSZmapGen, range(1,5))
	mask_arr = map(maskGen, range(1,5))
	masked_kSZ_arr = [kSZmap_arr[i]*mask_arr[i] for i in range(4)]
	def kSZxNoise(iinput):
		'''iinput = (Wx, iseed)
		return the cross power between kSZ and convergence maps, both with smoothed mask.
		'''
		Wx, iseed = iinput
		print 'kSZxNoise', Wx, iseed
		bmap = bmapGen(Wx, iseed)*mask_arr
		kSZmap = masked_kSZ_arr[Wx-1]
		edges = edgesGen(Wx)
		ell_arr, CC = WLanalysis.CrossCorrelate (bmap, kSZmap,edges=edges)
		return CC
	Wx_iseed_list = [[Wx, iseed] for Wx in range(1,5) for iseed in range(500)]
	
	p = MPIPool()
	CC_arr = array(p.map(kSZxNoise, Wx_iseed_list))
	for Wx in arange(1,5):
		np.save(kSZ_dir+'convxkSZ_500sim_W%s_%s.npy'%(Wx,freq), CC_arr[(Wx-1)*500:Wx*500])

