#! python
# ibrun python stampede_kSZxCFHT.py Wx freq
# Wx goes from 1..4, freq goes from 0..2
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
freq_arr = ['2freqs', '545217GHzclean', '857GHz', 'dusty']
freq = freq_arr[int(sys.argv[1])]
print 'frequency:', freq

#prefix = 'filterAfterSQ'
#kSZmapGen = lambda Wx: np.load(kSZ_dir+'Planck/LGMCA_W%s_flipper8192_kSZfilt_squared_T2filt_toJia.npy'%(Wx))
#kSZmapGen = lambda Wx: np.load('/Users/jia/CFHTLenS/kSZ/filterAfterSQ/LGMCA_W%s_flipper8192_kSZfilt_squared_T2filt_toJia.npy'%(Wx))

prefix = 'filterB4SQ'

dustGen = lambda i, freq: np.load(kSZ_dir + 'dust/map%s_LGMCAfilt_uK_W%i_flipper8192_toJia.npy'%(freq, i))

nosqkSZGen_dusty = lambda Wx: np.load(kSZ_dir+'Planck/LGMCA_W%s_flipper8192_kSZfilt_NOTsquared_toJia.npy'%(Wx))

def return_alpha (freq): 
	if freq == '545217GHzclean':
		alpha = -0.0045
	elif freq == '857GHz':
		alpha = -8e-5
	return alpha

def kSZGen_clean(Wx, freq = freq):
	'''This routine cleans the kSZ map by applying some alpha value
	Note that if freq = False, then return (kSZ_freq1*kSZ_freq2)
	'''
	kSZ_NSQ = nosqkSZGen_dusty(Wx)
	if freq == '2freqs':
		dust1 = dustGen(Wx, '545217GHzclean')
		dust2 = dustGen(Wx, '857GHz')
		alpha1 = return_alpha('545217GHzclean')
		alpha2 = return_alpha('857GHz')
		kSZ_NSQ_clean1 = (1+alpha1)*kSZ_NSQ-alpha1*dust1
		kSZ_NSQ_clean2 = (1+alpha2)*kSZ_NSQ-alpha2*dust2
		kSZ_NSQ_clean = kSZ_NSQ_clean1*kSZ_NSQ_clean2
	else:
		dust = dustGen(Wx, freq)
		alpha = return_alpha(freq)
		nosqkSZ_NSQ_clean = (1+alpha)*kSZ_NSQ-alpha*dust
		kSZ_NSQ_clean = nosqkSZ_NSQ_clean**2
	return kSZ_NSQ_clean

def kSZmapGen(Wx, freq = freq):#clean dust
	'''this returns a cleaned kSZ map, 
	if freq='2freqs', return kSZ_freq1*kSZ_freq2'''	
	if freq == 'dusty':
		return nosqkSZGen_dusty(Wx)**2
	else:
		return kSZGen_clean(Wx, freq=freq)


create_noise_KS = 0
cross_correlate_kSZ_noise = 1
fmask2_arr = [0.65790059649362265, 0.55660343674246793, 0.56069976969877666, 0.4024946100277122]
#####################################################
###################### constants ####################
#####################################################
sizes = (1330, 800, 1120, 950)
PPR512=8468.416479647716
PPA512=2.4633625
edgesGen = lambda Wx: linspace(5,75,7)*sizes[Wx-1]/1330.0

maskGen = lambda Wx: np.load(kSZ_dir+'mask/W%i_mask.npy'%(Wx))

bmapGen = lambda Wx, iseed: np.load(kSZ_dir+'CFHT/Noise/W%i_Noise_sigmaG10_%04d.npy'%(Wx, iseed))

kmapGen = lambda Wx: WLanalysis.readFits(kSZ_dir+'CFHT/conv/W%i_KS_1.3_lo_sigmaG10.fit'%(Wx))
#####################################################
############### operations ##########################
#####################################################
if create_noise_KS:
	'''
	create 500 noise maps by randomly rotate galaxies, also include weights and (1+m) correction. all maps are smoothed over 1 arcmin.
	'''
	#Wx=int(sys.argv[1])
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
		bmap = bmapGen(Wx, iseed)*mask_arr[Wx-1]
		kSZmap = masked_kSZ_arr[Wx-1]
		edges = edgesGen(Wx)
		
		### set mean to 0
		bmap -= mean(bmap)
		kSZmap -= mean(kSZmap)
		
		ell_arr, CC = WLanalysis.CrossCorrelate (bmap, kSZmap,edges=edges)
		CC = CC/fmask2_arr[Wx-1]
		return CC
	Wx_iseed_list = [[Wx, iseed] for Wx in range(1,5) for iseed in range(500)]
	
	p = MPIPool()
	CC_arr = array(p.map(kSZxNoise, Wx_iseed_list))
	for Wx in arange(1,5):
		np.save(kSZ_dir+'%s/convxkSZ_500sim_W%s_%s.npy'%(prefix, Wx,freq), CC_arr[(Wx-1)*500:Wx*500])
	print 'done cross correlate kSZ x 500 noise.'
	
	############# cross with CFHT ####################
	for Wx in range(1,5):
		kmap = kmapGen(Wx)*mask_arr[Wx-1]
		kSZmap = masked_kSZ_arr[Wx-1]
		
		## set mean to 0
		kmap -= mean(kmap)
		kSZmap -= mean(kSZmap)
		
		edges = edgesGen(Wx)
		CC_signal = WLanalysis.CrossCorrelate(kmap, kSZmap,edges=edges)[1]/fmask2_arr[Wx-1]
		np.save(kSZ_dir+'%s/convxkSZ_W%s_%s.npy'%(prefix, Wx,freq), CC_signal)

print 'done!done!done!done!'