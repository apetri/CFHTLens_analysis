#!python
#CMB lensing x CFHT

import numpy as np
from scipy import *
from pylab import *
import os, sys
import WLanalysis
from scipy import interpolate
from emcee.utils import MPIPool

try:
	year = int(sys.argv[1])
except Exception:
	year = 2015
print 'year', year

create_noise_KS = 0
cross_correlate_cmbl_noise = 0 # cross noise CFHT with real planck data
cross_correlate_plancknoise_CFHT = 1 # cross noise planck maps with real CFHT data

centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
sizes = (1330, 800, 1120, 950)
PPR512=8468.416479647716
PPA512=2.4633625
edgesGen = lambda Wx: linspace(1,50,6)*sizes[Wx-1]/1330.0#linspace(5,80,11)
#edgesGen = lambda Wx: logspace(0,log10(50),7)*sizes[Wx-1]/1330.0#linspace(5,80,11)
rad2pix=lambda x, size: around(size/2.0-0.5 + x*PPR512).astype(int)

#cmb_dir = '/Users/jia/Documents/weaklensing/cmblensing/'
cmb_dir = '/home1/02977/jialiu/cmblensing/'

############## for zcut = 0213 ################
#fmask2_arr = [0.69589653344034097, 0.56642933419641017, 0.5884087465608, 0.43618153901026568]
#W1, fsky=0.00146857, fsky2=0.00136595, fmask=0.74818035, fmask2=0.69589653
#W2, fsky=0.00047130, fsky2=0.00040226, fmask=0.66364528, fmask2=0.56642933
#W3, fsky=0.00090399, fsky2=0.00081903, fmask=0.64944117, fmask2=0.58840875
#W4, fsky=0.00049279, fsky2=0.00043682, fmask=0.49207419, fmask2=0.43618154

############# no z cut, planck 13+15, or 15 #################
fmask2_arr = array([0.71694393, 0.61341502, 0.61103857, 0.45879416])
#W1, fsky=0.00149613, fsky2=0.00140726, fmask=0.76221954, fmask2=0.71694393
#W2, fsky=0.00049488, fsky2=0.00043563, fmask=0.69684520, fmask2=0.61341502
#W3, fsky=0.00092528, fsky2=0.00085053, fmask=0.66473852, fmask2=0.61103857
#W4, fsky=0.00050892, fsky2=0.00045946, fmask=0.50817719, fmask2=0.45879416

#######################################################
########## map making #################################
#######################################################

#for fn in os.listdir(cmb_dir+'planck/'):
	#print fn
	#full_fn = cmb_dir+'planck/'+fn
	#data = genfromtxt(full_fn)
	#np.save(full_fn[:-4]_cat,data)

def list2coords(radeclist, Wx, offset=False):
	size=sizes[Wx-1]
	xy = zeros(shape = radeclist.shape)
	if offset:
		center = 0.5*(amin(radeclist,axis=0)+amax(radeclist, axis=0))
	else:
		center = centers[Wx-1]
	f_Wx = WLanalysis.gnom_fun(center)
	#xy = degrees(array(map(f_Wx,radeclist)))
	xy = array(map(f_Wx,radeclist))
	xy_pix = rad2pix(xy, size)
	return xy_pix

def interpGridpoints (xy, values, newxy, method='nearest'):
	newvalues = interpolate.griddata(xy, values, newxy, method=method)
	return newvalues

def cmblGen_fn (fn, offset=False, method='nearest'):
	'''put values to grid, similar to cmblGen, except take in the file name.
	'''
	
	Wx = int(fn[fn.index('W')+1])
	print 'Wx, fn:', Wx, fn
	size=sizes[Wx-1]
	#cmblCoord = load(fn[:-3]+'npy')
	cmblCoord = genfromtxt(fn)
	#cmblCoord = WLanalysis.readFits(fn)
	radeclist = cmblCoord[:,:-1]
	values = cmblCoord.T[-1]
	xy = list2coords(radeclist, Wx, offset=offset)
	X,Y=meshgrid(range(size),range(size))
	X=X.ravel()
	Y=Y.ravel()
	newxy=array([X,Y]).T
	newvalues = interpGridpoints (xy, values, newxy,method=method)
	cmblmap = zeros(shape=(size,size))
	cmblmap[Y,X]=newvalues	
	cmblmap[isnan(cmblmap)]=0.0
	if offset:
		cmblmap = cmblmap.T
	np.save(fn[:-3]+'npy', cmblmap)
	return cmblmap

def simGen(Wx, r):
	simfn = cmb_dir+'planck/sim/sim_%04d_kmap_CFHTLS_W%i.npy'%(r, Wx)
	#try:
	if os.path.isfile(simfn):
		return load(simfn)
	else:
		print 'smap not created', Wx, r
		smap = cmblGen_fn(simfn[:-3]+'txt', offset=False)
		return smap
	#except Exception:
		#print 'Wx, r error', Wx, r
		#return kmapGen(Wx)
#for fn in os.listdir(cmb_dir+'planck/'):	
	#if fn[-3:]=='txt':
		#print fn[:-3]+'npy'
		#full_fn = cmb_dir+'planck/'+fn
		#cmblGen_fn(full_fn, offset = False)

#def simfn_fcn(r): 
	#cmblGen_fn(cmb_dir+'planck/sim/sim_%04d_kmap_CFHTLS_W%i.txt'%(r, 1), offset=False)
#p = MPIPool()
#p.map(simfn_fcn, arange(32, 100))
##############################################
########### mapGens  #########################

############# z=0213 ##########
#kSZ_dir = '/home1/02977/jialiu/kSZ/'
#kmapGen = lambda i: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_KS_1.3_lo_sigmaG10.fit'%(i))
#galnGen = lambda i: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_galn_1.3_lo_sigmaG10.fit'%(i))
#bmapGen = lambda Wx, iseed: np.load(kSZ_dir+'CFHT/Noise/W%i_Noise_sigmaG10_%04d.npy'%(Wx, iseed))
#kmapGen = lambda Wx: WLanalysis.readFits(kSZ_dir+'CFHT/conv/W%i_KS_1.3_lo_sigmaG10.fit'%(Wx))

############## no z cut ##############
#ptsrc15Gen = lambda i: np.load(cmb_dir + 'planck/'+'kappamask_flipper2048_CFHTLS_W%i_map.npy'%(i))
#ptsrc13Gen = lambda i: np.load(cmb_dir + 'planck/'+'kappamask2013_flipper2048_CFHTLS_W%i.npy'%(i))
#galnGen = lambda Wx: load('/Users/jia/weaklensing/CFHTLenS/catalogue/Me_Mw_galn/W%i_galn_nocut.npy'%(Wx))

############# only need once on local #################
#def maskGen (Wx, sigma_pix=10):
	#galn = WLanalysis.smooth(galnGen(Wx),PPA512)
	#galn *= ptsrc13Gen(Wx)#*ptsrc15Gen(Wx)## add point source mask for cmbl
	#mask = zeros(shape=galn.shape)
	#mask[10:-10,10:-10] = 1 ## remove edge 10 pixels
	#idx = where(galn<0.5)
	#mask[idx] = 0
	#mask_smooth = WLanalysis.smooth(mask, sigma_pix)	
	######### print out fksy and fsky 2 ##########
	#sizedeg = (sizes[Wx-1]/512.0)**2*12.0
	#fsky = sum(mask_smooth)/sizes[Wx-1]**2*sizedeg/41253.0
	#fsky2 = sum(mask_smooth**2)/sizes[Wx-1]**2*sizedeg/41253.0
	#fmask = sum(mask_smooth)/sizes[Wx-1]**2
	#fmask2 = sum(mask_smooth**2)/sizes[Wx-1]**2
	#print 'W%i, fsky=%.8f, fsky2=%.8f, fmask=%.8f, fmask2=%.8f'%(Wx, fsky, fsky2, fmask,fmask2) 
	##############################################
	#return mask_smooth#fsky, fsky2#
#for Wx in range(1,5):
	#save(cmb_dir+'mask/W%i_mask13_noZcut.npy'%(Wx), maskGen(Wx)) 

#a = array([sum(maskGen(Wx)**2)/sizes[Wx-1]**2 for Wx in range(1,5)])

#######################################

kmapGen = lambda Wx: np.load(cmb_dir+'cfht/kmap_W%i_sigma10_noZcut.npy'%(Wx))

maskGen = lambda Wx: np.load(cmb_dir+'mask/W%i_mask1315_noZcut.npy'%(Wx))
#maskGen = lambda Wx: np.load(cmb_dir+'mask/W%i_mask13_noZcut.npy'%(Wx))
if year == 2015:
	cmblGen = lambda Wx: np.load(cmb_dir+'planck/dat_kmap_flipper2048_CFHTLS_W%i_map.npy'%(Wx))
elif year == 2013:
	cmblGen = lambda Wx: np.load(cmb_dir+'planck/COM_CompMap_Lensing_2048_R1.10_kappa_CFHTLS_W%i.npy'%(Wx))

bmapGen = lambda Wx, iseed: load(cmb_dir+'cfht/noise_nocut/W%i_Noise_sigmaG10_%04d.npy'%(Wx, iseed))

##########################################################
######### cross correlate ################################
##########################################################


if create_noise_KS:
	'''
	create 500 noise maps by randomly rotate galaxies, also include weights and (1+m) correction. all maps are smoothed over 1 arcmin.
	'''
	#Wx=int(sys.argv[1])
	p = MPIPool()	
	bmap_fn = lambda Wx, iseed: cmb_dir+'cfht/noise_nocut/W%i_Noise_sigmaG10_%04d.npy'%(Wx, iseed)

	Mexw = lambda Wx, txt: np.load(cmb_dir+'cfht/W%i_M%s_nocut.npy'%(Wx,txt))
	Me1_arr = [Mexw(Wx, 'e1w') for Wx in range(1,5)]
	Me2_arr = [Mexw(Wx, 'e2w') for Wx in range(1,5)]
	Mwm_arr = [Mexw(Wx, 'wm') for Wx in range(1,5)]
	def randmap (iseedWx):
		iseed, Wx = iseedWx
		Me1, Me2 = Me1_arr[Wx-1], Me2_arr[Wx-1]
		Mwm = Mwm_arr[Wx-1]
		Me1rnd, Me2rnd = WLanalysis.rndrot(Me1, Me2, iseed=iseed)
		Me1smooth = WLanalysis.weighted_smooth(Me1rnd, Mwm)
		Me2smooth = WLanalysis.weighted_smooth(Me2rnd, Mwm)
		kmap_rand = WLanalysis.KSvw(Me1smooth, Me2smooth)
		print Wx, iseed, kmap_rand.shape
		np.save(bmap_fn(Wx, iseed), kmap_rand)
	p.map(randmap, [[iseed, Wx] for iseed in arange(500) for Wx in range(1,5)])
	print 'done creating 500 noise KS maps'
	
if cross_correlate_cmbl_noise:
	cmblmap_arr = map(cmblGen, range(1,5))
	mask_arr = map(maskGen, range(1,5))
	masked_cmbl_arr = [cmblmap_arr[i]*mask_arr[i] for i in range(4)]
	def cmblxNoise(iinput):
		'''iinput = (Wx, iseed)
		return the cross power between cmbl and convergence maps, both with smoothed mask.
		'''
		Wx, iseed = iinput
		print 'cmblxNoise', Wx, iseed
		bmap = bmapGen(Wx, iseed)*mask_arr[Wx-1]
		cmblmap = masked_cmbl_arr[Wx-1]
		edges = edgesGen(Wx)
		
		### set mean to 0
		bmap -= mean(bmap)
		cmblmap -= mean(cmblmap)
		
		ell_arr, CC = WLanalysis.CrossCorrelate (bmap, cmblmap,edges=edges)
		CC = CC/fmask2_arr[Wx-1]
		return CC
	Wx_iseed_list = [[Wx, iseed] for Wx in range(1,5) for iseed in range(500)]
	
	p = MPIPool()
	CC_arr = array(p.map(cmblxNoise, Wx_iseed_list))
	for Wx in arange(1,5):
		np.save(cmb_dir+'CC_noZcut/CFHTxPlanck%04d_lensing_100sim_W%s_mask13.npy'%(year, Wx), CC_arr[(Wx-1)*500:Wx*500])
	print 'done cross correlate cmbl x 500 noise.'
	
	############# cross with CFHT ####################
	for Wx in range(1,5):
		kmap = kmapGen(Wx)*mask_arr[Wx-1]
		cmblmap = masked_cmbl_arr[Wx-1]
		
		## set mean to 0
		kmap -= mean(kmap)
		cmblmap -= mean(cmblmap)
		
		edges = edgesGen(Wx)
		CC_signal = WLanalysis.CrossCorrelate(kmap, cmblmap,edges=edges)[1]/fmask2_arr[Wx-1]
		np.save(cmb_dir+'CC_noZcut/CFHTxPlanck%04d_lensing_W%s_mask13.npy'%(year, Wx), CC_signal)

if cross_correlate_plancknoise_CFHT:
	kmap_arr = map(kmapGen, range(1,5))
	mask_arr = map(maskGen, range(1,5))
	masked_kmap_arr = [kmap_arr[i]*mask_arr[i] for i in range(4)]
	def cfhtxNoise(iinput):
		'''iinput = (Wx, iseed)
		return the cross power between cmbl and convergence maps, both with smoothed mask.
		'''
		Wx, iseed = iinput
		print 'cmblxNoise', Wx, iseed
		smap = simGen(Wx, iseed)*mask_arr[Wx-1]
		kmap = masked_kmap_arr[Wx-1]
		edges = edgesGen(Wx)
		
		### set mean to 0
		kmap -= mean(kmap)
		smap -= mean(smap)
		
		ell_arr, CC = WLanalysis.CrossCorrelate (smap, kmap,edges=edges)
		CC = CC/fmask2_arr[Wx-1]
		return CC
	Wx_iseed_list = [[Wx, iseed] for Wx in range(1,5)[::-1] for iseed in range(100)[::-1]]
	
	p = MPIPool()
	CC_arr = array(p.map(cfhtxNoise, Wx_iseed_list))
	for Wx in arange(1,5):
		np.save(cmb_dir+'CC_noZcut/CFHTxPlanck%04d_lensing_planck100sim_W%s_mask1315.npy'%(year, Wx), CC_arr[(Wx-1)*100:Wx*100])
	print 'done cross correlate cfht x 100 noise.'
print 'done!done!done!done!'