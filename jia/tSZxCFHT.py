##########################################################
### This code is for cross correlate CFHT with Planck tSZ. 
### It does the following:
### 1) put tSZ to a grid same as CFHT size
### 2) cross correlate CFHT with tSZ

import numpy as np
from scipy import *
from pylab import *
import os,sys
import WLanalysis
from scipy import interpolate

freq_arr = ['2freqs', '545217GHzclean', '857GHz', 'dusty']#
if int(sys.argv[1]):
	freq = freq_arr[int(sys.argv[1])]
else:
	freq = freq_arr[0]
print 'frequency:', freq

###################### knobs###########################
dusty = 0
plot_crosscorrelate_all = 1
create_noise_KS = 0
cross_cov_mat = 1
powspec_without_ells_factor = 0
clean_dust = 0
testCC = 0
test_powspec = 0

#plot_crosscorrelate_all_junk = 0
#######################################################

kSZ_dir = '/Users/jia/CFHTLenS/kSZ/newfil/'
#kSZ_dir = '/Users/jia/CFHTLenS/kSZ/ellmax3000/'
plot_dir = '/Users/jia/CFHTLenS/kSZ/plot/newfil/'
kSZCoordsGen = lambda i: WLanalysis.readFits(kSZ_dir+'LGMCA_W%i_flipper8192_kSZfilt_squared_toJia.fit'%(i))
#kSZCoordsGen = lambda i: WLanalysis.readFits(kSZ_dir+'kSZ2_W%i_ellmax3000.fit'%(i))
#noiseCoordsGen = lambda i: WLanalysis.readFits(kSZ_dir+'kSZ2_noise_W%i_ellmax3000.fit'%(i))
#offsetCoordsGen = lambda i: WLanalysis.readFits(kSZ_dir+'kSZ2_W%i_ellmax3000_offset.fit'%(i))

#CC_fcn = lambda Wx: kSZ_dir+'ptMask_Noise_convxkSZ_W%s.fit'%(Wx)# cross correlation file name for 500 sims
CC_fcn = lambda Wx, freq: kSZ_dir+'convxkSZ_500sim_W%s_%s'%(Wx,freq)
def return_alpha (freq): 
	if freq == '545217GHzclean':
		alpha = -0.0045
		#Power 6 bins larger steps -0.00093959731543624692
		#SNR 6bins -0.0024832214765100613#4bins -0.056124161073825493#6bins: -0.035928411633109614
	elif freq == '857GHz':
		alpha = -8e-5#
		#Power 6 bins larger steps -2.0134228187919292e-05
		#SNR 6 bins-0.00012080536912751662
		#4bins -0.00056963087248322131
		#6bins: -0.00035011185682326599
	return alpha

kmapGen = lambda i: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_KS_1.3_lo_sigmaG10.fit'%(i))
bmodeGen = lambda i: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_Bmode_1.3_lo_sigmaG05.fit'%(i))
galnGen = lambda i: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_galn_1.3_lo_sigmaG10.fit'%(i))

ptsrcGen = lambda i: np.load(kSZ_dir + 'null/'+'PSmaskRING_100-143-217-353-545_857_5sigma_Nside8192_BOOL_W%s_toJia.npy'%(i))

#offsetGen = lambda i: np.load(kSZ_dir + 'null/'+'LGMCA_W%soffset_flipper8192_kSZfilt_squared_toJia.npy'%(i))

offsetGen = lambda i: np.load(kSZ_dir + 'null/'+'LGMCA_W1offset_flipper8192_kSZfilt_squared_toJia.npy')[:sizes[i-1],:sizes[i-1]]

#def offsetGen(i):
	#if i==1:
		#offset = np.load(kSZ_dir + 'null/'+'LGMCA_W1offset_flipper8192_kSZfilt_squared_toJia.npy')[:sizes[i-1],:sizes[i-1]]
	#else:
		#offset = kSZmapGen(1)[:sizes[i-1],:sizes[i-1]]
	#return offset

noiseGen = lambda i: np.load(kSZ_dir + 'null/'+'LGMCA_noise_W%s_flipper8192_kSZfilt_squared_toJia.npy'%(i))
nosqkSZGen_dusty= lambda i: np.load(kSZ_dir + 'null/'+'LGMCA_W%s_flipper8192_kSZfilt_NOTsquared_toJia.npy'%(i))
dustGen = lambda i, freq: np.load(kSZ_dir + 'dust/'+'map%s_LGMCAfilt_uK_W%i_flipper8192_toJia.npy'%(freq, i))

def nosqkSZGen(Wx, freq = '2freqs'):#'857GHz'#
	'''This routine cleans the kSZ map by applying some alpha value
	Note that if freq = False, then return (kSZ_freq1*kSZ_freq2)
	'''
	kSZ_NSQ = nosqkSZGen_dusty(Wx)
	if freq in ['2freqs', 'dusty']:
		dust1 = dustGen(Wx, '545217GHzclean')
		dust2 = dustGen(Wx, '857GHz')
		alpha1 = return_alpha('545217GHzclean')
		alpha2 = return_alpha('857GHz')
		kSZ_NSQ_clean1 = (1+alpha1)*kSZ_NSQ[Wx-1]-alpha1*dust1
		kSZ_NSQ_clean2 = (1+alpha2)*kSZ_NSQ[Wx-1]-alpha2*dust2
		kSZ_NSQ_clean = kSZ_NSQ_clean1*kSZ_NSQ_clean2
	else:
		dust = dustGen(Wx, freq)
		alpha = return_alpha(freq)
		kSZ_NSQ_clean = (1+alpha)*kSZ_NSQ[Wx-1]-alpha*dust
	return kSZ_NSQ_clean

#kSZmapGen = lambda i, freq: WLanalysis.readFits(kSZ_dir+'kSZmap_W%i_nearest.fit'%(i))
def kSZmapGen(Wx, freq = '2freqs'):#clean dust
	'''this returns a cleaned kSZ map, 
	if freq='2freqs', return kSZ_freq1*kSZ_freq2'''
	kSZ_NSQ_clean = nosqkSZGen(Wx, freq=freq)
	if freq == '2freqs':
		return kSZ_NSQ_clean
	elif freq == 'dusty':
		return nosqkSZGen_dusty(Wx)**2
	else:
		return kSZ_NSQ_clean**2

def inverse_sum (CC_arr, errK_arr):
	'''both CC_arr, errK_arr should be (4 Wx x 6 bins)
	'''
	weightK = 1/errK_arr/sum(1/errK_arr, axis=0)
	CCK = sum(CC_arr*weightK,axis=0)
	errK = sum(errK_arr*weightK,axis=0)
	return CCK, errK

centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
sizes = (1330, 800, 1120, 950)
PPR512=8468.416479647716
PPA512=2.4633625
#edgesGen = lambda Wx: linspace(5,75,11)*sizes[Wx-1]/1330.0#linspace(5,80,11)
edgesGen = lambda Wx: linspace(5,75,7)*sizes[Wx-1]/1330.0#linspace(5,80,11)
rad2pix=lambda x, size: around(size/2.0-0.5 + x*PPR512).astype(int)
sigmaG_arr = [0.0, 1.0, 2.0, 5.0]

########## need the nex 5 lines only if changed masks #########
#fmask=lambda Wx: sum(maskGen(Wx))/sizes[Wx-1]**2
#fmask2=lambda Wx: sum(maskGen(Wx)**2)/sizes[Wx-1]**2
#fmask2_arr = array(map(fmask2, range(1,5)))
#sizedeg_arr = array([(sizes[Wx-1]/512.0)**2*12.0 for Wx in range(1,5)])
#fsky_arr = fmask_arr*sizedeg_arr/41253.0
##############################################################
fmask_arr = array([0.71254353473399756, 0.65613277819779758, 0.62261798469385554, 0.46024039865666794])
fmask2_arr = [0.65790059649362265, 0.55660343674246793, 0.56069976969877666, 0.4024946100277122]
sizedeg_arr= array([ 80.97381592,  29.296875  ,  57.421875  ,  41.31317139])
fsky_arr = array([ 0.00139862,  0.00046597,  0.00086665,  0.00046091])
ell_arr = array([  433.40339004,   900.1455024 ,  1366.88761476,  1833.62972711, 2300.37183947,  2767.11395182])
d_ell = ell_arr[1]-ell_arr[0]
#######################################################
###### convert from txt to fits #######################
#######################################################
#for fn in os.listdir(kSZ_dir+'null/'):
	#print fn
	#full_fn = kSZ_dir+'null/'+fn
	#data = genfromtxt(full_fn)
	#np.save(full_fn[:-3],data)

#kSZCoordsGen = lambda i: genfromtxt(kSZ_dir+'kSZ2_W%i_ellmax3000.txt'%(i))
#noiseCoordsGen = lambda i: genfromtxt(kSZ_dir+'kSZ2_noise_W%i_ellmax3000.txt'%(i))
#offsetCoordsGen = lambda i: genfromtxt(kSZ_dir+'kSZ2_W%i_ellmax3000_offset.txt'%(i))

#txt2fits_kSZ = lambda Wx: WLanalysis.writeFits(kSZCoordsGen(Wx),kSZ_dir+'kSZ2_W%i_ellmax3000.fit'%(Wx))
#txt2fits_noise = lambda Wx: WLanalysis.writeFits(noiseCoordsGen(Wx),kSZ_dir+'kSZ2_noise_W%i_ellmax3000.fit'%(Wx))
#txt2fits_offset = lambda Wx: WLanalysis.writeFits(offsetCoordsGen(Wx),kSZ_dir+'kSZ2_W%i_ellmax3000_offset.fit'%(Wx))
#map(txt2fits_kSZ,range(1,5))
#map(txt2fits_noise,range(1,5))
#map(txt2fits_offset,range(1,5))
#######################################################


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

def plotimshow(img,ititle,vmin=0, vmax=0.05):		 
	 #if vmin == None and vmax == None:
	imgnonzero=img[nonzero(img)]
	if vmin == None or vmax == None:
		std0 = std(imgnonzero)
		x0 = median(imgnonzero)
		vmin = x0-3*std0
		vmax = x0+3*std0
	im=imshow(img,interpolation='nearest',origin='lower',aspect=1,vmin=vmin,vmax=vmax)
	colorbar()
	title(ititle,fontsize=16)
	savefig(plot_dir+'%s.jpg'%(ititle))
	close()	


def kSZmapGen_fn (fn, offset=False, method='nearest'):
	'''put values to grid, similar to kSZmapGen, except take in the file name.
	'''
	
	Wx = int(fn[fn.index('W')+1])
	print 'Wx, fn:', Wx, fn
	size=sizes[Wx-1]
	kSZCoord = genfromtxt(fn)
	radeclist = kSZCoord[:,:-1]
	values = kSZCoord.T[-1]
	xy = list2coords(radeclist, Wx, offset=offset)
	X,Y=meshgrid(range(size),range(size))
	X=X.ravel()
	Y=Y.ravel()
	newxy=array([X,Y]).T
	newvalues = interpGridpoints (xy, values, newxy,method=method)
	kSZmap = zeros(shape=(size,size))
	kSZmap[Y,X]=newvalues	
	kSZmap[isnan(kSZmap)]=0.0
	if offset:
		kSZmap = kSZmap.T
	np.save(fn[:-4], kSZmap)	
	#return kSZmap

##############################################################
######################## operations ##########################
#for fn in os.listdir(kSZ_dir+'null/'):
	##print fn
	#full_fn = kSZ_dir+'null/'+fn
	#if 'offset' in  fn and 'txt' in fn:
		#print 'offset', fn
		#kSZmapGen_fn(full_fn, offset=True)
##############################################################

def maskGen (Wx, sigma_pix=10):
	galn = galnGen(Wx)
	galn *= ptsrcGen (Wx)## add point source mask for kSZ
	mask = zeros(shape=galn.shape)
	mask[25:-25,25:-25] = 1 ## remove edge 25 pixels
	idx = where(galn<0.5)
	mask[idx] = 0
	mask_smooth = WLanalysis.smooth(mask, sigma_pix)	
	######## print out fksy and fsky 2 ##########
	sizedeg = (sizes[Wx-1]/512.0)**2*12.0
	fsky = sum(mask_smooth)/sizes[Wx-1]**2*sizedeg/41253.0
	fsky2 = sum(mask_smooth**2)/sizes[Wx-1]**2*sizedeg/41253.0
	#print 'W%i, fsky=%.8f, fsky2=%.8f'%(Wx, fsky, fsky2) 
	#############################################
	return mask_smooth#fsky, fsky2#



def KSxkSZ (Wx, method='nearest', sigma_pix=10, freq='2freqs'):
	'''for Wx, get maps, calculate their cross power, auto power, and err
	'''
	print 'KSxkSZ',Wx
	KS = kmapGen(Wx)	
	kSZ = kSZmapGen (Wx, freq=freq)
	noise = noiseGen (Wx)
	Bmode = bmodeGen(Wx)
	offset = offsetGen (Wx)
	nosqkSZ = nosqkSZGen (Wx, freq=freq)
	if freq == '2freqs':
		nosqkSZ = sqrt(nosqkSZ)
		nosqkSZ[isnan(nosqkSZ)] = 0
	
	## masking
	edges = edgesGen(Wx)
	mask_smooth = maskGen(Wx, sigma_pix=sigma_pix)

	#err_arr = WLanalysis.PowerSpectrum(mask_smooth,sizedeg = sizedeg_arr[Wx-1], edges=edges)[0]
	#d_ell = err_arr[1]-err_arr[0]
	
	PS_fcn = lambda kmap: WLanalysis.PowerSpectrum(kmap*mask_smooth, sizedeg = sizedeg_arr[Wx-1], edges=edges)[-1]/fmask2_arr[Wx-1]
	autoK, autokSZ, autoB, autoBMODE, autoO, autoNSQ = map(PS_fcn,[KS, kSZ, noise, Bmode, offset, nosqkSZ])
	
	CC_fcn = lambda maps: WLanalysis.CrossCorrelate (maps[0]*mask_smooth,maps[1]*mask_smooth,edges=edges)[-1]/fmask2_arr[Wx-1]
	CCK, CCB, CCO, CCBMODE, CCNSQ = map(CC_fcn,[[KS,kSZ],[KS,noise],[KS,offset],[Bmode, kSZ],[KS, nosqkSZ]])
	
	#### get rid of the factors in kSZ_NSQ 10/14/2014
	factor = (ell_arr+1)/2/pi
	CCNSQ /= factor
	autoNSQ /= factor

	######## cross and auto err function ######
	CCerr_fcn = lambda pss: sqrt(pss[0]*pss[1]/fsky_arr[Wx-1]/(2*ell_arr+1)/d_ell)#cross correlae err
	ATerr_fcn = lambda ps: sqrt(2*ps**2/fsky_arr[Wx-1]/(2*ell_arr+1)/d_ell)#auto spec err
	
	errK, errB, errO, errBMODE, errNSQ = map(CCerr_fcn,[[autoK,autokSZ],[autoK,autoB],[autoK,autoO],[autoBMODE, autokSZ],[autoK/factor, autoNSQ]])
	
	autoKerr, autokSZerr, autoBerr, autoBMODEerr, autoOerr, autoNSQerr = map(ATerr_fcn, (autoK, autokSZ, autoB, autoBMODE, autoO, autoNSQ))
	
	return CCK, CCB, CCO, CCBMODE, CCNSQ, errK, errB, errO, errBMODE, errNSQ, autoK, autokSZ, autoB, autoBMODE, autoO, autoNSQ, autoKerr, autokSZerr, autoBerr, autoBMODEerr, autoOerr, autoNSQerr

	
if plot_crosscorrelate_all:
	
	method = 'nearest'
	CC_arr = array([KSxkSZ(Wx, freq=freq) for Wx in range(1,5)])
	# CC_arr rows: CCK, CCB, CCO, CCBMODE, CCNSQ, errK, errB, errO, errBMODE, errNSQ, autoK, autokSZ, autoB, autoBMODE, autoO, autoNSQ, autoKerr, autokSZerr, autoBerr, autoBMODEerr, autoOerr, autoNSQerr

	CCK_arr, CCB_arr, CCO_arr, CCBMODE_arr, CCNSQ_arr, errK_arr, errB_arr, errO_arr, errBMODE_arr, errNSQ_arr, autoK_arr, autokSZ_arr, autoB_arr, autoBMODE_arr, autoO_arr, autoNSQ_arr, autoKerr_arr, autokSZerr_arr, autoBerr_arr, autoBMODEerr_arr, autoOerr_arr, autoNSQerr_arr = [CC_arr[:,i] for i in range(CC_arr.shape[1])]
	
	sum_fcn = lambda CandErr: inverse_sum(CandErr[0], CandErr[1])[0]#CandErr = (CC_arr, errK_arr), both with shape (4x6)
	CCK, CCB, CCO, CCBMODE, CCNSQ = map(sum_fcn, [[CCK_arr,errK_arr], [CCB_arr, errB_arr], [CCO_arr, errO_arr], [CCBMODE_arr,errBMODE_arr], [CCNSQ_arr,errNSQ_arr]])

	autoK, autokSZ, autoB, autoBMODE, autoO, autoNSQ = map(sum_fcn, [[autoK_arr,autoKerr_arr], [autokSZ_arr,autokSZ_arr], [autoB_arr,autoBerr_arr], [autoBMODE_arr,autoBMODEerr_arr], [autoO_arr,autoOerr_arr], [autoNSQ_arr,autoNSQerr_arr]])
	
	factor = (ell_arr+1)/2/pi
	CCerr_fcn = lambda pss: sqrt(pss[0]*pss[1]/sum(fsky_arr)/(2*ell_arr+1)/d_ell)
	errK, errB, errO, errBMODE, errNSQ = map(CCerr_fcn,[[autoK,autokSZ],[autoK,autoB],[autoK,autoO],[autoBMODE, autokSZ],[autoK/factor, autoNSQ]])
		
	##############################################################
	####### err from the 500 simulated noise convergence maps ####
	####### (random rotation) ####################################
	##############################################################
	CCN = array([[load(CC_fcn(Wx,freq)+'.npy')/fmask2_arr[Wx-1]] for Wx in range(1,5)]).squeeze()
	errN_arr = std(CCN, axis=1)
	avgN_arr = mean(CCN, axis=1)
	avgN = inverse_sum(avgN_arr, errN_arr)[0]
	####### covariance #######
	weightN = 1/errN_arr/sum(1/errN_arr,axis=0)
	CCN_sum = sum(CCN*weightN.reshape(4,-1,6),axis=0)
	errN = std(CCN_sum, axis=0)
	#errN = sum(weightN*errN_arr,axis=0)#wrong: 1/sum(1/errN_arr, axis=0)
	#avgN = sum(avgN_arr*weightN,axis=0)
	##############################################################
	
	######## plotting
	f=figure(figsize=(8,6))
	ax=f.add_subplot(111)

	### (1) kappa x kSZ^2
	ax.errorbar(ell_arr, CCK, errK, fmt='o',color='b', label=r'$\kappa\times\,kSZ$  ')
	### (2) Bmode x kSZ^2
	#ax.errorbar(ell_arr, CCBMODE, errBMODE, fmt='o',color='g',label=r'$Bmode\times\,kSZ$')	
	#### (3) kappa x instrument noise
	#ax.errorbar(ell_arr, CCB, errB, fmt='o',color='r',label=r'$\kappa\times\,noise$')
	#### (4) kappa x offset CMB
	#ax.errorbar(ell_arr, CCO, errO, fmt='o',color='k',label=r'$\kappa\times\,Offset$')
	## (5) 500 sim #####
	ax.errorbar(ell_arr, avgN, errN, fmt='o',color='y',label=r'$\kappa\,noise\times\,kSZ$')
	######
	### (6) kappa x kSZ_NOTsquared
	#ax.errorbar(ell_arr, CCNSQ, errNSQ, fmt='o',color='m',label=r'$\kappa\times\,kSZ(no\,sq.)$')

	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=0)
	leg.get_frame().set_visible(False)

	ax.set_xlim(0,3000)
	ax.set_xlabel(r'$\ell$', fontsize=16)
	ax.set_ylabel(r'$\ell(\ell+1)P(\ell)/2\pi$', fontsize=16)
	ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

	################## rarely used commands ################
	#if freq:
		#ax.set_title(freq)
	#else:
		#ax.set_title('Dusty')
	#ax.set_title('%s, %s, alpha=%.5f'%(method, freq, alpha))
	#ax.set_ylim(-0.00001,0.00001)
	#ax.set_xscale('log')
	#ax.set_ylabel(r'$\ell\times P(\ell)$', fontsize=16)
	########################################################
	savefig(plot_dir+'test_CrossCorrelate_%s.jpg'%(freq))
	close()
	
	########################################################
	######### save to txt ##################################
	########################################################
	#text_arr = array([ell_arr, CCK, CCO, CCB, CCBMODE, CCNSQ, avgN, errK, errO, errB, errBMODE, errNSQ, errN]).T
	#savetxt(kSZ_dir+'CrossCorrelate_%s_ptsMask_kSZNSQ.txt'%(method), text_arr, header='ell\tkSZ-kappa\toffset-kappa\tnoise-kappa\tkSZ-Bmode\tkSZ_not_sq-kappa\tkSZ-kappa_noise\terr(kSZ-kappa)\terr(offset-kappa)\terr(noise-kappa)\terr(kSZ-Bmode)\terr(kSZ_not_sq-kappa\terr(kSZ-kappa_noise))')
	
	# for 6 bins
	text_arr = array([ell_arr, CCK, CCO, CCB, CCBMODE, CCNSQ, errK, errO, errB, errBMODE, errNSQ]).T
	savetxt(kSZ_dir+'CrossCorrelate_%s_clean_%s.txt'%(method,freq), text_arr, header='ell\tkSZ-kappa\toffset-kappa\tnoise-kappa\tkSZ-Bmode\tkSZ_not_sq-kappa\terr(kSZ-kappa)\terr(offset-kappa)\terr(noise-kappa)\terr(kSZ-Bmode)\terr(kSZ_not_sq-kappa)')
	
	########################################################
	
	############### junk plotting #################
	#CrossPower(CCK, avgN, errK, errN, method=method, noise='KappaNoise')
	#text_arr = array([ell_arr, CCK, avgN, errK, errN]).T
	#savetxt(kSZ_dir+'CrossCorrelate_%s_sigmaG10.txt'%(method), text_arr, header='ell\tkSZxkappa\tkSZxkappa_noise\terr(kSZxkappa)\terr(kSZxkappa_noise)')	
	#CrossPower(CCK, CCB, errK, errB, method=method, noise='noise')
	#CrossPower(CCK, CCO, errK, errO, method=method, noise='offset')
	#CrossPower(CCK, CCBMODE, errK, errBMODE, method=method, noise='Bmode')
	#CrossPower(CCK, CCNSQ, errK, errNSQ, method=method, noise='kSZ\,no\,sq')
	##############################################
	
	#####################################################
	##  test 7/28/2014 plot out each power spectrum: ####
	#####################################################
	#f=figure(figsize=(8,6))
	#ax=f.add_subplot(111)
	#colors=['r','b','g','m']
	#for i in range(4):
		#plot (ell_arr, CC_arr[i,1], colors[i]+'-', label='k x kSZ W%i'%(i+1))
		#plot (ell_arr, CC_arr[i,2], colors[i]+'.', label='k x noise W%i'%(i+1))
		#plot (ell_arr, CC_arr[i,5], colors[i]+'--', label='k x offset W%i'%(i+1))
	#leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
	#leg.get_frame().set_visible(False)
	##ax.set_xscale('log')
	#ax.set_xlabel(r'$\ell$', fontsize=16)
	#ax.set_ylabel(r'$\ell(\ell+1)P_{n\kappa}(\ell)/2\pi$', fontsize=16)
	#ax.set_title('%s, %s arcmin'%(method, sigmaG))
	#ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	##show()
	#savefig(plot_dir+'kSZxCFHT_byWx_%s_sigmaG%s_Bmode.jpg'%(method,sigmaG))
	#close()
	#######################################################
	
if cross_cov_mat:
	CCN = array([[load(CC_fcn(Wx, freq)+'.npy')/fmask2_arr[Wx-1]] for Wx in range(1,5)]).squeeze()
	errN_arr = std(CCN, axis=1)
	weightN = 1/errN_arr/sum(1/errN_arr,axis=0)
	## alternative weight, underestimate, errN = sum(weightN*errN_arr,axis=0)
	CCN_sum = sum(CCN*weightN.reshape(4,-1,6),axis=0)
	CCN_cov = np.cov(CCN_sum,rowvar=0)
	savetxt(kSZ_dir+'cov_kappa-kSZ_%s.txt'%(freq), CCN_cov)
	
	############ SNR ##############
	CC = genfromtxt(kSZ_dir+'CrossCorrelate_nearest_clean_%s.txt'%(freq)).T[1]
	CCerr = genfromtxt(kSZ_dir+'CrossCorrelate_nearest_clean_%s.txt'%(freq)).T[6]
	SNR = sqrt(sum(mat(CC)*mat(CCN_cov).I*mat(CC).T))
	SNR2 = sqrt(sum((CC/CCerr)**2))
	print 'SNR (%s):'%(freq),SNR, SNR2
	###############################
	
	############ plottting ########
	plotimshow(CCN_cov,'cov_%s'%(freq),vmin=None)
	## correlation matrix
	x = sqrt(diag(CCN_cov))
	X,Y=np.meshgrid(x,x)
	CCN_corr = CCN_cov/(X*Y)
	plotimshow(CCN_corr,'corr_%s'%(freq),vmin=None)
	###############################
		
	


if create_noise_KS:
	#from emcee.utils import MPIPool
	#p = MPIPool()
	#from multiprocessing import Pool
	#p = Pool(500)
	
	bmap_fn = lambda Wx, iseed: '/Users/jia/CFHTLenS/catalogue/Noise/W%i/W%i_Noise_sigmaG10_%04d.fit'%(Wx, Wx, iseed)
	kSZmap_arr = map(kSZmapGen, range(1,5))
	mask_arr = map(maskGen, range(1,5))
	masked_kSZ_arr = [kSZmap_arr[i]*mask_arr[i] for i in range(4)]
	def kSZxNoise(iinput):
		'''iinput = (Wx, iseed)
		return the cross power between kSZ and convergence maps, both with smoothed mask.
		'''
		Wx, iseed = iinput
		print 'kSZxNoise', Wx, iseed
		bmap = WLanalysis.readFits(bmap_fn(Wx, iseed))*mask_arr[Wx-1]
		bmap = WLanalysis.smooth(bmap, PPA512)#!!! the file is not smoothed before!!! 11/18/2014
		kSZmap = masked_kSZ_arr[Wx-1]
		edges = edgesGen(Wx)
		ell_arr, CC = WLanalysis.CrossCorrelate (bmap, kSZmap,edges=edges)
		return CC
	for Wx in range(1,5):#(1,):#
		#CC_fn = kSZ_dir+'ptMask_Noise_convxkSZ_W%s.fit'%(Wx)
		CC_arr = map(kSZxNoise, [[Wx, iseed] for iseed in range(500)])
		CC_fn = CC_fcn(Wx, freq)
		np.save(CC_fn, CC_arr)
		#WLanalysis.writeFits(array(CC_arr), CC_fn)
		
if powspec_without_ells_factor:
	Wx = 1
	
	sizedeg = (sizes[Wx-1]/512.0)**2*12.0
	KS = kmapGen(Wx)
	nosqkSZ = nosqkSZGen(Wx)
	
	mask_smooth = maskGen(Wx)
	KS *= mask_smooth
	nosqkSZ *= mask_smooth
	
	fmask = sum(mask_smooth**2)/sizes[Wx-1]**2
	fmask2 = sum(mask_smooth)/sizes[Wx-1]**2
	edges = edgesGen(Wx)
	ell_arr, autoK = WLanalysis.PowerSpectrum(KS, edges=edges, sizedeg = sizedeg)
	ell_arr, autoNSQ = WLanalysis.PowerSpectrum(nosqkSZ, edges=edges, sizedeg = sizedeg)
	ell_arr, crosspower = WLanalysis.CrossCorrelate(KS, nosqkSZ, edges=edges)
	
	factor = ell_arr*(ell_arr+1)/2/pi
	autoK /= fmask2*factor
	autoNSQ /= fmask2*factor
	crosspower /= fmask2*factor
	
	figure()
	subplot(221)
	loglog(ell_arr, autoK)
	title('kappa')
	xlim(ell_arr[0],ell_arr[-1])
	#savefig(plot_dir+'autoK.jpg')
	subplot(222)
	loglog(ell_arr, autoNSQ)
	title('kSZ_NSQ')
	xlim(ell_arr[0],ell_arr[-1])
	subplot(223)
	plot(ell_arr, crosspower)
	title('kappa x kSZ_NSQ')
	xlim(ell_arr[0],ell_arr[-1])
	savefig(plot_dir+'auto_K_kSZ_nosq.jpg')
	close()


if test_powspec:
	f=figure(figsize=(10,8))
	for Wx in range(1,5):
		print Wx
		kSZ = kSZmapGen (Wx)
		mask = zeros(shape=kSZ.shape)
		mask[25:-25,25:-25] = 1
		mask10 = WLanalysis.smooth(mask,10)
		mask20 = WLanalysis.smooth(mask,20)
		#plotimshow(kSZ,'kSZ_map_W%i'%(Wx),vmin=None,vmax=None)
		sizedeg = (sizes[Wx-1]/512.0)**2*12.0
		edges = edgesGen(Wx)
		ell_arr, autokSZ = WLanalysis.PowerSpectrum(kSZ, sizedeg = sizedeg, edges=edges)
		ell_arr, autokSZ_smooth10 = WLanalysis.PowerSpectrum(kSZ*mask10, sizedeg = sizedeg, edges=edges)
		ell_arr, autokSZ_smooth20 = WLanalysis.PowerSpectrum(kSZ*mask20, sizedeg = sizedeg, edges=edges)
		
		ax=f.add_subplot(2,2,Wx)
		ax.plot(ell_arr,autokSZ,label='no smooth')
		ax.plot(ell_arr,autokSZ_smooth10,'--',label='4 arcmin')
		ax.plot(ell_arr,autokSZ_smooth20,'-.',label='8 arcmin')
		#ax.set_xscale('log')
		#ax.set_yscale('log')
		ax.set_xlabel(r'$\ell$')
		ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$')
		ax.set_xlim(ell_arr[0],ell_arr[-1])
		title('W%i'%(Wx))
		if Wx==1:
			leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
			leg.get_frame().set_visible(False)
		#savefig(plot_dir+'test_powspec_W%i.jpg'%(Wx))
		#savetxt(kSZ_dir+'test_smooth_AutoPowspec_W%i.txt'%(Wx), array([ell_arr,autokSZ_smooth]).T, header='ell\tell(ell+1)P/2pi')
	savefig(plot_dir+'test_powspec_smooth_mult.jpg')
	close()
if testCC:
	Wx=1
	#kSZCoord=genfromtxt('/Users/jia/CFHTLenS/kSZ/Jiatest_W1.txt')
	#kSZCoord=genfromtxt('/Users/jia/CFHTLenS/kSZ/ellmax3000/Jiatest_W1_hires.txt')
	#radeclist = kSZCoord[:,:-1]
	#values = kSZCoord.T[-1]
	
	#size=sizes[Wx-1]
	#xy = list2coords(radeclist,Wx)
	#X,Y=meshgrid(range(size),range(size))
	#X=X.ravel()
	#Y=Y.ravel()
	#newxy=array([X,Y]).T
	
	#def plot_test_ps (method):
		#print method
		#newvalues = interpGridpoints (xy, values, newxy, method=method)
		#kSZmap = zeros(shape=(size,size))
		#kSZmap[Y,X]=newvalues
		#edges = edgesGen(Wx)
		#sizedeg = (sizes[Wx-1]/512.0)**2*12.0
		#ell_arr, ps = WLanalysis.PowerSpectrum(kSZmap, sizedeg=sizedeg, edges=edges)
		##WLanalysis.writeFits(kSZmap, kSZ_dir+'test_hires_powerspec_%s.fit'%(method))
		
		##plotimshow(kSZmap, 'test_powerspec_%s'%(method), vmin=None, vmax=None)
		
		#savetxt(kSZ_dir+'test_hires_powspec_W1_%s.txt'%(method),array([ell_arr, ps, ps/(ell_arr*(ell_arr+1)/(2*pi))]).T)
		#return ell_arr, ps
	sizedeg = (sizes[Wx-1]/512.0)**2*12.0
	kSZmap = lambda method: WLanalysis.readFits(kSZ_dir+'test_hires_powerspec_%s.fit'%(method))
	ps_prebin = lambda method: WLanalysis.PowerSpectrum(kSZmap(method),sizedeg=sizedeg)
	ps_afterbin = lambda method: WLanalysis.PowerSpectrum_Pell_binning(kSZmap(method),sizedeg=sizedeg)

	method_arr = ('nearest','linear','cubic')
	ellps_arr = array(map(ps_prebin, method_arr))
	ellps_after_arr = array(map(ps_afterbin, method_arr))
	ell_arr = ellps_arr[0][0]
	
	f=figure(figsize=(8,6))
	ax=f.add_subplot(111)
	ax.set_xlabel(r'$\ell$', fontsize=16)
	ax.set_ylabel(r'$\ell(\ell+1)P_{\kappa\kappa}(\ell)/2\pi$', fontsize=16)
	ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	i=0
	for ps1 in (ellps_arr[0,1],):
		ax.plot(ell_arr,ps1,label='bin P '+method_arr[i])
		i+=1
	i=0
	for ps in (ellps_after_arr[0,1],):
		ax.plot(ell_arr,ps,label='bin P(l+1)l '+method_arr[i])
		i+=1
	#ax.set_title(method)
	legend(loc=0)
	#ax.set_xscale('log')
	ax.set_xlim(0,3500)
	savefig(plot_dir+'test_hires_powspec_W1_binning.jpg')
	close()
	
if clean_dust:
	# purpose of this section is to clean out dust using dust maps at various frequencies
	#!!!
	
	freq = '545217GHzclean'#'857GHz'#

	if freq == '545217GHzclean':
		alpha_arr = linspace(-0.012, 0.003,11)
		#2nd steps: linspace(-0.019, 0.004,11)
		#first steps: linspace(-0.18, 0.05, 21)#150)
		#linspace(-0.002, 0.0005, 150)#
	elif freq == '857GHz':
		alpha_arr = linspace(-0.0002, 0.0, 21)
		#2nd steps: linspace(-0.019, 0.0155, 21)
		#1st steps: linspace(-0.18, 0.05, 21)
		#linspace(-5.4e-5,1.2e-5, 21)
	
	dustGen = lambda i, freq: np.load(kSZ_dir + 'dust/'+'map%s_LGMCAfilt_uK_W%i_flipper8192_toJia.npy'%(freq, i))
	######## (1) convert from coord to grid for all dust maps
	#for fn in os.listdir(kSZ_dir+'dust/'):
		#print fn
		#full_fn = kSZ_dir+'dust/'+fn
		#npy_fn = full_fn[:-4]+'npy'
		#if not os.path.isfile(npy_fn):
			#data = genfromtxt(full_fn)
			#kSZmapGen_fn(full_fn, offset=False)
	#print 'done creating grid map'
	############################################################
	# (2) function that takes in one alpha, spits out cross power	
	dust_arr = [dustGen(Wx, freq) for Wx in range(1,5)]
	mask_arr = map(maskGen, range(1,5))
	sizedeg_arr = array([(sizes[Wx-1]/512.0)**2*12.0 for Wx in range(1,5)])
	fmask_arr = array([sum(mask_arr[Wx-1]**2)/sizes[Wx-1]**2 for Wx in range(1,5)])
	fsky_arr = fmask_arr*sizedeg_arr/41253.0
	
	kSZ_NSQ_arr = map(nosqkSZGen_dusty, range(1,5))
	
	#kmap_arr = map(kmapGen, range(1,5))
	kmap_arr = [kmapGen(Wx)*mask_arr[Wx-1] for Wx in range(1,5)]
	
	edges_arr = map(edgesGen, range(1,5))
	#fsky2_arr = [0.001291, 0.000395, 0.000780, 0.000403]
	fmask2_arr = [sum(mask_arr[Wx-1])/sizes[Wx-1]**2 for Wx in range(1,5)]
	
	ell_arr = WLanalysis.PowerSpectrum(ones(shape=(1330,1330)), sizedeg=sizedeg_arr[0], edges=edges_arr[Wx-1])[0]
	d_ell = ell_arr[1]-ell_arr[0]
	factor = (ell_arr+1)/(2*pi)#because we want ell*P, only one power of ell needed
	
	def theory_err(map1, map2, Wx):	
		auto1 = WLanalysis.PowerSpectrum(map1, sizedeg = sizedeg_arr[Wx-1], edges=edges_arr[Wx-1])[-1]/fmask2_arr[Wx-1]/factor
		auto2 = WLanalysis.PowerSpectrum(map2, sizedeg = sizedeg_arr[Wx-1], edges=edges_arr[Wx-1])[-1]/fmask2_arr[Wx-1]/factor	
		errNSQ = sqrt(auto1*auto2/fsky_arr[Wx-1]/(2*ell_arr+1)/d_ell)
		return errNSQ
		
	def crosspower_Wx (Wx, freq, alpha):
		dust = dust_arr[Wx-1]#!!!
		kSZ_NSQ_clean = (1+alpha)*kSZ_NSQ_arr[Wx-1]-alpha*dust
		kSZ_NSQ_clean *= mask_arr[Wx-1]
		ell_arr, ps = WLanalysis.CrossCorrelate(kSZ_NSQ_clean, kmap_arr[Wx-1], edges=edges_arr[Wx-1])
		ps /= fmask2_arr[Wx-1]*factor
		errNSQ = theory_err(kmap_arr[Wx-1], kSZ_NSQ_clean, Wx)
		return ps, errNSQ
	
	def minimize_dust(alpha, freq):
		print freq, alpha
		a = array([crosspower_Wx(Wx, freq, alpha) for Wx in range(1,5)])
		CC_arr, errK_arr = a[:,0,:], a[:,1,:]
		CC, err = inverse_sum(CC_arr, errK_arr)
		return CC, err
	
		
	#results = array([minimize_dust(alpha, freq) for alpha in alpha_arr])#alpha x 2 x 6
	#CCK_arr, errK_arr = results[:,0,:], results[:,1,:]
	
	################### save file ####################################
	#np.save(kSZ_dir+'clean_dust_%sGHz_alpha'%(freq), concatenate([alpha_arr.reshape(-1,1), CCK_arr, errK_arr],axis=1))
	################################################################
	#data_from_file = np.load(kSZ_dir+'clean_dust_%sGHz_alpha_largesteps.npy'%(freq)).T
	data_from_file = np.load(kSZ_dir+'clean_dust_%sGHz_alpha.npy'%(freq)).T
	alpha_arr = data_from_file[0]
	CCK_arr = data_from_file[1:7]
	errK_arr = data_from_file[7:]
		
	###################### plot cleaned power vs alpha##############
	f=figure()
	for i in range(CCK_arr.shape[0]):
		ax=f.add_subplot(3,2,i+1)
		errorbar(alpha_arr, abs(CCK_arr[i]), errK_arr[i], label='%ith bin'%(i+1))
		legend(fontsize=10)
		if i > 3:
			xlabel('alpha')
		if i in (0, 2, 4):
			ylabel('abs[ell x P(ell)]')
		if i <2:
			title('%s GHz'%(freq))
		ax.locator_params(nbins=4)
	savefig(plot_dir+'clean_dust_%sGHz_alpha_test.jpg'%(freq))
	close()
	################################################################
	
	### kinda useless - min alpha for 6 individual bins #################
	min_idx = argmin(abs(CCK_arr),axis=1)
	alpha_min_arr=array([alpha_arr[i] for i in min_idx])
	CCK_min_arr = array([CCK_arr[i, min_idx[i]] for i in range(len(min_idx))])
	#alpha_min = mean(alpha_min_arr)
	alpha_min = mean(alpha_min_arr[[0,1,2,5]])
	#####################################################
	
	SNR = sqrt(sum((CCK_arr/errK_arr)**2,axis=0))#to find alpha that minimize SNR
	sumCCK = sqrt(sum(CCK_arr**2,axis=0))#to find alpha that minimize total power
	print '%s alpha (minimize power):'%(freq),alpha_arr[argmin(sumCCK)]
	print '%s alpha (minimize SNR):'%(freq),alpha_arr[argmin(SNR)]
	
	########### plot out total P and SNR vs. alpha ##########
	figure(figsize=(12,8))
	subplot(221)
	plot(alpha_arr, sumCCK)
	title(freq)
	ylabel('total power')
	#xlabel('alpha')

	subplot(222)
	errorbar(ell_arr, CCK_arr[:,argmin(sumCCK)]/factor, errK_arr[:,argmin(sumCCK)],fmt='o')
	title('alpha(minimize tot power) = %s'%(alpha_arr[argmin(sumCCK)]))
	#xlabel('ell')
	ylabel('ell x P')
		
	subplot(223)
	plot(alpha_arr, SNR)
	ylabel('SNR')
	#title('%s alpha (minimize SNR): %s'%(freq, alpha_arr[argmin(SNR)]))
	xlabel('alpha')

	subplot(224)
	errorbar(ell_arr, CCK_arr[:,argmin(SNR)]/factor, errK_arr[:,argmin(SNR)],fmt='o')
	title('alpha(minimize SNR) = %s'%(alpha_arr[argmin(SNR)]))
	xlabel('ell')
	ylabel('ell x P')

	plt.subplots_adjust(hspace=0.2, wspace=0.3)
	savefig(plot_dir+'clean_dust_alpha_min_%s.jpg'%(freq))
	close()
#########################################################

def CrossPower(CCK, CCB, errK, errB, method='nearest', sigma_pix=10, signa='kappa', noise='noise'):
	
	f=figure(figsize=(8,6))
	ax=f.add_subplot(111)

	ax.errorbar(ell_arr, CCK, errK, fmt='o',color='b', label=r'$\kappa\times\,kSZ$  ')
	ax.errorbar(ell_arr, CCB, errB, fmt='o',color='r',label=r'$%s\times\,kSZ$'%(noise))

	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=0)
	leg.get_frame().set_visible(False)
	#ax.set_xscale('log')
	ax.set_xlim(0,3000)
	ax.set_xlabel(r'$\ell$', fontsize=16)
	ax.set_ylabel(r'$\ell(\ell+1)P_{n\kappa}(\ell)/2\pi$', fontsize=16)
	ax.set_title('%s, %s pix mask, 1 arcmin smooth conv. map'%(method, sigma_pix))
	ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	savefig(plot_dir+'CrossCorrelate_kSZxCFHT_%s_sigmapix%s_%s.jpg'%(method,sigma_pix, noise))
	close()

def KSxkSZ_junk (Wx, method='nearest', sigma_pix=10, freq='2freqs'):
	print 'KSxkSZ',Wx
	KS = kmapGen(Wx)
	Bmode = bmodeGen(Wx)
	kSZ = kSZmapGen (Wx, freq=freq)
	noise = noiseGen (Wx)
	offset = offsetGen (Wx)
	nosqkSZ = nosqkSZGen (Wx, freq=freq)
	if freq == '2freqs':
		nosqkSZ = sqrt(nosqkSZ)
		nosqkSZ[isnan(nosqkSZ)] = 0
	
	## masking
	mask_smooth = maskGen(Wx, sigma_pix=sigma_pix)
	KS *= mask_smooth
	kSZ *= mask_smooth
	noise *= mask_smooth
	offset *= mask_smooth
	Bmode *= mask_smooth
	nosqkSZ *= mask_smooth
	
	sizedeg = (sizes[Wx-1]/512.0)**2*12.0
	fmask2 = sum(mask_smooth**2)/sizes[Wx-1]**2
	fmask = sum(mask_smooth)/sizes[Wx-1]**2	
	fsky = fmask*sizedeg/41253.0# 41253.0 deg is the full sky in degrees
	fsky2 = fmask2*sizedeg/41253.0
	
	edges = edgesGen(Wx)
	ell_arr, CCK = WLanalysis.CrossCorrelate (KS,kSZ,edges=edges)	
	ell_arr, CCB = WLanalysis.CrossCorrelate (KS,noise,edges=edges)
	ell_arr, CCO = WLanalysis.CrossCorrelate (KS,offset,edges=edges)
	ell_arr, CCBMODE = WLanalysis.CrossCorrelate (Bmode, kSZ, edges=edges)
	ell_arr, CCNSQ = WLanalysis.CrossCorrelate (KS, nosqkSZ, edges=edges)
	
	CCK /= fmask2
	CCB /= fmask2
	CCO /= fmask2
	CCBMODE /= fmask2
	#### get rid of the factors in kSZ_NSQ 10/14/2014
	factor = (ell_arr+1)/2/pi
	CCNSQ /= fmask2*factor

	# error
	autoK = WLanalysis.PowerSpectrum(KS, sizedeg = sizedeg, edges=edges)[-1]/fmask2
	autokSZ = WLanalysis.PowerSpectrum(kSZ, sizedeg = sizedeg, edges=edges)[-1]/fmask2
	autoB = WLanalysis.PowerSpectrum(noise, sizedeg = sizedeg, edges=edges)[-1]/fmask2
	autoO = WLanalysis.PowerSpectrum(offset, sizedeg = sizedeg, edges=edges)[-1]/fmask2
	autoBMODE = WLanalysis.PowerSpectrum(Bmode, sizedeg = sizedeg, edges=edges)[-1]/fmask2
	### get rid of one ell 10/14/2014
	autoNSQ = WLanalysis.PowerSpectrum(nosqkSZ, sizedeg = sizedeg, edges=edges)[-1]/(fmask2*factor)#fmask2#

	d_ell = ell_arr[1]-ell_arr[0]
	##################### junk ############
	#errK = sqrt(autoK*autokSZ/fsky/d_ell)
	#errB = sqrt(autoK*autoB/fsky/d_ell)
	#errO = sqrt(autoK*autoO/fsky/d_ell)
	#errBMODE = sqrt(autoBMODE*autokSZ/fsky/d_ell)
	#######################################
	errK = sqrt(autoK*autokSZ/fsky/(2*ell_arr+1)/d_ell)
	errB = sqrt(autoK*autoB/fsky/(2*ell_arr+1)/d_ell)
	errO = sqrt(autoK*autoO/fsky/(2*ell_arr+1)/d_ell)
	errBMODE = sqrt(autoBMODE*autokSZ/fsky/(2*ell_arr+1)/d_ell)
	#errNSQ = sqrt(autoK*autoNSQ/fsky/(2*ell_arr+1)/d_ell)
	####### below line get rid of the ell factor for kSZ_NSQ, kappa
	errNSQ = sqrt(autoK/factor*autoNSQ/fsky/(2*ell_arr+1)/d_ell)
	print fsky, fmask2
	return ell_arr, CCK, CCB, errK, errB, CCO, errO, CCBMODE, errBMODE, CCNSQ, errNSQ, autoK, autokSZ, autoB, autoO, autoBMODE, autoNSQ


def plot_crosscorrelate_all_junk():
	
	method = 'nearest'
	CC_arr = array([KSxkSZ_junk(Wx, method=method, freq=freq) for Wx in range(1,5)])
	# CC_arr rows: 0 ell_arr, 1 CCK-x, 2 CCB, 3 errK-x, 4 errB
	# 5 CCO, 6 errO, 7 CCBMODE-x, 8 errBMODE-x, 9 CCNSQ-x, 10 errNSQ-x
	# 11 autoK, 12 autoKSZ, 13 autoB, 14 autoO, 15 autoBMODE, 16 autoNSQ

	#autoK_arr = CC_arr [:, 11]
	#autoKSZ_arr=CC_arr [:, 12]
	#autoB_arr = CC_arr [:, 13]
	#autoO_arr = CC_arr [:, 14]
	#autoBMODE_arr=CC_arr [:, 15]
	#autoNSQ_arr=CC_arr [:, 16]
	
	errK_arr = CC_arr[:,3]
	errB_arr = CC_arr[:,4]#pure instrumentation noise
	errO_arr = CC_arr[:,6]#using offset map
	errBMODE_arr = CC_arr[:,8]
	errNSQ_arr = CC_arr[:,10]
	
	weightK = 1/errK_arr/sum(1/errK_arr, axis=0)
	weightB = 1/errB_arr/sum(1/errB_arr, axis=0)
	weightO = 1/errO_arr/sum(1/errO_arr, axis=0)
	weightBMODE = 1/errBMODE_arr/sum(1/errBMODE_arr, axis=0)
	weightNSQ = 1/errNSQ_arr/sum(1/errNSQ_arr, axis=0)
	
	######################### error calculation ###############
	# averaged over 4 err
	errK = 1/sum(1/errK_arr, axis=0)
	errB = 1/sum(1/errB_arr, axis=0)
	errO = 1/sum(1/errO_arr, axis=0)
	errBMODE = 1/sum(1/errBMODE_arr, axis=0)
	errNSQ = 1/sum(1/errNSQ_arr, axis=0)
	
	# average the power spectrum first, then calculate error
	# should be more correct
	#fsky=0.00287#sum of all 4 patches
	#errK = sqrt(autoK*autokSZ/fsky/(2*ell_arr+1)/d_ell)
	#errB = sqrt(autoK*autoB/fsky/(2*ell_arr+1)/d_ell)
	#errO = sqrt(autoK*autoO/fsky/(2*ell_arr+1)/d_ell)
	#errBMODE = sqrt(autoBMODE*autokSZ/fsky/(2*ell_arr+1)/d_ell)
	#############################################################
	
	ell_arr = CC_arr[0,0]
	CCBMODE = sum(CC_arr[:,7]*weightBMODE,axis=0)
	CCK = sum(CC_arr[:,1]*weightK,axis=0)
	CCB = sum(CC_arr[:,2]*weightB,axis=0)#pure instrumentation noise
	CCO = sum(CC_arr[:,5]*weightO,axis=0)#using offset map
	CCNSQ = sum(CC_arr[:,9]*weightNSQ,axis=0)
	
	return CCK, CCB, CCO, CCNSQ
	###############################################################
	######## err from the 500 simulated noise convergence maps ####
	######## (random rotation) ####################################
	###############################################################
	##CCN = array([[load(CC_fcn(Wx,freq)+'.npy')/fmask2_arr[Wx-1]] for Wx in range(1,5)]).squeeze()
	##errN_arr = std(CCN, axis=1)
	##avgN_arr = mean(CCN, axis=1)
	##weightN = 1/errN_arr/sum(1/errN_arr,axis=0)
	##errN = sum(weightN*errN_arr,axis=0)#wrong: 1/sum(1/errN_arr, axis=0)
	##avgN = sum(avgN_arr*weightN,axis=0)
	###############################################################
	
	######### plotting
	#f=figure(figsize=(8,6))
	#ax=f.add_subplot(111)

	#### (1) kappa x kSZ^2
	#ax.errorbar(ell_arr, CCK, errK, fmt='o',color='b', label=r'$\kappa\times\,kSZ$  ')
	#### (2) Bmode x kSZ^2
	##ax.errorbar(ell_arr, CCBMODE, errBMODE, fmt='o',color='g',label=r'$Bmode\times\,kSZ$')	
	##### (3) kappa x instrument noise
	##ax.errorbar(ell_arr, CCB, errB, fmt='o',color='r',label=r'$\kappa\times\,noise$')
	##### (4) kappa x offset CMB
	##ax.errorbar(ell_arr, CCO, errO, fmt='o',color='k',label=r'$\kappa\times\,Offset$')

	### (5) 500 sim #####
	#ax.errorbar(ell_arr, avgN, errN, fmt='o',color='y',label=r'$\kappa\,noise\times\,kSZ$')
	#######
	#### (6) kappa x kSZ_NOTsquared
	##ax.errorbar(ell_arr, CCNSQ, errNSQ, fmt='o',color='m',label=r'$\kappa\times\,kSZ(no\,sq.)$')

	#leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=0)
	#leg.get_frame().set_visible(False)

	#ax.set_xlim(0,3000)
	#ax.set_xlabel(r'$\ell$', fontsize=16)
	#ax.set_ylabel(r'$\ell(\ell+1)P(\ell)/2\pi$', fontsize=16)
	#ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

	################### rarely used commands ################
	##if freq:
		##ax.set_title(freq)
	##else:
		##ax.set_title('Dusty')
	##ax.set_title('%s, %s, alpha=%.5f'%(method, freq, alpha))
	##ax.set_ylim(-0.00001,0.00001)
	##ax.set_xscale('log')
	##ax.set_ylabel(r'$\ell\times P(\ell)$', fontsize=16)
	#########################################################
	#savefig(plot_dir+'test_CrossCorrelate_%s.jpg'%(freq))
	#close()
	
	########################################################
	######### save to txt ##################################
	########################################################
	#text_arr = array([ell_arr, CCK, CCO, CCB, CCBMODE, CCNSQ, avgN, errK, errO, errB, errBMODE, errNSQ, errN]).T
	#savetxt(kSZ_dir+'CrossCorrelate_%s_ptsMask_kSZNSQ.txt'%(method), text_arr, header='ell\tkSZ-kappa\toffset-kappa\tnoise-kappa\tkSZ-Bmode\tkSZ_not_sq-kappa\tkSZ-kappa_noise\terr(kSZ-kappa)\terr(offset-kappa)\terr(noise-kappa)\terr(kSZ-Bmode)\terr(kSZ_not_sq-kappa\terr(kSZ-kappa_noise))')
	
	# for 6 bins
	#text_arr = array([ell_arr, CCK, CCO, CCB, CCBMODE, CCNSQ, errK, errO, errB, errBMODE, errNSQ]).T
	#savetxt(kSZ_dir+'CrossCorrelate_%s_clean_%s.txt'%(method,freq), text_arr, header='ell\tkSZ-kappa\toffset-kappa\tnoise-kappa\tkSZ-Bmode\tkSZ_not_sq-kappa\terr(kSZ-kappa)\terr(offset-kappa)\terr(noise-kappa)\terr(kSZ-Bmode)\terr(kSZ_not_sq-kappa)')
	
	########################################################
	
	############### junk plotting #################
	#CrossPower(CCK, avgN, errK, errN, method=method, noise='KappaNoise')
	#text_arr = array([ell_arr, CCK, avgN, errK, errN]).T
	#savetxt(kSZ_dir+'CrossCorrelate_%s_sigmaG10.txt'%(method), text_arr, header='ell\tkSZxkappa\tkSZxkappa_noise\terr(kSZxkappa)\terr(kSZxkappa_noise)')	
	#CrossPower(CCK, CCB, errK, errB, method=method, noise='noise')
	#CrossPower(CCK, CCO, errK, errO, method=method, noise='offset')
	#CrossPower(CCK, CCBMODE, errK, errBMODE, method=method, noise='Bmode')
	#CrossPower(CCK, CCNSQ, errK, errNSQ, method=method, noise='kSZ\,no\,sq')
	##############################################
	
	#####################################################
	##  test 7/28/2014 plot out each power spectrum: ####
	#####################################################
	#f=figure(figsize=(8,6))
	#ax=f.add_subplot(111)
	#colors=['r','b','g','m']
	#for i in range(4):
		#plot (ell_arr, CC_arr[i,1], colors[i]+'-', label='k x kSZ W%i'%(i+1))
		#plot (ell_arr, CC_arr[i,2], colors[i]+'.', label='k x noise W%i'%(i+1))
		#plot (ell_arr, CC_arr[i,5], colors[i]+'--', label='k x offset W%i'%(i+1))
	#leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
	#leg.get_frame().set_visible(False)
	##ax.set_xscale('log')
	#ax.set_xlabel(r'$\ell$', fontsize=16)
	#ax.set_ylabel(r'$\ell(\ell+1)P_{n\kappa}(\ell)/2\pi$', fontsize=16)
	#ax.set_title('%s, %s arcmin'%(method, sigmaG))
	#ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	##show()
	#savefig(plot_dir+'kSZxCFHT_byWx_%s_sigmaG%s_Bmode.jpg'%(method,sigmaG))
	#close()
	#######################################################
	
#a=plot_crosscorrelate_all_junk()