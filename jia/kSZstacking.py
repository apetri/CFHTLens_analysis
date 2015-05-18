##########################################################
### This code is to stack kSZ maps at peak location
### or massive galaxy location

import numpy as np
from scipy import *
from pylab import *
import os,sys
import WLanalysis

kSZ_dir = '/Users/jia/CFHTLenS/kSZ/newfil/'
plot_dir = '/Users/jia/CFHTLenS/kSZ/stacking/plot/'
freq_arr = ['2freqs', '545217GHzclean', '857GHz', 'dusty']

sizes = (1330, 800, 1120, 950)
PPA512 = 2.4633625
PPR512=8468.416479647716
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
def return_alpha (freq): 
	if freq == '545217GHzclean':
		alpha = -0.0045
	elif freq == '857GHz':
		alpha = -8e-5
	return alpha

kmapGen = lambda i, sigmaG: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_KS_1.3_lo_sigmaG%2d.fit'%(i, sigmaG*10))

ptsrcGen = lambda i: np.load(kSZ_dir + 'null/'+'PSmaskRING_100-143-217-353-545_857_5sigma_Nside8192_BOOL_W%s_toJia.npy'%(i))

galnGen = lambda i: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_galn_1.3_lo_sigmaG10.fit'%(i))

def maskGen (Wx, sigmaG=1.0, sigma_pix=0):
	galn = galnGen(Wx)
	galn = WLanalysis.smooth(galn, PPA512*sigmaG)
	mask = ones(shape=galn.shape)
	idx = where(galn<0.5)
	mask[idx] = 0
	mask *= ptsrcGen(Wx)
	mask_smooth = WLanalysis.smooth(mask, sigma_pix)
	return mask_smooth

nosqkSZGen_dusty= lambda i: np.load(kSZ_dir + 'null/'+'LGMCA_W%s_flipper8192_kSZfilt_NOTsquared_toJia.npy'%(i))

dustGen = lambda i, freq: np.load(kSZ_dir + 'dust/'+'map%s_LGMCAfilt_uK_W%i_flipper8192_toJia.npy'%(freq, i))

def nosqkSZGen(Wx, freq = '2freqs'):
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
	elif freq == 'dusty':
		kSZ_NSQ_clean = nosqkSZGen_dusty(Wx)**2
	else:
		dust = dustGen(Wx, freq)
		alpha = return_alpha(freq)
		kSZ_NSQ_clean = (1+alpha)*kSZ_NSQ-alpha*dust
	return kSZ_NSQ_clean

def peaklocs (Wx, sigmaG=1.0):
	'''return peaks values and locations of peaks in the lensing map'''
	kmap = kmapGen(Wx, sigmaG)
	mask = maskGen(Wx, sigmaG)
	peaks_mat = WLanalysis.peaks_mat(kmap)
	peaks_mat [mask==0] = nan
	idx = where(~isnan(peaks_mat))
	return kmap[idx], array(idx).T

rad2pix=lambda x, size: around(size/2.0-0.5 + x*PPR512).astype(int)
def gallocs (Wx, Mcut=1e12):
	icat = np.load('/Users/jia/CFHTLenS/obsPK/W%s_cat_z0213_ra_dec_redshift_weight_MAGi_Mvir_Rvir_DL.npy'%(Wx)).T#ra, dec, redshift, weight, i, Mhalo, Rvir, DL
	ra, dec, redshift, weight, i, Mhalo, Rvir, DL = icat
	center = centers[Wx-1]
	f_Wx = WLanalysis.gnom_fun(center)#turns to radians
	xy = array(f_Wx(icat[:2])).T
	xy_pix = rad2pix(xy, size=sizes[Wx-1])
	return xy_pix[Mhalo>Mcut]
	
def ipatch_fcn(imap, pix=25):
	dx = pix/2
	def ipatch(loc):
		x, y = loc
		iimap = imap[x-dx:x+dx+1, y-dx:y+dx+1]
		if amin(iimap.shape)< 2*dx+1:
			iimap = zeros(shape=(25,25))
		return iimap
	return ipatch
	
def patch_fcn (Wx, sigmaG=1.0, pix=25, freq='2freqs'):
	#kappa, locs = peaklocs(Wx, sigmaG=sigmaG)
	#locs = locs[kappa>2*std(kappa)]
	locs = gallocs (Wx, Mcut=1e12)
	nosqkSZ = nosqkSZGen(Wx, freq=freq)
	kSZmap = nosqkSZ**2
	patches_nosq = array(map(ipatch_fcn(nosqkSZ), locs))
	patches_sqed = array(map(ipatch_fcn(kSZmap), locs))
	return patches_nosq, patches_sqed

sigmaG_arr = [1.0, 1.8, 3.5, 5.3, 8.9]
for sigmaG in (1.0,):#sigmaG_arr:
	freq = freq_arr[1]
	
	######### std
	nosqkSZ = nosqkSZGen(1, freq=freq)
	kSZmap = nosqkSZ**2
	mask = maskGen(1, sigmaG)
	nosqstd, kSZstd = std(nosqkSZ[mask>0]),std(kSZmap[mask>0])
	###########
	patches = [patch_fcn (i, sigmaG = sigmaG, freq=freq) for i in range(1,5)]
	patches_nosq = concatenate([patches[i][0] for i in range(4)],axis=0)
	patches_sqed = concatenate([patches[i][1] for i in range(4)],axis=0)
	stack_nosq = sum(patches_nosq,axis=0)/patches_nosq.shape[0]
	stack_sqed = sum(patches_sqed,axis=0)/patches_sqed.shape[0]
	print sigmaG, 'halos',patches_nosq.shape[0]
	figure(figsize=(12,6))
	subplot(121)
	imshow(stack_nosq/nosqstd)
	colorbar()
	#title('non-squared %s'%(freq))
	title('non-squared')
	
	subplot(122)
	imshow(stack_sqed/kSZstd)
	colorbar()
	title('kSZ squared')# (sigmaG %s)' %(sigmaG))
	savefig(plot_dir+'stack_1e12halos.jpg')
	#savefig(plot_dir+'stack_allhalos_sigmaG%02d.jpg'%(sigmaG*10))
	close()