#!python
# Jia Liu 2015/06/01
# This code calculates the stacking signal for tSZ and CFHT on clusters

import WLanalysis
import os
import numpy as np
from scipy import *
import sys
from scipy.integrate import quad
import scipy.optimize as op
from scipy import interpolate
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.gridspec as gridspec
from scipy import ndimage as snd

centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
PPA512 = 2.4633625
sizes = (1330, 800, 1120, 950)
sizedeg_arr = array([(sizes[Wx-1]/512.0)**2*12.0 for Wx in range(1,5)])
prefix_arr = ('nilc_ymap', 'milca_ymap', 'GARY_ymap', 'JCH_ymap50')

tSZ_dir = '/Users/jia/weaklensing/tSZxCFHT/'
plot_dir = tSZ_dir+'plot/'

cat_dir = '/Users/jia/weaklensing/CFHTLenS/catalogue/'

#kmapGen = lambda Wx: load(cat_dir+'kmap_W%i_sigma10_zcut13.npy'%(Wx))
kmapGen = lambda Wx: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_KS_1.3_lo_sigmaG18.fit'%(Wx))

tmapGen = lambda prefix, Wx: load(tSZ_dir+'planck/%s_CFHTLS_W%i.npy'%(prefix, Wx))

maskGen = lambda i: load(tSZ_dir+'mask/W%i_mask.npy'%(i))

#generate cluster locations in [nx2] matrix, RA, DEC in deg.
def locs_fcn(Wx, sig_cut = 10.0):
	ra, dec, z, sig = genfromtxt('/Users/jia/weaklensing/tSZxCFHT/cfhtlens_3DMF_clusters/cfhtlens_3DMF_clusters_W%i.cat'%(Wx)).T
	idx_cut = where((z<0.7)&(sig>sig_cut))[0]
	radec_list = (array([ra,dec]).T)[idx_cut]
	locs = WLanalysis.list2coords(radec_list, Wx)
	return locs

def ipatch_fcn(imap, pix=45):
	dx = pix/2
	def ipatch(loc):
		x, y = loc
		iimap = imap[x-dx:x+dx+1, y-dx:y+dx+1]
		if amin(iimap.shape)< 2*dx+1:
			iimap = zeros(shape=(25,25))
		return iimap
	return ipatch

def patch_fcn (Wx, sigmaG=1.0, pix=45, sig_cut = 10.0):
	'''For Wx, stack on clusters
	'''
	
	mask = maskGen(Wx)
	#mask[mask==0] = nan
	#maps_arr = [kmapGen(Wx), tmapGen(prefix_arr[0], Wx), tmapGen(prefix_arr[1], Wx)]
	kmap = kmapGen(Wx)*mask
	nilc = tmapGen(prefix_arr[0], Wx)*mask
	milc = tmapGen(prefix_arr[1], Wx)*mask
	#print 'W%i  %.2e  %.2e  %.2e'% (Wx, std(kmap), std(nilc), std(milc))
	#print 'W%i  %.2e  %.2e  %.2e'% (Wx, mean(kmap), mean(nilc), mean(milc))
	locs = locs_fcn(Wx, sig_cut = sig_cut)
	####### shuffle one axis to get random direction
	#shuffle(locs.T[0])
	#shuffle(locs.T[1])
	patches_kmap = array(map(ipatch_fcn(kmap), locs))
	patches_nilc = array(map(ipatch_fcn(nilc), locs))
	patches_milc = array(map(ipatch_fcn(milc), locs))
	
	return patches_kmap, patches_nilc, patches_milc

fn_arr = ['kmap', 'nilc', 'milc']
sig_cut = 10.0
out = [patch_fcn(Wx, sig_cut = sig_cut) for Wx in range(1,5)]
out_all = array([concatenate([out[i][j] for i in range(4)], axis=0) for j in range(3)])
figure(figsize=(15,4))		
for i in range(3):
	subplot(1,3,i+1)
	idx_nomask = ~isnan(sum(out_all[i], axis=(1,2)))
	imshow(mean(out_all[i][idx_nomask],axis=0))
	colorbar()
	title ('%s, sig>%i(%i)'%(fn_arr[i], sig_cut, out_all[i].shape[0]))
	#title ('%s, random'%(fn_arr[i]))
	i += 1
savefig(plot_dir+'stacking_all_z07_sig%i.jpg'%(sig_cut))#_random2
close()
