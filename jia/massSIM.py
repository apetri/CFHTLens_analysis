#!/afs/rhic.bnl.gov/@sys/opt/astro/SL64/anaconda/bin
# yeti: /vega/astro/users/jl3509/tarball/anacondaa/bin/python
# Jia Liu 2014/3/7
# Overview: this code creates mass maps from simulation
################ steps #####################
#1) smoothing, use random galaxy direction, and w as wegith
#2) KSvw
#3) count peaks, MF, powerspectrum 

import WLanalysis
from emcee.utils import MPIPool
import os
import numpy as np
from scipy import *
import scipy.ndimage as snd

########## define constants ############
ngal_arcmin = 5.0
zmax=1.3
zmin=0.2

ngal_cut = ngal_arcmin*(60**2*12)/512**2# = 0.82, cut = 5 / arcmin^2
PPR512=8468.416479647716#pixels per radians
PPA512=2.4633625
rad2pix=lambda x: around(512/2.0-0.5 + x*PPR512).astype(int) #from radians to pixel location

full_dir = '/direct/astro+astronfs01/workarea/jia/CFHT/full_subfields/'
KS_dir = '/direct/astro+astronfs01/workarea/jia/CFHT/KSsim/'
sim_dir = '/direct/astro+astronfs01/workarea/jia/CFHT/galaxy_catalogue_128R'

sigmaG_arr = (0.5, 1, 1.8, 3.5, 5.3, 8.9)

## generate random rotation while preserve galaxy size and shape info

rndrot = lambda gamma: -gamma*exp(-4j*pi*rand(len(a)))

#### test
# a=rand(10)+rand(10)*1j
# b=rndrot(a)
# abs(a)==abs(b) #return True

## create index file for z cut ###
def zcut_idx (i, zmin=zmin, zmax=zmax):
	'''return index for z cut
	'''
	fn = full_dir+'zcut_idx_subfield%i'%(i)
	
	if WLanalysis.TestComplete(fn):
		idx = genfromtxt(fn)
	
	else:
		zs = np.genfromtxt(full_dir+'full_subfield'+str(i) ,usecols=[2, 3, 4])
		# z_peak, z_rnd1, z_rnd2

		# redshift cut 0.2< z <1.3
		print 'zs', i
		idx = np.where((amax(zs,axis=1) <= zmax) & (amin(zs,axis=1) >= zmin))[0]
		savetxt(fn,idx)
	return idx

for i in arange(1,14):
	zcut_idx (i)
