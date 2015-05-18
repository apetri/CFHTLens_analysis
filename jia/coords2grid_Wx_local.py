#!python

##########################################################
### This code does the following:
### 1) organizes the CFHT catalogue to 4 Wx fields, with proper 
### format that's easy to use in the future
### 2) converts RA DEC to (x, y) radian, using Genomonic projection,
### centered at map center
### 3) final products: convergence maps and galcount maps

import numpy as np
from scipy import *
import os
import WLanalysis

############## CFHT catalogue download commands ########
#### SELECT
#### ALPHA_J2000, DELTA_J2000, e1, e2, weight, Z_B, m, c2
#### FROM
#### cfht.clens
#### WHERE
#### weight>=0.0001
#### AND MASK<=1
#### AND star_flag>=0
#### AND star_flag<=0
##########################################################
############## constants #################################
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
sizes = (1330, 800, 1120, 950)
PPR512=8468.416479647716
PPA512=2.4633625
rad2pix=lambda x, size: around(size/2.0-0.5 + x*PPR512).astype(int)

RA1 =(30.0, 39.0)#starting RA for W1
DEC1=(-11.5,-3.5)
RA2 =(132.0, 137.0)
DEC2=(-6.0,-0.5)
RA3 =(208.0, 221.0)
DEC3=(51.0, 58.0)
RA4 =(329.5, 336.0)
DEC4=(-1.2, 5.0)
RAs=(RA1,RA2,RA3,RA4)
DECs=(DEC1,DEC2,DEC3,DEC4)

########################################################
cat_dir = '/Users/jia/weaklensing/CFHTLenS/catalogue/'
#textfile = genfromtxt(cat_dir+'CFHTLenS_downloads/CFHTLens_2015-02-25T04-54-07.tsv')#374M
#save(cat_dir+'CFHTLenS_downloads/All_RA_Dec_e12_w_z_m_c.npy', textfile.astype(float32))

textfile = load(cat_dir+'CFHTLenS_downloads/All_RA_Dec_e12_w_z_m_c.npy')[1:]
RA, DEC, e1, e2, weight, zB, m, c2 = textfile.T

####################################

def list2coords(radeclist, Wx):
	'''from radeclist for Wx to xy position in radians for specific map
	'''
	xy = zeros(shape = radeclist.shape)
	center = centers[Wx-1]
	f_Wx = WLanalysis.gnom_fun(center)
	xy = array(f_Wx(radeclist))
	#xy = array(map(f_Wx, radeclist))#in radians
	return xy

def construct_kmap (Wx, sigmaG=1.0, append = 'zcut13'):
	
	isize=sizes[Wx-1]
	iRA0, iRA1 = RAs[Wx-1]
	idx = where((RA<iRA1) & (RA>iRA0) & (zB<1.3))[0]
	
	print Wx, len(idx)
	
	iRA, iDEC, e1, e2, w, izB, m, c2 = textfile[idx].T
	e2 -= c2
	
	radeclist = array([iRA, iDEC]).T
	xy = list2coords(radeclist, Wx)
	y, x = xy
	
	k = array([e1*w, e2*w, (1+m)*w])

	A, galn = WLanalysis.coords2grid(x, y, k, size=isize)
	Me1, Me2, Mw = A
	Me1_smooth = WLanalysis.weighted_smooth(Me1, Mw, PPA=PPA512, sigmaG=sigmaG)
	Me2_smooth = WLanalysis.weighted_smooth(Me2, Mw, PPA=PPA512, sigmaG=sigmaG)
	kmap = WLanalysis.KSvw(Me1_smooth, Me2_smooth)
		
	save(cat_dir+'Me_Mw_galn/W%i_Me1w_%s.npy'%(Wx, append), A[0])
	save(cat_dir+'Me_Mw_galn/W%i_Me2w_%s.npy'%(Wx, append), A[1])
	save(cat_dir+'Me_Mw_galn/W%i_Mwm_%s.npy'%(Wx, append), A[2])
	save(cat_dir+'Me_Mw_galn/W%i_galn_%s.npy'%(Wx, append), galn)
	save(cat_dir+'kmap_W%i_sigma%02d_%s.npy'%(Wx, sigmaG*10, append), kmap)
	
map(construct_kmap, range(1,5))
#kmapGen = lambda Wx: load(cat_dir+'kmap_W%i_sigma10_noZcut.npy'%(Wx))

##########################
#### galn x cmb lensing ##
##########################
############# galaxies only ########
####textfile = genfromtxt(cat_dir+'CFHTLenS_downloads/CFHTLens_2015-02-26T02:54:49.tsv')
####save(cat_dir+'CFHTLenS_downloads/CFHTLens_2015-02-26T02:54:49.npy', textfile.astype(float32))

#galntextfile = load (cat_dir+'CFHTLenS_downloads/CFHTLens_2015-02-26T02:54:49.npy')[1:]
#RA, DEC, w, zB, MAG_i = galntextfile.T

def construct_galn (Wx, sigmaG=1.0):
	print Wx
	isize=sizes[Wx-1]
	iRA0, iRA1 = RAs[Wx-1]
	idx = where((RA<iRA1) & (RA>iRA0))[0]
	iRA, iDEC, iw, izB, iMAG_i = galntextfile[idx].T
	radeclist = array([iRA, iDEC]).T
	xy = list2coords(radeclist, Wx)
	y, x = xy
	k = array([w,])
	A, galn = WLanalysis.coords2grid(x, y, k, size=isize)
	galn_smooth = WLanalysis.smooth(galn, PPA512)
	save('/Users/jia/weaklensing/cmblensing/OmoriHolder_galn_W%i.npy'%(Wx),galn)
	save('/Users/jia/weaklensing/cmblensing/OmoriHolder_galnSmooth_W%i.npy'%(Wx),galn_smooth)
#map(construct_galn, range(1,5))