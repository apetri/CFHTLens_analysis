#!~/anaconda/bin/python
# Jia Liu 2014/5/21
# What the code does: create mass maps for 100 cosmologies, for the CFHT emulator project
# Cluster: XSEDE Stampede

import WLanalysis
from emcee.utils import MPIPool
import os
import numpy as np
from scipy import *
import scipy.ndimage as snd
import sys

########## define constants ############
sim_dir = '/home1/02977/jialiu/cat/'
KS_dir = '/scratch/02977/jialiu/KSsim/'
cosmo_arr = os.listdir(sim_dir)
params = genfromtxt(KS_dir+'cosmo_params.txt')

i_arr=arange(1,14)
R_arr=arange(1,1001)

SIMfn = lambda i, cosmo, R: sim_dir+'%s/emulator_subfield%i_WL-only_%s_4096xy_%04dr.fit'%(cosmo, i, cosmo, R)

KSfn = lambda i, cosmo, R, sigmaG: KS_dir+'%s/SIM_KS_sigma%02d_subfield%i_%s_%04dr.fit'%(cosmo, sigmaG*10, i, cosmo,R)


def fileGen(i, R, cosmo):
	'''
	Input:
	i: subfield range from (1, 2..13)
	R: realization range from (1..1000)
	cosmo: one of the 100 cosmos
	Return:
	Me1 = e1*w
	Me2 = e2*w
	Mw = w (1+m)
	'''
	Mw_fn = KS_dir+'SIM_Mw_subfield%i.fit'%(i) # same for all R
	galn_fn = KS_dir+'SIM_galn_subfield%i.fit'%(i) # same for all R
	Mw = WLanalysis.readFits(Mw_fn)

	y, x, e1, e2, w, m = (WLanalysis.readFits(KS_dir+'yxewm_subfield%i_zcut0213.fit'%(i))).T
	k, s1, s2 = (WLanalysis.readFits(SIMfn(i,cosmo,R)).T)[[0,1,2]]
	s1 *= (1+m)
	s2 *= (1+m)
	eint1, eint2 = WLanalysis.rndrot(e1, e2, iseed=R)#random rotation	
	## get reduced shear
	e1, e2 = WLanalysis.eobs_fun(s1, s2, k, eint1, eint2)

	print 'coords2grid', i, R, cosmo
	A, galn = WLanalysis.coords2grid(x, y, array([k, e1*w, e2*w]))
	Mk, Ms1, Ms2 = A

	### add Mk just for comparison ###
	Mk_fn = KS_dir+'SIM_Mk/%s/SIM_Mk_subfield%i_%s_%04dr.fit'%(cosmo, i, cosmo, R)
	try:
		WLanalysis.writeFits(Mk, Mk_fn)
	except Exception:
		print 'file already exist, but no worries'
		pass
		
	return Ms1, Ms2, Mw



