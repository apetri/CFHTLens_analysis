#!python
# Jia Liu2015/02/15
# This code calculates the peak correlation function in 2 ways
# Cluster: XSEDE Stampede

import WLanalysis
from emcee.utils import MPIPool
import os
import numpy as np
from scipy import *
import scipy.ndimage as snd
import sys

############### constants #################
bins = 25
KS_dir = '/work/02977/jialiu/KSsim/'
sim_dir = '/home1/02977/jialiu/cat/'
peaks_corr_dir = '/work/02977/jialiu/peaks_corr/'
backup_dir = '/work/02977/jialiu/backup/'
cosmo_arr = os.listdir(sim_dir)
params = genfromtxt(backup_dir+'cosmo_params.txt')
sigmaG_arr = array([1, 1.8, 3.5, 5.3])#, 8.9])#0.5,
R_arr = arange(1,1001)
PPA512 = 2.4633625
i_arr = range(1,14)
#edges = logspace(log10(5),log10(512*sqrt(2)),bins+1)
edges = linspace (5, 512, bins+1)

kmapGen = lambda i, cosmo, R, sigmaG: np.load(KS_dir+'%s/subfield%i/sigma%02d/SIM_KS_sigma%02d_subfield%i_%s_%04dr.npy'%(cosmo, i, sigmaG*10, sigmaG*10, i, cosmo,R))

maskGen = lambda i, sigmaG: WLanalysis.readFits(backup_dir+'mask/CFHT_mask_ngal5_sigma%02d_subfield%02d.fits'%(sigmaG*10, i))

########## bash command to create directories #########
####cd /home1/02977/jialiu/cat/
####for i in *
####do for sigmaG in 05 10 18 35 53 89
####do mkdir -pv -m 750 /work/02977/jialiu/peaks_corr_single/${i}/sigma${sigmaG}
####done
####done
########################################################

ipeaklist_fn = lambda cosmo, R, sigmaG: peaks_corr_dir+'peaks_corr_single/%s/sigma%02d/Peaklist_sigma%02d_%s_%04dr.npy'%(cosmo, sigmaG*10, sigmaG*10, cosmo, R)

icorr_fn = lambda cosmo, R, sigmaG: peaks_corr_dir+'peaks_corr_single/%s/sigma%02d/Corrlist_sigma%02d_%s_%04dr.npy'%(cosmo, sigmaG*10, sigmaG*10, cosmo, R)

mask_all_arr = array([[maskGen(i, sigmaG) for sigmaG in sigmaG_arr]for i in i_arr])
#i, R, cosmo, sigmaG = 1, 99, cosmo_arr[5], 8.9
def single_peaklist(i, cosmo, R, sigmaG):
	'''for subfield=i, cosmo, realization=R, smoothing scale sigmaG, return the 2 correlation functions.
	return: distance(in pixel), kappa1, kappa2
	'''
	mask = mask_all_arr[i-1, int(where(sigmaG_arr==sigmaG)[0])]
	kmap = kmapGen(i, cosmo, R, sigmaG)
	
	peakmat = WLanalysis.peaks_mat(kmap)
	loc = where((~isnan(peakmat))&(mask>0))#location of peaks
	peaks = peakmat[loc]#kappa values at peaks
	
	###### get all combination of the 2 peaks ########
	peakmesh = array(meshgrid(peaks, peaks))
	peakcplx = peakmesh[0] + 1j*peakmesh[1]#make 2 peaks to complex number
	peaksarr = concatenate([diag(peakcplx,j) for j in range(1,len(peaks)-1)])#only take upper right triangle
	###### the distance for all the peak pairs ########
	iloc = loc[0] + 1j*loc[1]#turn x,y into X=x+iy, prepare for meshing
	x, y = meshgrid(iloc,iloc)
	distmat = abs(x-y)
	dist = concatenate([diag(distmat,j) for j in range(1,len(peaks)-1)])
	return dist, real(peaksarr),imag(peaksarr)

def single_corr(iRcosmosigmaG, edges = edges):
	'''for certain cosmology cosmo, realization R and smoothing sigmaG, return the correlations:
	w_DDRR = (DD-2DR+RR)/RR
	w_Corr = <kappa_i kappa_j> 
	(combining 13 subfields).
	'''	
	R, cosmo, sigmaG = iRcosmosigmaG
	
	fn = icorr_fn(cosmo, R, sigmaG)
	if not os.path.isfile(fn):
		if not ipeaklist_fn(cosmo, R, sigmaG):
			ipeaklist =  concatenate([single_peaklist(i, cosmo, R, sigmaG) for i in i_arr], axis=-1)
			save(fn, ipeaklist)
		else:
			ipeaklist = load(ipeaklist_fn(cosmo, R, sigmaG))
		out_DDRR = zeros(len(edges)-1)
		out_Corr = zeros(len(edges)-1)
		for i in range(len(edges)-1):
			idx = where( (ipeaklist[0]>edges[i]) & (ipeaklist[0]<edges[i+1]))[0]
			out_DDRR[i] = len(idx)
			out_Corr[i] = mean(ipeaklist[1,idx]*ipeaklist[2,idx])
		save(fn, [out_Corr, out_DDRR])
	else:
		out_Corr, out_DDRR = load(fn)
	#fn = ipeaklist_fn(cosmo, R, sigmaG)
	#if not os.path.isfile(fn):
		#ipeaklist =  concatenate([single_peaklist(i, cosmo, R, sigmaG) for i in i_arr], axis=-1)
		#save(fn, ipeaklist)
	#else:
		#ipeaklist = load(fn)
	#out_DDRR = zeros(len(edges)-1)
	#out_Corr = zeros(len(edges)-1)
	#for i in range(len(edges)-1):
		#idx = where( (ipeaklist[0]>edges[i]) & (ipeaklist[0]<edges[i+1]))[0]
		#out_DDRR[i] = len(idx)
		#out_Corr[i] = mean(ipeaklist[1,idx]*ipeaklist[2,idx])
	print R, sigmaG, out_Corr.shape, out_DDRR.shape, cosmo
	return out_Corr, out_DDRR

pool = MPIPool()

cosmosigmaG_arr = [[cosmo, sigmaG] for sigmaG in sigmaG_arr[::-1] for cosmo in cosmo_arr]
def create_peakpeak_arr (cosmosigmaG):
	cosmo, sigmaG = cosmosigmaG
	fn_kappa = peaks_corr_dir+'sum/Corr_kappa_sigma%02d_%s.npy'%(sigmaG*10, cosmo)
	fn_counts = peaks_corr_dir+'sum/Corr_counts_sigma%02d_%s.npy'%(sigmaG*10, cosmo)
	if os.path.isfile(fn_kappa) and os.path.isfile(fn_counts):
		return None
	else:
		RcosmosigmaG = [[R, cosmo, sigmaG] for R in R_arr]
		peakpeak_arr = array(map(single_corr, RcosmosigmaG))
		save(fn_kappa, peakpeak_arr[:,0,:])
		save(fn_counts, peakpeak_arr[:,1,:])
		return None
pool.map(create_peakpeak_arr, cosmosigmaG_arr)

#for cosmo in cosmo_arr:
	#for sigmaG in (1.0,):#sigmaG_arr:
		##print cosmo, sigmaG
		#fn_kappa = peaks_corr_dir+'sum/Corr_kappa_sigma%02d_%s.npy'%(sigmaG*10, cosmo)
		#fn_counts = peaks_corr_dir+'sum/Corr_counts_sigma%02d_%s.npy'%(sigmaG*10, cosmo)
		#if os.path.isfile(fn_kappa) and os.path.isfile(fn_counts):
			#continue
		#else:
			#RcosmosigmaG = [[R, cosmo, sigmaG] for R in R_arr]
			#peakpeak_arr = array(pool.map(single_corr, RcosmosigmaG))#1000 x 2 x 25
			#save(fn_kappa, peakpeak_arr[:,0,:])
			#save(fn_counts, peakpeak_arr[:,1,:])
print 'Done-Done-Done-Done!!!'