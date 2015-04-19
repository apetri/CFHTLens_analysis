#!python
# Jia Liu2015/02/15
# This code calculates the peak correlation function in 2 ways
# Cluster: XSEDE Stampede

import WLanalysis
import os
import numpy as np
from scipy import *
import scipy.ndimage as snd
import sys

############### constants #################

bins = 25
from emcee.utils import MPIPool
pool = MPIPool()
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
		ipeaklist =  concatenate([single_peaklist(i, cosmo, R, sigmaG) for i in i_arr], axis=-1)
		save(ipeaklist_fn(cosmo, R, sigmaG), ipeaklist)
		
		out_DDRR = zeros(len(edges)-1)
		out_Corr = zeros(len(edges)-1)
		
		for i in range(len(edges)-1):
			idx = where( (ipeaklist[0]>edges[i]) & (ipeaklist[0]<edges[i+1]))[0]
			out_DDRR[i] = len(idx)
			out_Corr[i] = mean(ipeaklist[1,idx]*ipeaklist[2,idx])
		save(fn, [out_Corr, out_DDRR])
	else:
		#print fn
		ierror = 0
		try:
			test = load(fn)
		except Exception:
			ierror = 1
			test = zeros(shape=(5,5))
		if test.shape[0] != 2 or test.shape[1] != 25 or ierror:
			print '!!! error', R, sigmaG, cosmo
			
			ipeaklist =  concatenate([single_peaklist(i, cosmo, R, sigmaG) for i in i_arr], axis=-1)
			save(ipeaklist_fn(cosmo, R, sigmaG), ipeaklist)
		
			out_DDRR = zeros(len(edges)-1)
			out_Corr = zeros(len(edges)-1)
			for i in range(len(edges)-1):
				idx = where( (ipeaklist[0]>edges[i]) & (ipeaklist[0]<edges[i+1]))[0]
				out_DDRR[i] = len(idx)
				out_Corr[i] = mean(ipeaklist[1,idx]*ipeaklist[2,idx])
			save(fn, [out_Corr, out_DDRR])
		else:
			out_Corr, out_DDRR = test
		
		print R, sigmaG, out_Corr.shape, out_DDRR.shape, cosmo
		
	return out_Corr, out_DDRR



cosmosigmaG_arr = [[cosmo, sigmaG] for sigmaG in sigmaG_arr[::-1] for cosmo in cosmo_arr]
def create_peakpeak_arr (cosmosigmaG):
	cosmo, sigmaG = cosmosigmaG
	fn_kappa = peaks_corr_dir+'sum/Corr_kappa_sigma%02d_%s.npy'%(sigmaG*10, cosmo)
	fn_counts = peaks_corr_dir+'sum/Corr_counts_sigma%02d_%s.npy'%(sigmaG*10, cosmo)
	if os.path.isfile(fn_kappa) and os.path.isfile(fn_counts):
		return None
	else:
		RcosmosigmaG = [[R, cosmo, sigmaG] for R in R_arr]
		peakpeak_arr = array(pool.map(single_corr, RcosmosigmaG))
		save(fn_kappa, peakpeak_arr[:,0,:])
		save(fn_counts, peakpeak_arr[:,1,:])
		return None

############# operation
#map(create_peakpeak_arr, cosmosigmaG_arr)

############ organize files

#cosmo_params = genfromtxt('/work/02977/jialiu/backup/cosmo_params.txt')
#im, iw, s8 = cosmo_params.T

#cosmo_arr = ['emu1-512b240_Om%.3f_Ol%.3f_w%.3f_ns0.960_si%.3f'%(im[i], 1-im[i], iw[i], s8[i]) for i in range(91)]

#fn_kappa = lambda sigmaG, cosmo: peaks_corr_dir+'sum/Corr_kappa_sigma%02d_%s.npy'%(sigmaG*10, cosmo)
#fn_counts =  lambda sigmaG, cosmo: peaks_corr_dir+'sum/Corr_counts_sigma%02d_%s.npy'%(sigmaG*10, cosmo)

#for sigmaG in sigmaG_arr:
	#all_kappa = array([[np.load(fn_kappa (sigmaG, cosmo))] for cosmo in cosmo_arr]).squeeze()
	#all_counts = array([[np.load(fn_counts (sigmaG, cosmo))] for cosmo in cosmo_arr]).squeeze()
	## array size (91, 1000, 50)
	#avg_kappa = mean(all_kappa, axis=1)
	#save(peaks_corr_dir+'peakpeak_kappa_avg_%02d.npy'%(sigmaG*10), avg_kappa)
	#avg_counts = mean(all_counts, axis=1)
	#save(peaks_corr_dir+'peakpeak_counts_avg_%02d.npy'%(sigmaG*10), avg_counts)
	
	
	
################# CFHT maps, local operation #######################
#def CFHT_single_peaklist(i, sigmaG):
	#'''for subfield=i.
	#return: distance(in pixel), kappa1, kappa2
	#'''
	#mask = WLanalysis.readFits('/Users/jia/CFHTLenS/catalogue/mask/CFHT_mask_ngal5_sigma%02d_subfield%02d.fits'%(sigmaG*10, i))
	#kmap = WLanalysis.readFits('/Users/jia/CFHTLenS/CFHTKS/CFHT_KS_sigma%02d_subfield%02d.fits'%(sigmaG*10, i))

	#peakmat = WLanalysis.peaks_mat(kmap)
	#loc = where((~isnan(peakmat))&(mask>0))#location of peaks
	#peaks = peakmat[loc]#kappa values at peaks

	####### get all combination of the 2 peaks ########
	#peakmesh = array(meshgrid(peaks, peaks))
	#peakcplx = peakmesh[0] + 1j*peakmesh[1]#make 2 peaks to complex number
	#peaksarr = concatenate([diag(peakcplx,j) for j in range(1,len(peaks)-1)])#only take upper right triangle
	####### the distance for all the peak pairs ########
	#iloc = loc[0] + 1j*loc[1]#turn x,y into X=x+iy, prepare for meshing
	#x, y = meshgrid(iloc,iloc)
	#distmat = abs(x-y)
	#dist = concatenate([diag(distmat,j) for j in range(1,len(peaks)-1)])
	#return dist, real(peaksarr),imag(peaksarr)

#def CFHT_single_corr(sigmaG, edges = edges):
	#'''for certain cosmology cosmo, realization R and smoothing sigmaG, return the correlations:
	#w_DDRR = (DD-2DR+RR)/RR
	#w_Corr = <kappa_i kappa_j> 
	#(combining 13 subfields).
	#'''	
	#ipeaklist =  concatenate([CFHT_single_peaklist(i, sigmaG) for i in i_arr], axis=-1)

	#out_DDRR = zeros(len(edges)-1)
	#out_Corr = zeros(len(edges)-1)

	#for i in range(len(edges)-1):
		#idx = where( (ipeaklist[0]>edges[i]) & (ipeaklist[0]<edges[i+1]))[0]
		#out_DDRR[i] = len(idx)
		#out_Corr[i] = mean(ipeaklist[1,idx]*ipeaklist[2,idx])
	#save('/Users/jia/weaklensing/CFHTLenS/peakpeak/Corr_kappa_sigma%02d_CFHT.npy'%(sigmaG*10), out_Corr)
	#save('/Users/jia/weaklensing/CFHTLenS/peakpeak/Corr_counts_sigma%02d_CFHT.npy'%(sigmaG*10), out_DDRR)
		
	#return out_Corr, out_DDRR

#a = map (CFHT_single_corr, sigmaG_arr)
######## plot results
#edges = linspace (5, 512, bins+1)
#center=WLanalysis.edge2center(edges)*PPA512
#plot_dir = '/Users/jia/Documents/weaklensing/CFHTLenS/peakpeak/plot/'
#for sigmaG in (1.0, 1.8, 3.5, 5.3):
	
	#CorrKappa_CFHT = load('/Users/jia/weaklensing/CFHTLenS/peakpeak/Corr_kappa_sigma%02d_CFHT.npy'%(sigmaG*10)) 
	
	#DDRR_CFHT = load('/Users/jia/weaklensing/CFHTLenS/peakpeak/Corr_counts_sigma%02d_CFHT.npy'%(sigmaG*10))
	
	#CorrKappa = load('/Users/jia/weaklensing/CFHTLenS/peakpeak/peakpeak_kappa_avg_%02d.npy'%(sigmaG*10))
	
	#DDRR = load('/Users/jia/weaklensing/CFHTLenS/peakpeak/peakpeak_counts_avg_%02d.npy'%(sigmaG*10))
	
	#subplot(211)
	#for i in range(91):
		#plot(center, CorrKappa[i], color=rand(3)) 
	#plot(center, CorrKappa_CFHT, 'k.')
	#subplot(212)
	#for i in range(91):
		#plot(center, DDRR[i], color=rand(3))
	#plot(center, DDRR_CFHT, 'k.')
	#savefig(plot_dir+'plot_dir_sigma%02d.jpg'%(sigmaG*10))
	#close()
	

print 'Done-Done-Done-Done!!!'