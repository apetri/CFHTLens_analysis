# Jia Liu 09/25/2014
# This code tests MF covariance and interpolation
# using results run by A. Petri

import numpy as np
from scipy import *
import scipy.optimize as op
from scipy import interpolate#,stats
import os
import WLanalysis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import matplotlib.gridspec as gridspec 
import scipy.ndimage as snd

############# knobs ############################
plot_all_cov = 0
plot_fidu_obs_MF = 0
plot_contours = 0
#############################################
#50 bins between kappa=-0.15 and kappa=0.15
plot_dir = '/Users/jia/weaklensing/CFHTLenS/plot/MF/'
#params =  genfromtxt('/Users/jia/CFHTLenS/emulator/cosmo_params.txt')

################### stampede ################
params = genfromtxt('/scratch/02977/jialiu/KSsim/cosmo_params.txt')

loadMF = lambda Om, w, si8, i: np.load('/scratch/02918/apetri/Storage/wl/features/CFHT/cfht_masked_BAD/Om%.3f_Ol%.3f_w%.3f_ns0.960_si%.3f/subfield%i/sigma10/minkowski_all.npy'%(Om,1-Om,w,si8,i))

loadMF_cov = lambda i: np.load('/scratch/02918/apetri/Storage/wl/features/CFHT/cfht_masked_BAD/Om0.260_Ol0.740_w-1.000_ns0.960_si0.800_cov/subfield%i/sigma10/minkowski_all.npy'%(i))

loadMF_obs = lambda i: np.load('/scratch/02918/apetri/Storage/wl/features/CFHT/cfht_masked_BAD/observations/subfield%i/sigma10/minkowski_all.npy'%(i))

fsky = array([0.800968170166,0.639133453369,0.686164855957,0.553855895996,
		0.600227355957,0.527587890625,0.671237945557,0.494361877441,
		0.565235137939,0.592998504639,0.584747314453,0.530345916748,
		0.417697906494])
fsky_all = array([0.839298248291,0.865875244141,0.809467315674,
		  0.864688873291,0.679264068604,0.756385803223,
		  0.765892028809,0.747268676758,0.77250289917,
		  0.761451721191,0.691867828369,0.711254119873,
		  0.745429992676])
### 91 params
def sumMF (iparam, loadcov=False):
	print iparam
	Om, w, si8 = iparam
	if loadcov:
		iMF = array([loadMF_cov(i) for i in range(1,14)])
		#iMF = zeros(shape=(13,1000,150))
		#for i in range(1,14):
			#iMF[i] = loadMF_cov(i)
	else:
		iMF = array([loadMF(Om, w, si8, i) for i in range(1,14)])
	isumMF = sum(fsky.reshape(13,1,1)*iMF,axis=0)/float(sum(fsky))
	return isumMF

### organize files on stampede, and then need to download 
### to local for further analysis

isumMF_cov = sumMF(iparam,loadcov=1)
np.save('/home1/02977/jialiu/KSsim/MF_sum/MF_sum_cov',isumMF_cov)

#all_MF = array(map(sumMF, params))#shape=(91,1000,150)
#np.save('/home1/02977/jialiu/KSsim/MF_sum/MF_sum_91cosmo',all_MF.reshape(91,-1))
##########################################################

Mat_MF = np.load('/Users/jia/CFHTLenS/emulator/MF/MF_sum_91cosmo.npy').reshape(91,1000,-1)
avg_MF = mean(Mat_MF,axis=1)
obs_MF = np.load('/Users/jia/CFHTLenS/emulator/MF/obs.npy')

spline_interps = WLanalysis.buildInterpolator(avg_MF,params)

def interp_cosmo (params, spline_interps = spline_interps):
	'''Interpolate the MF for certain param.
	Params: list of 3 parameters = (om, w, si8)
	'''	
	im, wm, sm = params
	gen_MF = lambda ibin: spline_interps[ibin](im, wm, sm)
	MF_interp = array(map(gen_MF, range(len(spline_interps))))
	MF_interp = MF_interp.reshape(-1,1).squeeze()
	return MF_interp

cov_mat = cov(Mat_MF[48], rowvar=0)

def probMap (obs, cov_mat, spline_interps, ms, ss):
	'''return a probability map for observation'''
	cov_inv = mat(cov_mat).I
	xsize=len(ms)
	ysize=len(ss)
	heatmap = zeros(shape=(xsize,ysize))
	for i in range(xsize):
		for j in range(ysize):
			MF_interp = interp_cosmo((ms[i],-1,ss[j]), spline_interps = spline_interps)	
			del_N = np.mat(MF_interp - obs)
			chisq = float(del_N*cov_inv*del_N.T)
			heatmap[i,j] = chisq
	P = exp(-heatmap/2)
	P /= sum(P)
	return P
######### minimizer, or bruth force? #############


################################################
################## plotting routines ###########

def drawcontour2D(H, V, xvalues, yvalues, ititle, iylabel='simga_8', ixlabel='omega_m'):
	X, Y = np.meshgrid(xvalues, yvalues)
	figure(figsize=(6,8))
	im=imshow(H.T, interpolation='bilinear', origin='lower', cmap=cm.gray, extent=(xvalues[0], xvalues[-1], yvalues[0], yvalues[-1]))
	CS=plt.contour(X, Y, H.T, levels=V, origin='lower', extent=(xvalues[0], xvalues[-1], yvalues[0], yvalues[-1]), colors=('y', 'm', 'green'), linewidths=2)
	xlabel(ixlabel)
	ylabel(iylabel)
	title(ititle)
	savefig(plot_dir+ititle+'.jpg')
	close()

if plot_contours:
	idx = range(50)	
	ms = linspace(0.2,0.8,40)
	ss = linspace(0.2,1.0,41)
	iobs = obs_MF[idx]
	icov_mat = cov(Mat_MF[48][:,idx], rowvar=0)
	ispline_interps = array(spline_interps)[idx]
	P = probMap(iobs, icov_mat, ispline_interps, ms, ss)
	V = WLanalysis.findlevel(P)
	drawcontour2D(P, V, ms, ss, 'MF0_50bins')
	
	
def plotcov (Cov, ititle):
	X, Y = meshgrid(diag(Cov),diag(Cov))	
	Corr = Cov/sqrt(X*Y)
	J, K = meshgrid(avg_MF[48],avg_MF[48])
	Cov_rel = Cov/sqrt(abs(J*K))
	
	figure(figsize=(10,10))
	subplot(221)
	imshow(Cov,interpolation='nearest')#,origin='lower')
	colorbar(format='%.1e')
	title(ititle+'_Cov')
	
	subplot(222)
	imshow(Cov_rel,interpolation='nearest')#,origin='lower')
	title(ititle+'_Cov_rel')
	colorbar()
	
	subplot(223)
	imshow(Corr,interpolation='nearest')#,origin='lower')
	colorbar()
	title(ititle+'_Corr')
		
	savefig(plot_dir+'%s.jpg'%(ititle))
	close()	

if plot_all_cov:
	cov_arr = (cov_mat, cov_mat_m, cov_mat_w, cov_mat_s)
	title_arr = ('mat_1000r', 'mat_om', 'mat_w', 'mat_si8')
	mat_mf_m = array([interp_cosmo ((m, -1.0, 0.80)) for m in linspace(0.2, 0.8, 100)])
	mat_mf_w = array([interp_cosmo ((0.26, w, 0.80)) for w in linspace(-2.2,-0.4,100)])
	mat_mf_s = array([interp_cosmo ((0.26, -1.0, s)) for s in linspace(0.2, 0.8, 100)])
	cov_mat_m = cov(mat_mf_m, rowvar=0)
	cov_mat_w = cov(mat_mf_w, rowvar=0)
	cov_mat_s = cov(mat_mf_s, rowvar=0)
	for i in range(4):
		print i
		plotcov(cov_arr[i], title_arr[i])

if plot_fidu_obs_MF:
	fidu_MF = interp_cosmo ((0.26, -1.0, 0.80))

	figure(figsize=(8,6))
	subplot(311)
	plot(obs_MF, '--', label='observation')
	plot(fidu_MF, '-', label='fiducial')
	legend(fontsize=12)
	ylabel('MF')

	subplot(312)
	#plot((obs_MF-fidu_MF)/diag(cov_mat), '--')
	plot(obs_MF/fidu_MF-1, '--')
	plot(zeros(150),'k-')
	ylabel('obs_MF/fidu_MF-1')

	subplot(313)
	plot((obs_MF-fidu_MF)/diag(cov_mat), '--')
	plot(zeros(150),'k-')
	ylabel('(obs_MF-fidu_MF)/diag(cov)')
	#plt.subplots_adjust(hspace=0.0)
	savefig(plot_dir+'MF_fidu_obs.jpg')
	close()
