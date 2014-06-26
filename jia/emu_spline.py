# Jia Liu 06/05/2014
# This code uses spline interpolation for CFHT emulator
# currently only work with Jia's laptop

import numpy as np
from scipy import *
import scipy.optimize as op
import emcee
from scipy import interpolate,stats
import os
import WLanalysis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import matplotlib.gridspec as gridspec 
from sklearn.gaussian_process import GaussianProcess
from scipy.spatial import cKDTree
import scipy.ndimage as snd

emu_dir = '/Users/jia/CFHTLenS/emulator/'
sigmaG=1.8#1.0
# First, read in the 91 cosmology power spectrum
cosmo_params =  genfromtxt('/Users/jia/CFHTLenS/emulator/cosmo_params.txt')
def getps (cosmo_param):
	om, w, si8 = cosmo_param
	print cosmo_param
	ps_fn = 'SIM_powspec_sigma%02d_emu1-512b240_Om%.3f_Ol%.3f_w%.3f_ns0.960_si%.3f.fit'%(sigmaG*10, om,1-om,w,si8)
	ps = WLanalysis.readFits(emu_dir+'powspec_sum/sigma%02d/'%(sigmaG*10)+ps_fn)
	return ps
############# mistake!!! stampede and mine is not the same order!!!!######
####ps_fn_arr = os.listdir(emu_dir+'powspec_sum/sigma05/')
####getps = lambda ps_fn: WLanalysis.readFits(emu_dir+'powspec_sum/sigma05/'+ps_fn)
####ps_mat = array(map(getps, ps_fn_arr))[:,:,11:] # array [91, 1000, 50], changed to [91, 1000, 39]
##########################################################################

#ps_mat = array(map(getps, cosmo_params))[:,:,11:]
#WLanalysis.writeFits(ps_mat.reshape(91,-1),emu_dir+'powspec_sum/ps_mat_sigma05.fit')

ps_mat = WLanalysis.readFits(emu_dir+'powspec_sum/ps_mat_sigma05.fit').reshape(91,1000,-1)

ps_std = std(ps_mat, axis=1)# array [91, 50]
ps_stdlog = log10(ps_std)
ps_fidu = WLanalysis.readFits('/Users/jia/CFHTLenS/KSsim/powspec_sum13fields/SIM_powspec_sigma05_rz1_mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800_1000R.fit')[:,11:]
fidu_avg = mean(ps_fidu,axis=0)
fidu_std = std(ps_fidu,axis=0)
cov_mat = mat(cov(ps_fidu,rowvar=0))
cov_inv = cov_mat.I
####### try interpolate in log space ######
#ps_mat = log10(ps_mat)
###########################################
ps_avg = mean(ps_mat,axis=1) # array [91, 50]
ps_CFHT = WLanalysis.readFits('/Users/jia/CFHTLenS/KSsim/powspec_sum13fields/CFHT_powspec_sigma05.fit')[11:]
gs = gridspec.GridSpec(2,1,height_ratios=[3,1]) 
ell_arr = logspace(log10(110.01746692),log10(25207.90813028),50)[11:]
############### peaks ###################
x = linspace(-0.04, 0.12, 26)
x = x[:-1]+0.5*(x[1]-x[0])

def getpk (cosmo_param, sigmaG=1.0, bins = 25):
	om, w, si8 = cosmo_param
	pk_fn = 'SIM_peaks_sigma%02d_emu1-512b240_Om%.3f_Ol%.3f_w%.3f_ns0.960_si%.3f_600bins.fit'%(sigmaG*10, om,1-om,w,si8)
	pk600bins = WLanalysis.readFits(emu_dir+'peaks_sum/sigma%02d/'%(sigmaG*10)+pk_fn)
	pk = pk600bins.reshape(1000, -1, 600/bins)
	pk = sum(pk, axis = -1)
	return pk

########## wrong wrong wrong!!! ########
###pk_fn_arr = os.listdir(emu_dir+'peaks_sum/sigma%02d/'%(sigmaG*10))
###pk_mat = array(map(getpk, cosmo_params))	
###WLanalysis.writeFits(pk_mat.reshape(91,-1),emu_dir+'peaks_sum/pk_mat_sigma%02d_25bins.fit'%(sigmaG*10))
#############################################


######## knobs ##############
ps_replaced_by_good = 0
ps_replaced_by_nicaea = 0
ps_replaced_with_pk = 0# use the same plotting routine wrote for powspec to to peaks, simply make ps_mat = pk_mat
test_interp_method = 0#this is using spline
draw2Dplane = 0
test_gp = 0
test_pca_ncomp = 0
project_sims_3D = 0
check_bad_ps = 0
bad_KSmap = 0
check_ps_sum = 0
test_MCMC = 0
peaks_13subfield_sum = 0
try_mask_powspec = 0
check_shear_bad_ps_kmap = 0

if ps_replaced_by_good:
	bad_arr = array([6,14,24,27,31,32,33,38,42,43,44,45,53,54,55,61,63,64,65,66,67,72,74,75, 76,81,82,83,84,88,89])
	good_arr = delete(arange(91), bad_arr)
	#ps_avg = ps_avg[good_arr]
if ps_replaced_by_nicaea:
	P_ell_noise_arr = WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/nicaea_params/P_kappa_noise_arr91.fit')
	ps_avg = P_ell_noise_arr[:,11:]
if ps_replaced_with_pk:
	print 'ps_replaced_with_pk'
	pk_mat = (WLanalysis.readFits(emu_dir+'peaks_sum/pk_mat_sigma%02d_25bins.fit'%(sigmaG*10))).reshape(91,1000,-1)
	pk_avg = mean(pk_mat,axis=1)
	pk_std = std(pk_mat, axis=1)
	ps_mat = pk_mat
	ps_avg = pk_avg
	ps_std = pk_std
	ell_arr = x
	ps_fidu = WLanalysis.readFits('/Users/jia/CFHTLenS/KSsim/peaks_sum13fields/SIM_peaks_sigma10_rz1_mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800_1000R_025bins.fit')
	fidu_avg = mean(ps_fidu,axis=0)
	fidu_std = std(ps_fidu,axis=0)
	cov_mat = mat(cov(ps_fidu,rowvar=0))
	cov_inv = cov_mat.I
	ps_CFHT = WLanalysis.readFits('/Users/jia/CFHTLenS/KSsim/peaks_sum13fields/CFHT_peaks_sigma%i_25bins.fits'%(sigmaG*10))
# Second interpolate for individual bins
# interpolate.griddata(points, values, xi, method='linear/cubic/nearest')
# points = params, values = individual bin
# build 3D spline with scipy.interpolate.splprep

def test_interp_cosmo(i, ifcn='multiquadric', smooth=0):
	params_missing = cosmo_params[i]
	ps_missing = ps_avg[i]
	if ps_replaced_by_good:
		irm = concatenate([[i,],bad_arr])
	else:
		irm = i
	params_model = delete(cosmo_params, irm, 0)
	#print len(params_model)
	m, w, s = params_model.T
	im, wm, sm = params_missing
	def gen_ps (ibin):
		ps_model = delete(ps_avg, irm, 0)[:,ibin]
		#ps_interp = interpolate.griddata(params_model, ps_model, array([params_missing,]), method = 'nearest')
		ps_interp = interpolate.Rbf(m, w, s, ps_model, function=ifcn, smooth=smooth)(im, wm, sm)
		return ps_interp
	ps_interp = array(map(gen_ps, range(ps_avg.shape[-1]))).squeeze()
	#print ibin, ps_interp/ps_missing-1
	return ps_missing, ps_interp

def plotimshow(img,ititle,vmin=None,vmax=None):		 
	 #if vmin == None and vmax == None:
	imgnonzero=img[nonzero(img)]
	if vmin == None:
		std0 = std(imgnonzero)
		x0 = median(imgnonzero)
		vmin = x0-3*std0
		vmax = x0+3*std0
	im=imshow(img,interpolation='nearest',origin='lower',aspect=1,vmin=vmin,vmax=vmax)
	colorbar()
	title(ititle,fontsize=16)
	savefig(plot_dir+'%s.jpg'%(ititle))
	close()	
	
plot_dir = '/Users/jia/weaklensing/CFHTLenS/plot/'
ps_avg_all = mean(ps_avg, axis=0)

if test_interp_method:
	ismooth=0
	if ps_replaced_by_good:
		irange = good_arr
	else:
		irange = range(91)
	for i in irange:
		print 'Rbs',i
		lss = ('-.','.','--','*','d','+','1','2','3')
		gs = gridspec.GridSpec(2,1,height_ratios=[3,1]) 
		f=figure(figsize=(8,8))
		ax=f.add_subplot(gs[0])
		ax2=f.add_subplot(gs[1],sharex=ax)
		#ax=subplot(111)
		
		k=0
		ax2.plot(ell_arr, zeros(len(ell_arr)),'k-')
		
		for ifcn in ('multiquadric','inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'):
			a=test_interp_cosmo(i, ifcn=ifcn, smooth=ismooth)
			ax.plot(ell_arr, a[1],lss[k],label='%s'%(ifcn))
			ax2.plot(ell_arr, a[1]/a[0]-1,lss[k])
			k+=1
		#ax2.plot(ell_arr, a[0]/ps_avg_all-1, 'b-', linewidth=2)
		#ax.errorbar(ell_arr, a[0], ps_std[i],label='actual',linewidth=2)
		ax.plot(ell_arr, a[0], label='actual',linewidth=2)
		if ps_replaced_by_good:
			ax.set_title('good #%i Smooth=%i, Rbs [%.3f,%.3f, %.3f]'%(i, ismooth, cosmo_params[i,0], cosmo_params[i,1], cosmo_params[i,2]))
		else:
			ax.set_title('cosmo #%i Smooth=%i, Rbs [%.3f,%.3f, %.3f]'%(i, ismooth, cosmo_params[i,0], cosmo_params[i,1], cosmo_params[i,2]))
		
		if not ps_replaced_with_pk:
			ax.set_xscale('log')
			ax.set_yscale('log')
			ax.set_xlabel('ell')
			ax.set_xlim(ell_arr[0],ell_arr[-1])
			ax.set_ylim(5e-5, 5e-2)
		
		ax2.set_ylim(-0.5,0.5)
		leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
		leg.get_frame().set_visible(False)
		plt.setp(ax.get_xticklabels(), visible=False) 
		plt.subplots_adjust(hspace=0.0)
		
		
		if ps_replaced_with_pk:
			savefig(plot_dir+'Rbs_peaks/Rbs_smooth%i_emu_peaks_cosmo%02d.jpg'%(ismooth,i))
		elif ps_replaced_by_nicaea:
			savefig(plot_dir+'Rbs_nicaea/Rbs_smooth%i_emu_nicaea_cosmo%02d.jpg'%(ismooth,i))
		elif ps_replaced_by_good:
			savefig(plot_dir+'Rbs_60good/Rbs_smooth%i_emu_good_cosmo%02d.jpg'%(ismooth,i))
		else:
			savefig(plot_dir+'RBs_reorder/Rbs_smooth0_emu_powspec_cosmo%02d.jpg'%(i))
		close()

# now plot out the interpolated plane vs actual points
m, w, s = cosmo_params.T
params_min = amin(cosmo_params,axis=0)
params_max = amax(cosmo_params,axis=0)

if draw2Dplane:
	ibin=20
	ismooth=0
	#X, Y = meshgrid(linspace(params_min[1],params_max[1],100), linspace(params_min[2],params_max[2],100))
	X, Y = meshgrid(linspace(params_min[0],params_max[0],100), linspace(params_min[2],params_max[2],100))
	w_dummy = -ones(shape=(100,100))
	s_dummy = 0.8*ones(shape=(100,100))
	m_dummy = 0.26*ones(shape=(100,100))
	for ibin in range(len(ell_arr)):
		ps = ps_avg.T[ibin]
		ps_interp = interpolate.Rbf(m, w, s, ps, smooth = ismooth)
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		#ax.scatter(m, s, ps)	
		Z = ps_interp(X,w_dummy,Y)
		#Z = ps_interp(m_dummy, X, Y)
		ax.plot_wireframe(X, Y, Z)#, rstride=10, cstride=10)
		ax.set_xlabel('Omega_m')
		ax.set_ylabel('sigma_8')
		#ax.set_xlabel('w')
		#ax.set_zlabel('power spectrum #%i bin'%(ibin))
		ax.set_title('power spectrum #%i bin smooth = 0'%(ibin))
		#show()
		savefig(plot_dir+'Rbs_nicaea/emu_Rbs_nicaea_smooth%i_powspec_%02dbin_w-100.jpg'%(ismooth,ibin))
		close()

#from multiprocessing import Pool
#p = Pool(91)
labels=('$\Omega_m$','$w$','$\sigma_8$')
if test_gp:
	if ps_replaced_with_pk:
		print 'ps_replaced_with_pk'
		ps_mat = pk_mat
		ps_avg = pk_avg
		ps_std = pk_std
		ell_arr = x
	seed(222)
	colors = rand(30,3)
	corr_arr = ('absolute_exponential', 'squared_exponential',
         'cubic', 'linear')#'generalized_exponential',
	def test_gp_cosmo (var):#(i, corr='squared_exponential'):
		i, corr = var
		print i, corr
		params_model = delete(cosmo_params, i, 0)
		#dy_model = delete(ps_std, i, 0)
		dy_model = delete(ps_std, i, 0)
		params_missing = cosmo_params[i]
		ps_missing = ps_avg[i]
		im, wm, sm = params_missing
		def gen_ps (ibin):
			y = delete(ps_avg, i, 0)[:,ibin]
			dy = dy_model[:, ibin]
			gp = GaussianProcess(corr=corr, nugget=(dy / y) ** 2, random_start=100)
			#gp = GaussianProcess(corr=corr, random_start=100)
			gp.fit(params_model, y)
			y_pred, MSE = gp.predict(params_missing, eval_MSE=True)
			return y_pred, MSE
		ps_interp, ps_MSE = (array(map(gen_ps, range(ps_avg.shape[-1]))).squeeze()).T
		#print ibin, ps_interp/ps_missing-1
		
		########## log to linear ####
		#ps_interp = 10**ps_interp
		#ps_MSE = 10**ps_MSE
		#ps_missing = 10**ps_missing
		#############################
		
		############### plotting ##########
		#f=figure(figsize=(8,8))
		#ax=f.add_subplot(gs[0])
		#ax2=f.add_subplot(gs[1],sharex=ax)
		#ax.errorbar(ell_arr, ps_missing, ps_std[i], color='b',label='actual powspec')
		#ax.errorbar(ell_arr, ps_interp, ps_MSE, color='r', label='GP fit')
		#ax2.plot(ell_arr, zeros(len(ell_arr)),'k-')
		#ax2.plot(ell_arr, ps_interp/ps_missing-1,color='r')
		

		
		#ax2.set_ylim(-0.5, 0.5)
		#leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
		#leg.get_frame().set_visible(False)
		#plt.setp(ax.get_xticklabels(), visible=False) 
		#plt.subplots_adjust(hspace=0.0)
		#ax.set_title('%s cosmo %i [%.3f,%.3f, %.3f]'%(corr, i, cosmo_params[i,0], cosmo_params[i,1], cosmo_params[i,2]))
		#ax.set_xlim(ell_arr[0],ell_arr[-1])
		#if not ps_replaced_with_pk:
			#ax2.set_xscale('log')
			#ax.set_xscale('log')
			#ax.set_yscale('log')
			#ax.set_ylim(1e-5, 1e-2)
			#savefig(plot_dir+'GP_test_%s_cosmo%i.jpg'%(corr,i))
		#else:
			#savefig(plot_dir+'GP_test_peaks/GP_test_%s_cosmo%i.jpg'%(corr,i))
		#close()
		########## plotting end #########
		
		return ps_missing, ps_interp, ps_MSE

	#a=test_gp_cosmo(20)
	#aa=map(test_gp_cosmo, range(91))
	var_arr = [[i, corr] for i in range(91) for corr in corr_arr]
	a=array(map(test_gp_cosmo, var_arr)).squeeze()###a.shape=(40, 3, 39)
	
	for j in arange(0,a.shape[0],4):
		print j
		colors = ('r','b','g','m','y','k')
		f=figure(figsize=(8,8))
		ax=f.add_subplot(gs[0])
		ax2=f.add_subplot(gs[1],sharex=ax)
		
		for k in arange(4):
			i, corr = var_arr[j+k]
			ps_missing, ps_interp, ps_MSE = a[j+k]
			if k==0:
				ax.errorbar(ell_arr, ps_missing, ps_std[i], color='k', label='actual powspec')
			#ax.errorbar(ell_arr, ps_interp, ps_MSE, color=colors[k], label='GP %s'%(corr))
			ax.plot(ell_arr, ps_interp, color=colors[k], label='GP %s'%(corr))
			ax2.plot(ell_arr, zeros(len(ell_arr)),'k-')
			ax2.plot(ell_arr, ps_interp/ps_missing-1, color=colors[k])
		
		
		ax2.set_ylim(-0.1, 0.1)
		leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
		leg.get_frame().set_visible(False)
		plt.setp(ax.get_xticklabels(), visible=False) 
		plt.subplots_adjust(hspace=0.0)
		ax.set_title('cosmo %i [%.3f,%.3f, %.3f]'%(i, cosmo_params[i,0], cosmo_params[i,1], cosmo_params[i,2]))
		ax.set_xlim(ell_arr[0],ell_arr[-1])
		
		if not ps_replaced_with_pk:
			ax2.set_xscale('log')
			ax.set_xscale('log')
			ax.set_yscale('log')
			ax.set_ylim(1e-5, 1e-2)
			savefig(plot_dir+'GP_test_powspec_reorder/GP_test_powspec_cosmo%02d.jpg'%(i))
		else:
			savefig(plot_dir+'GP_test_peaks_reorder/GP_test_cosmo%02d.jpg'%(i))
		
		close()


	#aa=a[1::4]
	#ps_missing = aa[:,0,:]
	#ps_interp = aa[:,1,:]
	#ps_diff = abs(ps_missing/ps_interp-1)
	#ps_diff_norm = ps_diff/ps_std
	#ps_MSE = aa[:,2,:]

	#for m in arange(21,39):#arange(39):
		#print m
		#z=ps_diff[:,m]
		#f=figure(figsize=(8,8))
		#for ij in [[0,1],[1,2],[2,0]]:
			#i,j=ij
			
			#ax=f.add_subplot(2,2,i+1)#, projection='3d')
			#param1, param2 = cosmo_params.T[ij]
			#ax.scatter(param1, param2, c=z, s=50, vmax=0.4, vmin=0)#, cmap='gray')
			##ax.colorbar()
			##big is white, small is blakc	
			#ax.set_xlabel(labels[i],fontsize=14)#,labelpad=0)
			#ax.set_ylabel(labels[j],fontsize=14,labelpad=0)
			#if i==0:
				#ax.set_title('bin #%i'%(m))
		#plt.subplots_adjust(wspace=0.25,hspace=0.25)
		#savefig(plot_dir+'goodness_fit_vmax04_linear_bin%i.jpg'%(m))
		#close()

	###### find relation between goodness of fit vs. average distance to N nearest neighbors
	#kdt = cKDTree(cosmo_params)
	#nearest_neighbors = kdt.query(cosmo_params, k=11)
	#nnd, nnidx = nearest_neighbors
	#nnd = nnd[:,1:] #0th is self
	#chisq_arr = (ps_missing-ps_interp)**2/ps_std**2
	##chisq_sumbins = sum(chisq_arr, axis=1)

	#for ibin in range(39):
		#f=figure(figsize=(12,8))
		#chisq = ps_diff_norm[:,ibin]#chisq_arr[:,ibin]
		##chisq = log10(ps_diff[:,ibin])
		##chisq = ps_diff[:,ibin]
		#for n in range(1,13):
			#d_avg = average(nnd[:,:n],axis=1)
			#slope, intercept, r_value, p_value, std_err = stats.linregress(d_avg,chisq)
			#x=array([amin(d_avg),amax(d_avg)])
			#y=intercept+slope*x
			
			#ax=f.add_subplot(3,4,n)
			#ax.scatter(d_avg, chisq, s=5)
			#ax.plot(x, y,'r-', linewidth=2, label='r=%.2f'%(r_value))
			##ax.set_title('N = %i'%(n),fontsize=12)
			#x0,x1=amin(d_avg)-0.02, amax(d_avg)+0.02
			#ax.set_xlim(x0,x1)
			#ax.set_xticks(arange(round(x0,1),x1,0.1))
			#ax.set_xlabel('Distance (N = %i)'%(n))
			#ax.set_ylabel('(fit-true)/true')
			
			#leg=ax.legend(loc=0, ncol=1, labelspacing=.2, prop={'size':10})
			#leg.get_frame().set_visible(False)
			
			#if n==1:
				#ax.set_title('%i bin'%(ibin))
		#plt.subplots_adjust(hspace=0.3,wspace=0.3,left=0.1,right=0.98)
		#savefig(plot_dir+"NN_chisq_psdiffnorm_%ibin.jpg"%(ibin))
		#close()
	################## finish KDTree
		
		
	

if test_pca_ncomp:	
	from sklearn.decomposition import PCA
	individual_comp = 0
	
	if individual_comp: 
		def pca_var(i):
			print i
			pca = PCA(n_components=30)
			ps = ps_mat[i]
			ps /= ps_std[i]#normalized each bin
			pspca = pca.fit(ps)
			variance = pca.explained_variance_ratio_
			return variance
		all_variance = array(map(pca_var, range(91)))
		
		f=figure(figsize=(8,8))
		ax=f.add_subplot(211)
		ax2=f.add_subplot(212,sharex=ax)
		seed(222)
		for i in range(91):
			cl=rand(3)
			ax.plot(range(30),all_variance[i],color=cl)
			ax2.plot(range(30),cumsum(all_variance[i]), color=cl)
		ax2.set_xlabel('Number of components')
		ax2.set_ylabel('Cumulative sum of variance')
		ax.set_yscale('log')
		ax2.set_yscale('log')
		ax.set_ylabel('Variance')
		#plt.setp(ax.get_xticklabels(), visible=False) 
		#plt.subplots_adjust(hspace=0.0)
		savefig(plot_dir+'PCA_components_normed.jpg')
		close()
	
	ps = mean(ps_mat,axis=0)#average of 91 cosmos to find PCA
	ps /= std(ps, axis=0)
	pca = PCA(n_components=30)
	pspca = pca.fit(ps)
	variance = pca.explained_variance_ratio_

	##### make plot #######
	#f=figure(figsize=(8,8))
	#ax=f.add_subplot(211)
	#ax2=f.add_subplot(212,sharex=ax)
	#ax.plot(range(30),variance)
	#ax2.plot(range(30),cumsum(variance))
	#ax2.set_xlabel('Number of components')
	#ax2.set_ylabel('Cumulative sum of variance')
	#ax.set_yscale('log')
	#ax.set_ylabel('Variance')
	#savefig(plot_dir+'PCA_components_normed_sum91.jpg')
	#close()
	testps = ps[0]
	testps_trans=pspca.transform(testps).ravel()
	plot(testps)
	plot(testps_trans)

if project_sims_3D:
	for i in range(39):
		
		labels = (('Omega_m','w'),('w','sigma8'),('sigma8','Omega_m'))
		k=0
		for pairs in ((m,w),(w,s),(s,m)):
			fig=figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.scatter(pairs[0], pairs[1], ps_avg[:,i])
			ax.set_xlabel(labels[k][0])
			ax.set_xlabel(labels[k][1])
			ax.set_title('%i bin'%(i))
			savefig(plot_dir+'ps_3D_%ibin_k%i.jpg'%(i,k))
			close()
			k+=1	

if check_bad_ps:
	if ps_replaced_with_pk:
		print 'ps_replaced_with_pk'
		ps_mat = pk_mat
		ps_avg = pk_avg
		ps_std = pk_std
		ell_arr = x
	#bad_arr = array([6,14,24,27,31,32,33,38,42,43,44,45,53,54,55,61,63,64,65,66,67,72,74,75, 76,81,82,83,84,88,89])	
	for i in range(91):
		print i
		seed(222)
		f=figure(figsize=(8,6))
		ax=f.add_subplot(111)
		isbad = 0
		for j in range(1000):
			ps = ps_mat[i,j]
			if mean(abs(ps-ps_avg[i])/5/ps_std[i])>1:
				ax.plot(ell_arr,ps,color=rand(3),label=str(j))	
				isbad = 1
			else:
				ax.plot(ell_arr,ps,color=rand(3))
		ax.errorbar(ell_arr,ps_avg[i],ps_std[i],color='k',linewidth=2)
		ax.set_title('cosmo %i [%.3f,%.3f, %.3f]'%(i, cosmo_params[i,0], cosmo_params[i,1], cosmo_params[i,2]))
		
		try:
			leg = ax.legend (ncol=3, labelspacing=0.3, prop={'size':12}, loc=0, title = 'ps-avg > 5 std')
			leg.get_frame().set_visible(False)
		except Exception:
			pass
		
		if not ps_replaced_with_pk:
			ax.set_xscale('log')
			ax.set_yscale('log')
			ax.set_xlabel('ell')
			ax.set_xlim(ell_arr[0],ell_arr[-1])
			#ax.set_ylim(5e-5, 5e-2)
			if isbad:
				savefig(plot_dir+'bad_powspec/sigmaG%02d_ps_mat_%i.jpg'%(sigmaG*10,i))
			else:
				savefig(plot_dir+'good_powspec/sigmaG%02d_ps_mat_%i.jpg'%(sigmaG*10,i))
		else:
			savefig(plot_dir+'pk_mat_%i.jpg'%(i))
		close()

galcount = array([342966,365597,322606,380838,
		263748,317088,344887,309647,
		333731,310101,273951,291234,
		308864]).astype(float)
galcount /= sum(galcount)
if bad_KSmap:
	#fn_arr = ('Om0.446_Ol0.554_w-1.212_ns0.960_si1.486_0114r',)#cosmo9
	#n=9
	#r = 114#but 113 in my arr
	fn_arr =('Om0.755_Ol0.245_w-0.456_ns0.960_si1.359_0553r',)
	n=22
	r=553
	for fn in fn_arr:
		summed_ps = ps_mat[9,r-1]#from previously calculated model
		print fn
		sf_ps_arr = zeros(shape=(13,50))
		for i in range(1,14):
			print 'subfield',i
			ifn = emu_dir+'debug_ps/kmap/SIM_KS_sigma05_subfield%i_emu1-512b240_%s.fit'%(i,fn)
			kmap = WLanalysis.readFits(ifn)
			plotimshow(kmap, fn+'sf%i'%(i))
			ips = WLanalysis.PowerSpectrum(kmap)[1]
			sf_ps_arr[i-1] = ips
		ax = subplot(111)
		for i in range(13):
			ax.plot(ell_arr, sf_ps_arr[i,11:], label='%i'%(i+1))
		summed_ps_sf = sum(sf_ps_arr*galcount.reshape(13,1),axis=0)
		ax.plot(ell_arr, summed_ps, '--', label='sum',linewidth=2)
		ax.plot(ell_arr, summed_ps_sf[11:], '.', label='recalc sum',linewidth=2)
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.set_xlabel('ell')
		ax.set_xlim(ell_arr[0],ell_arr[-1])
		ax.set_ylim(5e-5, 5e-2)
		leg=ax.legend(loc=0, ncol=2, labelspacing=0.3, prop={'size':12})
		leg.get_frame().set_visible(False)
		ax.set_title(fn)
		savefig(plot_dir+'debug_ps/bad_KSmap'+fn+'.jpg')
		close()
if check_ps_sum:
	#r_arr = [401,883]
	summed_ps = ps_mat[45, 780]
	gen_ps_mat = lambda i: WLanalysis.readFits(emu_dir+'debug_ps/SIM_powspec_sigma05_subfield%i_emu1-512b240_Om0.361_Ol0.639_w-0.606_ns0.960_si0.171_0780r.fit'%(i))
	fn='Om0.361_Ol0.639_w-0.606_ns0.960_si0.171_0780r'
	sf_ps_arr = array(map(gen_ps_mat, range(1,14)))
	
	ax = subplot(111)
	for i in range(13):
		ax.plot(ell_arr, sf_ps_arr[i,11:], label='%i'%(i+1))
	summed_ps_sf = sum(sf_ps_arr*galcount.reshape(13,1),axis=0)
	
	ax.plot(ell_arr, summed_ps, '--', label='sum',linewidth=2)
	ax.plot(ell_arr, summed_ps_sf[11:], '.', label='recalc sum',linewidth=2)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('ell')
	ax.set_xlim(ell_arr[0],ell_arr[-1])
	ax.set_ylim(5e-5, 5e-2)
	leg=ax.legend(loc=0, ncol=2, labelspacing=0.3, prop={'size':12})
	leg.get_frame().set_visible(False)
	ax.set_title(fn)
	savefig(plot_dir+'debug_ps/bad_KSmap'+fn+'_from_stampede.jpg')
	close()

######### build interpolator, a serious one (6/19/2014) ########
m, w, s = cosmo_params.T
spline_interps = list()
for ibin in range(ps_avg.shape[-1]):
	ps_model = ps_avg[:,ibin]
	iinterp = interpolate.Rbf(m, w, s, ps_model)
	spline_interps.append(iinterp)

gp_interps = list()
for ibin in range(ps_avg.shape[-1]):
	y = ps_avg[:,ibin]
	dy = ps_std[:,ibin]
	gp = GaussianProcess(corr='squared_exponential', nugget=(dy / y) ** 2, random_start=100)
	gp.fit(cosmo_params, y)
	gp_interps.append(gp)
	
def interp_cosmo (params, method = 'multiquadric'):
	'''Interpolate the powspec for certain param.
	Params: list of 3 parameters = (om, w, si8)
	Method: "multiquadric" for spline (default), and "GP" for Gaussian process.
	'''	
	im, wm, sm = params
	if method == 'multiquadric':
		gen_ps = lambda ibin: spline_interps[ibin](im, wm, sm)
		
	elif method == 'GP':
		gen_ps = lambda ibin: gp_interps[ibin].predict(params)
	ps_interp = array(map(gen_ps, range(ps_avg.shape[-1])))
	ps_interp = ps_interp.reshape(-1,1).squeeze()
	return ps_interp

if test_MCMC:

	# prior
	mmin,wmin,smin=amin(cosmo_params,axis=0)
	mmax,wmax,smax=amax(cosmo_params,axis=0)
	fidu_params = (0.26, -1, 0.8)
	
	steps = 1000
	burn = 100 # make sure burn < steps
	
	#obs = fidu_avg#ps_CFHT#
	#method = 'multiquadric'#'GP'#
	for obs in (fidu_avg,):#(fidu_avg, ps_CFHT):
		for method in ('GP',):#('multiquadric','GP'):
			print bool(obs[3]== fidu_avg[3]), method
			def lnprior(params):
				'''This gives the flat prior.
				Returns:
				0:	if the params are in the regions of interest.
				-inf:	otherwise.'''
				m, w, s = params
				if 0.0 < m < 1.2 and -4.0 < w < 1.0 and 0.1 < s < 1.6:
					return 0.0
				else:
					return -np.inf

			def lnlike (params, obs):
				'''This gives the likelihood function, assuming Gaussian distribution.
				Returns: -log(chi-sq)
				'''
				model = interp_cosmo (params, method = method)
				del_N = np.mat(model - obs)
				chisq = del_N*cov_inv*del_N.T
				Ln = -log(chisq) #likelihood, is the log of chisq
				return float(Ln)
			
			def lnprob(params, obs):
				lp = lnprior(params)
				if not np.isfinite(lp):
					return -np.inf
				else:
					return lp + lnlike(params, obs)
			
			nll = lambda *args: -lnlike(*args)
			result = op.minimize(nll, fidu_params, args=(obs,))
			print result
			
			ndim, nwalkers = 3, 100
			pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
			
			print 'run sampler'
			sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(obs,))

			print 'run mcmc'
			sampler.run_mcmc(pos, steps)
			samples = sampler.chain[:, burn:, :].reshape((-1, ndim))

			errors = array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84],axis=0))))
			print 'Result\t[best fits, lower error, upper error]'
			print 'Omega_m:\t', errors[0]
			print 'w:\t\t', errors[1]
			print 'sigma_8:\t', errors[2]
			
			import triangle	
			fig = triangle.corner(samples, labels=["$\Omega_m$", "$w$", "$\sigma_8$"],  plot_datapoints=False)#,truths=fidu_params)
			#fig.savefig(plot_dir+"emu_triangle_MCMC_fidu_%s.jpg"%(method))
			if ps_replaced_with_pk:
				print 'ps_replaced_with_pk'
				if obs[3] == fidu_avg[3]:
					fn = "emu_fidu_%s_peaks"%(method)
				elif obs[3] == ps_CFHT[3]:
					fn = "emu_CFHT_%s_peaks"%(method)
			else:
				if obs[3] == fidu_avg[3]:
					fn ='emu_fidu_%s'%(method)
				elif obs[3] == ps_CFHT[3]:
					fn ='emu_CFHT_%s'%(method)
			fig.savefig(plot_dir+'triangle_'+fn+'.jpg')
			close()
	

			best_fit = errors[:,0]
			ps_interp=interp_cosmo(best_fit,method=method)
		f=figure(figsize=(8,8))
		ax=f.add_subplot(gs[0])
		ax2=f.add_subplot(gs[1],sharex=ax)
		ax.errorbar(ell_arr,obs,fidu_std,color='r',label=fn)
		ax.plot(ell_arr,ps_interp,'b-',label=method)
		ax2.plot(ell_arr,zeros(len(ell_arr)),'r-')
		ax2.plot(ell_arr,ps_interp/obs-1,'b-')

		if not ps_replaced_with_pk:
			ax.set_xscale('log')
			ax2.set_xscale('log')
			ax.set_yscale('log')
			ax.set_xlabel('ell')
			ax.set_ylim(5e-5, 1e-2)
			ax.set_xlim(ell_arr[0],ell_arr[-1])

		ax2.set_ylim(-0.1, 0.1)
		leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
		leg.get_frame().set_visible(False)
		plt.setp(ax.get_xticklabels(), visible=False) 
		plt.subplots_adjust(hspace=0.0)
		savefig(plot_dir+'frac_diff'+fn+'.jpg')
		close()


#best_spline_params=(0.46875562, -1.52936207, 0.5170796 )
#best_GP_params=(0.46119148, -1.44506788, 0.51769554)
#fidu_avg=ps_CFHT
#ps_spline=interp_cosmo(best_spline_params)
#ps_GP=interp_cosmo(best_GP_params,method='GP')
#f=figure(figsize=(8,8))
#ax=f.add_subplot(gs[0])
#ax2=f.add_subplot(gs[1],sharex=ax)
#ax.plot(ell_arr,fidu_avg,'r-',label='CFHT')
#ax.plot(ell_arr,ps_spline,'b-',label='Spline')
#ax.plot(ell_arr,ps_GP,'m-',label='GP')
#ax2.plot(ell_arr,zeros(len(ell_arr)),'r-')
#ax2.plot(ell_arr,ps_spline/fidu_avg-1,'b-')
#ax2.plot(ell_arr,ps_GP/fidu_avg-1,'m-')

#ax.set_xscale('log')
#ax2.set_xscale('log')
#ax.set_yscale('log')
#ax.set_xlabel('ell')

#ax.set_xlim(ell_arr[0],ell_arr[-1])
#ax.set_ylim(5e-5, 1e-2)
#ax2.set_ylim(-0.1, 0.1)
#leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
#leg.get_frame().set_visible(False)
#plt.setp(ax.get_xticklabels(), visible=False) 
#plt.subplots_adjust(hspace=0.0)
#savefig(plot_dir+'test_MC_CFHT.jpg')
#close()

if peaks_13subfield_sum:
	pk_gen = lambda i: WLanalysis.readFits('/Users/jia/CFHTLenS/CFHTKS/CFHT_peaks_sigma%i_subfield%02d_025bins.fits'%(sigmaG*10, i))
	pk_mat = array(map(pk_gen,range(1,14)))
	pk_sum = sum(pk_mat, axis=0)
	WLanalysis.writeFits(pk_sum,'/Users/jia/CFHTLenS/KSsim/peaks_sum13fields/CFHT_peaks_sigma%i_25bins.fits'%(sigmaG*10))

if try_mask_powspec:
	kmap = WLanalysis.readFits(emu_dir+'debug_ps/kmap/SIM_KS_sigma05_subfield2_emu1-512b240_Om0.446_Ol0.554_w-1.212_ns0.960_si1.486_0114r.fit')
	mask = WLanalysis.readFits('/Users/jia/CFHTLenS/KSsim/mask/CFHT_mask_ngal5_sigma05_subfield02.fits')
	ps_nomask = WLanalysis.PowerSpectrum(kmap)[1][11:]
	ps_mask = WLanalysis.PowerSpectrum(kmap*mask)[1][11:]
	loglog(ell_arr,ps_nomask,label='no mask')
	loglog(ell_arr,ps_mask,label='mask')
	xlim(ell_arr[0],ell_arr[-1])
	legend()
	show()

#for i in (13,):
	#print i
	#mask = WLanalysis.readFits('/Users/jia/CFHTLenS/KSsim/mask/CFHT_mask_ngal5_sigma05_subfield%02d.fits'%(i))
	#plotimshow(mask,'mask_sf%02d'%(i),vmin=0,vmax=1)

if check_shear_bad_ps_kmap:
	sigmaG = 0.5
	R = 553#999
	y, x, e1, e2, w, m = WLanalysis.readFits('/Users/jia/CFHTLenS/catalogue/yxewm_subfield5_zcut0213.fit').T
	
	k, s1, s2 = (WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/debug_ps/emulator_subfield5_WL-only_emu1-512b240_Om0.755_Ol0.245_w-0.456_ns0.960_si1.359_4096xy_%04dr.fit'%(R)).T)[[0,1,2]]
	
	s1m = (1+m)*s1
	s2m = (1+m)*s2
	eint1, eint2 = WLanalysis.rndrot(e1, e2, iseed=553)#random rotation	
	## get reduced shear
	e1red, e2red = WLanalysis.eobs_fun(s1m, s2m, k, eint1, eint2)
	
	print 'coords2grid'
	A, galn = WLanalysis.coords2grid(x, y, array([k, e1red*w, e2red*w, w]))
	Mk, Ms1, Ms2, Mw = A
	
	Me1_smooth = WLanalysis.weighted_smooth(Ms1, Mw, sigmaG=sigmaG)
	Me2_smooth = WLanalysis.weighted_smooth(Ms2, Mw, sigmaG=sigmaG)
	Mk_smooth = WLanalysis.weighted_smooth(Mk, Mw, sigmaG=sigmaG )
	Mk_vw = WLanalysis.KSvw(Me1_smooth, Me2_smooth)
	
	#plotimshow(Mk_vw,'test_bad_kmap_Mkvw_%04dr'%(R), vmin = -0.25, vmax = 0.25)
	#plotimshow(Mk_smooth, 'test_bad_kmap_Mkconv_%04dr'%(R))#, vmin = -0.04, vmax = 0.08)
	
	#plotimshow(Me1_smooth,'test_bad_kmap_Me1smooth_%04dr'%(R))
	#plotimshow(Me2_smooth,'test_bad_kmap_Me2smooth_%04dr'%(R))
	
	PPA512=2.4633625
	sigma = sigmaG * PPA512
	#smooth_w = snd.filters.gaussian_filter(Mw.astype(float),sigma,mode='constant')
	#plotimshow(smooth_w, 'test_bad_kmap_smoothw_%04dr_sigmaG%i'%(R,sigmaG*10))
	
	#Me1smooth_noweight = snd.filters.gaussian_filter(Ms1.astype(float),sigma,mode='constant')
	#plotimshow(Me1smooth_noweight, 'test_bad_kmap_Me1smooth_noweight_%04dr_sigmaG%i'%(R,sigmaG*10))
	
	B, galn = WLanalysis.coords2grid(x, y, array([eint1, s1m, s1, e1red]))
	Meint1, Ms1m, Ms1b, Me1red = B
	Me1red_smoothed=WLanalysis.smooth(Me1red,sigma)
	
	#plotimshow(Meint1, 'bad_kmap_%04dr_sigmaG%i_Meint1'%(R,sigmaG*10))
	#plotimshow(Ms1m, 'bad_kmap_%04dr_sigmaG%i_Ms1m'%(R,sigmaG*10))
	#plotimshow(Ms1b, 'bad_kmap_%04dr_sigmaG%i_Ms1b'%(R,sigmaG*10))
	#plotimshow(Me1red, 'bad_kmap_%04dr_sigmaG%i_Me1red'%(R,sigmaG*10))
	#plotimshow(Me1red_smoothed,'bad_kmap_%04dr_sigmaG%i_Me1red_smoothed'%(R,sigmaG*10))
	
	Mg1 = Ms1/(1-Mk)##g1=s1m
	Mg1_smoothed = WLanalysis.smooth(Mg1, sigma)
	plotimshow(Mg1, 'bad_kmap_%04dr_sigmaG%i_Mg1'%(R,sigmaG*10))
	plotimshow(Mg1_smoothed, 'bad_kmap_%04dr_sigmaG%i_Mg1_smoothed'%(R,sigmaG*10))
	Mksmoothed = WLanalysis.smooth(1-Mk, sigma)
	plotimshow(Mksmoothed, 'bad_kmap_%04dr_sigmaG%i_1-Mk_smoothed'%(R,sigmaG*10),vmin=-0.1,vmax=0.1)
		#eobs = (g+eint)/(1-g*eint)

	bad_location = where(Me1red_smoothed>10)
	Mk_bad_location = Mksmoothed[bad_location]

plot_Mw = 0
if plot_Mw:
	sigmaG=1.0
	PPA512=2.4633625
	sigma = sigmaG * PPA512
	for i in range(1,14):
		Mw = WLanalysis.readFits(emu_dir+'debug_ps/Mw/SIM_Mw_subfield%i.fit'%(i))
		Mw_smoothed = WLanalysis.smooth(Mw, sigma)
		plotimshow(Mw,'Mw_smoothed_sigmaG%02d_sf%02d'%(sigmaG*10,i))
		
recreate_mask = 0
if recreate_mask:
	PPA512=2.4633625
	sigmaG_arr = (0.5, 1, 1.8, 3.5, 5.3, 8.9)
	for i in range(1,14):
		y, x, e1, e2, w, m = WLanalysis.readFits('/Users/jia/CFHTLenS/catalogue/yxewm_subfield%i_zcut0213.fit'%(i)).T
		A, galn = WLanalysis.coords2grid(x, y, array([w,]))
		for sigmaG in sigmaG_arr:
			fn = '/Users/jia/CFHTLenS/KSsim/mask/CFHT_mask_ngal5_sigma%02d_subfield%02d.fits'%(sigmaG*10, i)
			#galn_smooth = WLanalysis.smooth(galn, PPA512*sigmaG)
			#mask=(galn_smooth>5/PPA512**2).astype(int)
			#WLanalysis.writeFits(mask,fn)
			mask = WLanalysis.readFits(fn)
			plotimshow(mask, 'CFHT_mask_ngal5_sigma%02d_subfield%02d'%(sigmaG*10, i), vmin=0, vmax=1)
			
		