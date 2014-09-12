# Jia Liu 06/05/2014
# This code uses spline interpolation for CFHT emulator
# currently only work with Jia's laptop

import numpy as np
import triangle
from scipy import *
#import scipy.optimize as op
import emcee
from scipy import interpolate#,stats
import os
import WLanalysis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import matplotlib.gridspec as gridspec 
from sklearn.gaussian_process import GaussianProcess
from scipy.spatial import cKDTree
import scipy.ndimage as snd

######## knobs ##############

######## official plots #####
CFHT_fields_and_masks = 0#plot out the whole 4 fields
contour_peaks_smoothing = 0
contour_peaks_fieldselect = 0
contour_peaks_powspec = 0
include_w = 0
contour_including_w = 0
sample_interpolation = 0
interp_2D_plane = 0
good_bad_powspec = 0
sample_points = 0#for final fits wiht 3 random points
good_bad_peaks = 0
contour_ps_fieldselect = 0
m_correction = 0
######## tests ##############
compare_pk_contour_andrea = 0
bad_pointings = 1
chisq_heat_map = 0
ps_remove_4bins = 0

ps_replaced_by_good = 0
ps_replaced_by_nicaea = 0
ps_replaced_with_pk = 0# use the same plotting routine wrote for powspec to to peaks, simply make ps_mat = pk_mat
combined_ps_pk = 0
test_interp_method = 0#this is using spline
draw2Dplane = 0
test_gp = 0
test_pca_ncomp = 0
project_sims_3D = 0
check_bad_ps = 0#plot out all 1000 ps for each realization, see if there's outliers
bad_KSmap = 0
check_ps_sum = 0
test_MCMC = 0
peaks_13subfield_sum = 0
try_mask_powspec = 0
check_shear_bad_ps_kmap = 0
build_CFHT_KS_PS_PK = 0
cosmo_params_2D = 0
single_interpolation_fidu99 = 0

draw_contour_chisq_map = 0 #contour using chisq
draw_contour_smoothed_MCMC_map = 0
quick_test_ps_pk_plot = 0
dC_dp = 0 #covariance matrix inverse dependence on parameter
CFHT_ps_5bins = 0 # manually change the 5 outliers in CFHT ps, and do chisq
pk_last_2bins = 0
ps_only_2bins = 0

varying_C = 0
CFHT2pcf = 0
combined_smoothing_scale = 0
CFHT_ps_full_vs_good_sky = 0
correlation_matrix = 0
ps_from_2pcf = 0
std_converge = 0
theory_powspec_err = 1

cosmo_labels = [r'${\rm\Omega_m}$',r'$\rm{w}$',r'${\rm\sigma_8}$']

kmin = -0.04 # lower bound of kappa bin = -2 SNR
kmax = 0.12 # higher bound of kappa bin = 6 SNR
if bad_pointings:
	emu_dir = '/Users/jia/CFHTLenS/emulator/goodonly/'
	plot_dir = '/Users/jia/weaklensing/CFHTLenS/plot/goodonly/'
else:
	emu_dir = '/Users/jia/CFHTLenS/emulator/'
	plot_dir = '/Users/jia/weaklensing/CFHTLenS/plot/'

sigmaG=1.0#1.0#1.0
sigmaG_arr = (0.5, 1, 1.8, 3.5, 5.3, 8.9)
bins=25#50


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

ps_mat_fn = emu_dir+'powspec_sum/ps_mat_sigma05.fit'
if os.path.isfile(ps_mat_fn):
	ps_mat = WLanalysis.readFits(ps_mat_fn).reshape(91,1000,-1)
	#ps_mat = ps_mat[:,:,:-15]###cut ell
else:	
	ps_mat = array(map(getps, cosmo_params))[:,:,11:]
	if bad_pointings:
		ps_mat/=7.6645622253410002#sum(fsky)
	WLanalysis.writeFits(ps_mat.reshape(91,-1), ps_mat_fn)



######### fiducial cosmo, use #46 from ps_mat instead, because 
######### actual difu has no mask ######
#ps_fidu = WLanalysis.readFits('/Users/jia/CFHTLenS/KSsim/powspec_sum13fields/SIM_powspec_sigma05_rz1_mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800_1000R.fit')[:,11:]


####### try interpolate in log space ######
#ps_mat = log10(ps_mat)
###########################################


fsky = array([0.800968170166,0.639133453369,0.686164855957,0.553855895996,
		0.600227355957,0.527587890625,0.671237945557,0.494361877441,
		0.565235137939,0.592998504639,0.584747314453,0.530345916748,
		0.417697906494])
fsky_all = array([0.839298248291,0.865875244141,0.809467315674,
		  0.864688873291,0.679264068604,0.756385803223,
		  0.765892028809,0.747268676758,0.77250289917,
		  0.761451721191,0.691867828369,0.711254119873,
		  0.745429992676])

def genCFHTps(i, both=True):
	kmap = WLanalysis.readFits('/Users/jia/CFHTLenS/CFHTKS/CFHT_KS_sigma05_subfield%02d.fits'%(i))
	mask = WLanalysis.readFits('/Users/jia/CFHTLenS/catalogue/mask/BAD_CFHT_mask_ngal5_sigma05_subfield%02d.fits'%(i))
	mask_all = WLanalysis.readFits('/Users/jia/CFHTLenS/catalogue/mask/CFHT_mask_ngal5_sigma05_subfield%02d.fits'%(i))
	
	ps = 1/fsky[i-1]*WLanalysis.PowerSpectrum(kmap*mask, sizedeg=12.0)[-1]
	ps_all = 1/fsky_all[i-1]*WLanalysis.PowerSpectrum(kmap*mask_all, sizedeg=12.0)[-1]
	if both:
		return ps, ps_all
	else:
		return ps
if bad_pointings:

	galcount = fsky/sum(fsky)
	ps_CFHT_mat_fn = emu_dir+'CFHT_powspec_sigma05.fit'
	
	## ps_CFHT 
	#ps_arr = array(map(genCFHTps,range(1,14)))
	#ps_CFHT_mat = ps_arr[:,0,11:]
	#ps_CFHT_all_mat = ps_arr[:,1,11:]
	
	#### pure mean
	#ps_CFHT = mean(ps_CFHT_mat,axis=0)
	#ps_CFHT_all = mean(ps_CFHT_all_mat,axis=0)
	
	#### weighted by area covered
	#ps_CFHT = (sum(sqrt(fsky).reshape(-1,1)*ps_CFHT_mat, axis=0))/sum(sqrt(fsky))
	#ps_CFHT_all = (sum(sqrt(fsky_all).reshape(-1,1)*ps_CFHT_all_mat, axis=0))/sum(sqrt(fsky_all))
	
	ell_arr = logspace(log10(110.01746692),log10(25207.90813028),50)[11:]
	
	#show()
	
	if os.path.isfile(ps_CFHT_mat_fn):
		ps_CFHT_mat = WLanalysis.readFits(ps_CFHT_mat_fn)
	else:
		ps_CFHT_mat = array(map(genCFHTps,range(1,14))).squeeze()
		WLanalysis.writeFits(ps_CFHT_mat,ps_CFHT_mat_fn)
	ps_CFHT = (sum(galcount.reshape(-1,1)*ps_CFHT_mat, axis=0))[11:]
	
	## plot plot out one by one field
	
	#print 'ps_CFHT',ps_CFHT
	
	### all sky ps
	#maskAllGen = lambda i: WLanalysis.readFits('/Users/jia/CFHTLenS/catalogue/mask/CFHT_mask_ngal5_sigma05_subfield%02d.fits'%(i))	
	#mask_all_arr = array(map(maskAllGen, range(1,14)))
	#for i in range(13):
		#print sum(mask_all_arr[i]).astype(float)/512**2
	#ps_CFHT_mat_all = WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/CFHT_powspec_sigma05.fit')
	#ps_CFHT_all = (sum(ps_CFHT_mat_all/fsky_all.reshape(-1,1), axis=0))[11:]
	
else:
	galcount = array([342966,365597,322606,380838,
		263748,317088,344887,309647,
		333731,310101,273951,291234,
		308864]).astype(float)
	galcount /= sum(galcount)
	ps_CFHT_mat = WLanalysis.readFits('/Users/jia/CFHTLenS/CFHTKS/CFHT_powspec_sigma05.fit')
	ps_CFHT = (sum(galcount.reshape(-1,1)*ps_CFHT_mat, axis=0))[11:]
	#ps_CFHT = (galcount.reshape(-1,1))*ps_CFHT_mat


gs = gridspec.GridSpec(2,1,height_ratios=[3,1]) 
ell_arr = logspace(log10(110.01746692),log10(25207.90813028),50)[11:]
ell_arr0 = ell_arr.copy()
############### peaks ###################
x = linspace(-0.04, 0.12, bins+1)#26)
x = x[:-1]+0.5*(x[1]-x[0])

def getpk (cosmo_param, sigmaG=sigmaG, bins = bins):
	print cosmo_param
	om, w, si8 = cosmo_param
	pk_fn = 'SIM_peaks_sigma%02d_emu1-512b240_Om%.3f_Ol%.3f_w%.3f_ns0.960_si%.3f_600bins.fit'%(sigmaG*10, om,1-om,w,si8)
	pk600bins = WLanalysis.readFits(emu_dir+'peaks_sum/sigma%02d/'%(sigmaG*10)+pk_fn)
	pk = pk600bins.reshape(1000, -1, 600/bins)
	pk = sum(pk, axis = -1)
	return pk


PPA512=2.4633625

		
	
def create_dir_if_nonexist(dirname):
	try:
		os.mkdir(dirname)
	except Exception:
		print 'error'
		pass
	
if ps_replaced_by_good:
	bad_arr = array([6,14,24,27,31,32,33,38,42,43,44,45,53,54,55,61,63,64,65,66,67,72,74,75, 76,81,82,83,84,88,89])
	good_arr = delete(arange(91), bad_arr)
	#ps_avg = ps_avg[good_arr]

if ps_replaced_with_pk:
	print 'ps_replaced_with_pk'
	pk_mat_fcn = lambda sigmaG: emu_dir+'peaks_sum/pk_mat_sigma%02d_%02dbins.fit'%(sigmaG*10,bins)
	
	if combined_smoothing_scale:
		pk_mat_gen = lambda sigmaG:WLanalysis.readFits(pk_mat_fcn(sigmaG)).reshape(91,1000,-1)
		pk_mat = concatenate(map(pk_mat_gen, [1.0,1.8]),axis=-1)
		print 'pk_mat.shape',pk_mat.shape
	
	else:
		pk_mat_fn = pk_mat_fcn(sigmaG)
		if os.path.isfile(pk_mat_fn):
			pk_mat = WLanalysis.readFits(pk_mat_fn).reshape(91,1000,-1)
		else:
			pk_mat = array(map(getpk, cosmo_params))
			WLanalysis.writeFits(pk_mat.reshape(91,-1), pk_mat_fn)
	
	
	#pk_mat /=13.0#andrea test
	pk_avg = mean(pk_mat,axis=1)
	pk_std = std(pk_mat, axis=1)
	ps_mat = pk_mat
	ps_avg = pk_avg
	ps_std = pk_std
	ell_arr = x
	#ps_fidu = WLanalysis.readFits('/Users/jia/CFHTLenS/KSsim/peaks_sum13fields/SIM_peaks_sigma10_rz1_mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800_1000R_025bins.fit')
	#fidu_avg = mean(ps_fidu,axis=0)
	#fidu_std = std(ps_fidu,axis=0)
	#cov_mat = mat(cov(ps_fidu,rowvar=0))
	#cov_inv = cov_mat.I
	if bad_pointings:
		ps_CFHT_mat_fn = emu_dir+'CFHT_peaks_sigma%02d_bins%i.fit'%(sigmaG*10,bins)
	else:
		ps_CFHT_mat_fn = '/Users/jia/CFHTLenS/CFHTKS/CFHT_peaks_sigma%02d_%03dbins.fit'%(sigmaG*10,bins)
	if os.path.isfile(ps_CFHT_mat_fn):
		ps_CFHT_mat = WLanalysis.readFits(ps_CFHT_mat_fn)
	else:
		def genCFHTpk (i):
			kmap = WLanalysis.readFits('/Users/jia/CFHTLenS/CFHTKS/CFHT_KS_sigma%02d_subfield%02d.fits'%(sigmaG*10,i))
			mask = WLanalysis.readFits('/Users/jia/CFHTLenS/catalogue/mask/BAD_CFHT_mask_ngal5_sigma%02d_subfield%02d.fits'%(sigmaG*10,i))
			pk = WLanalysis.peaks_mask_hist(kmap*mask, mask, bins, kmin = kmin, kmax = kmax)
			return pk
		ps_CFHT_mat = array(map(genCFHTpk,range(1,14))).squeeze()
		WLanalysis.writeFits(ps_CFHT_mat,ps_CFHT_mat_fn)
	ps_CFHT = sum(ps_CFHT_mat, axis=0)
	if combined_smoothing_scale:
		ps_CFHT1 =WLanalysis.readFits( '/Users/jia/CFHTLenS/CFHTKS/CFHT_peaks_sigma10_025bins.fit')
		ps_CFHT2 =WLanalysis.readFits( '/Users/jia/CFHTLenS/CFHTKS/CFHT_peaks_sigma18_025bins.fit')
		ps_CFHT = sum(concatenate((ps_CFHT1, ps_CFHT2),axis=-1),axis=0)
	#ps_CFHT /=13.0 #andrea test	

if pk_last_2bins:
	ps_CFHT = ps_CFHT[:-2]
	ell_arr = ell_arr[:-2]
	ps_mat = ps_mat[:,:,:-2]
	ps_avg = std(ps_mat, axis=1)
	ps_avg = mean(ps_mat,axis=1)
	
if ps_only_2bins:
	#begin = 16
	begin = int(sys.argv[1])
	ps_CFHT = ps_CFHT[begin:begin+2]
	#ell_arr = ell_arr[begin:begin+2]
	ps_mat = ps_mat[:,:,begin:begin+2]
	ps_avg = std(ps_mat, axis=1)
	ps_avg = mean(ps_mat,axis=1)

if ps_remove_4bins:
	#ellcut = int(sys.argv[1])
	
	#idx = delete(idx,[0,1,6,7,24,25])
	#idx = delete(idx,range(22))
	#idx = delete(idx,range(ellcut))
	#i0,i1=int(sys.argv[1]),int(sys.argv[2])
	#idx = range(i0,i1) 
	if ps_replaced_with_pk:
		idx = where(ps_CFHT > 20)[0]
		print ps_CFHT, idx
		idx999 = idx.copy()
		print 'len(idx)',len(idx)
	ps_CFHT = ps_CFHT[idx]
	#ell_arr = ell_arr[idx]
	ps_mat = ps_mat[:,:,idx]
	ps_avg = std(ps_mat, axis=1)
	ps_avg = mean(ps_mat,axis=1)
	
if combined_ps_pk:
	print 'combined_ps_pk'
	
	pk_mat_fn = emu_dir+'peaks_sum/pk_mat_sigma%02d_25bins.fit'%(sigmaG*10)
	if os.path.isfile(pk_mat_fn):
		pk_mat = WLanalysis.readFits(pk_mat_fn).reshape(91,1000,-1)
	else:
		pk_mat = array(map(getpk, cosmo_params))
		WLanalysis.writeFits(pk_mat.reshape(91,-1), pk_mat_fn)
	
	#pk_avg = mean(pk_mat,axis=1)
	#pk_std = std(pk_mat, axis=1)	
	#pk_CFHT_mat = WLanalysis.readFits('/Users/jia/CFHTLenS/CFHTKS/CFHT_peaks_sigma%02d_025bins.fit'%(sigmaG*10))
	
	if bad_pointings:
		pk_CFHT_mat_fn = emu_dir+'CFHT_peaks_sigma%02d.fit'%(sigmaG*10)
	else:
		pk_CFHT_mat_fn = '/Users/jia/CFHTLenS/CFHTKS/CFHT_peaks_sigma%02d_025bins.fit'%(sigmaG*10)
	#pk_CFHT_mat = WLanalysis.readFits(pk_CFHT_mat_fn)
	#pk_CFHT = sum(pk_CFHT_mat, axis=0)
	
	if combined_smoothing_scale:
		pk_mat_fcn = lambda sigmaG: emu_dir+'peaks_sum/pk_mat_sigma%02d_%02dbins.fit'%(sigmaG*10,bins)
		pk_mat_gen = lambda sigmaG:WLanalysis.readFits(pk_mat_fcn(sigmaG)).reshape(91,1000,-1)
		pk_mat = concatenate(map(pk_mat_gen, [1.0,1.8]),axis=-1)
		print 'pk_mat.shape',pk_mat.shape
		
	pk_CFHT1 =WLanalysis.readFits( '/Users/jia/CFHTLenS/CFHTKS/CFHT_peaks_sigma10_025bins.fit')
	pk_CFHT2 =WLanalysis.readFits( '/Users/jia/CFHTLenS/CFHTKS/CFHT_peaks_sigma18_025bins.fit')
	pk_CFHT = sum(concatenate((pk_CFHT1, pk_CFHT2),axis=-1),axis=0)
	
	#ps_CFHT_mat = concatenate([ps_CFHT_mat,pk_CFHT_mat],axis=-1)
	ps_CFHT = concatenate([ps_CFHT, pk_CFHT])
	
	ps_mat = concatenate([ps_mat,pk_mat],axis=-1)
	ps_avg = mean(ps_mat,axis=1)
	ps_std = std(ps_mat, axis=1)
	ell_arr = concatenate([ell_arr,x])	
#######################################################
#### this is the fiducial cosmology for testing purpose
#### previously used the fiducial from the 4 runs, but 
#### after I changed to WL approximation in 7/1/2014, use
#### cosmo #48 as fiducial cosmo, and delete it from the 91
#### simulations when building interpolator for MCMC
#### uncomment the next 6 lines to get fiducial model
#######################################################

rmidx=where(amin(average(ps_mat,axis=1),axis=0)==0)
if len(rmidx[0])>0:
	print 'rmidx',rmidx
	ps_mat = delete(ps_mat,rmidx,axis=-1)
	ps_CFHT = delete(ps_CFHT,rmidx[0])

ps_fidu = ps_mat[48]
fidu_params = cosmo_params[48]

#ps_mat = delete(ps_mat,48,axis=0)
#cosmo_params = delete(cosmo_params,48,axis=0)

ps_avg = mean(ps_mat,axis=1) # array [91, 50]
ps_std = std(ps_mat, axis=1)# array [91, 50]
#ps_stdlog = log10(ps_std)
#try:
	#cov_inv = mat(cov_mat).I
#except Exception:
	#print 'errors'
	#rmidx = where(sum(cov_mat,axis=1)==0)
	#ps_mat = delete(ps_mat,rmidx,axis=-1)
	#ps_fidu = ps_mat[48]
	#cov_mat = cov(ps_fidu,rowvar=0)
	#cov_inv = mat(cov_mat).I
	#ell_arr = delete(ell_arr,rmidx)

if ps_replaced_by_nicaea:
	P_ell_noise_arr = WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/nicaea_params/P_kappa_noise_arr91.fit')
	ps_avg = P_ell_noise_arr[:,11:]
	
cov_mat = cov(ps_fidu,rowvar=0)
cov_inv = mat(cov_mat).I	
fidu_avg = mean(ps_fidu,axis=0)
fidu_std = std(ps_fidu,axis=0)

if CFHT_ps_full_vs_good_sky:
	for i in range(1,14):
		mask = WLanalysis.readFits('/Users/jia/CFHTLenS/catalogue/mask/BAD_CFHT_mask_ngal5_sigma05_subfield%02d.fits'%(i))
		f=figure(figsize=(10,6))
		ax = f.add_subplot(221)
		ax2 = f.add_subplot(223)
		ax3 = f.add_subplot(122)
		ps, ps_all = genCFHTps(i, both=1)
		ax.plot(ell_arr, ps[11:], '-',color='k',label='pass fields')
		ax.plot(ell_arr, ps_all[11:], 'r--', label='all fields')
		ax2.plot(ell_arr, ps_all[11:]/ps[11:]-1,'r--')
		ax2.plot(ell_arr, zeros(len(ell_arr)),'k')
		
		ax3.imshow(mask,origin='lower')
		ax.set_xscale('log')
		ax2.set_xscale('log')
		ax.set_yscale('log')
		ax2.set_xlabel(r'$\ell$')
		ax2.set_ylabel('frac diff')
		ax.set_xlim(ell_arr[0],ell_arr[-1])
		ax2.set_xlim(ell_arr[0],ell_arr[-1])
		ax.set_title('subfield%i'%(i))
		leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
		leg.get_frame().set_visible(False)
		plt.subplots_adjust(hspace=0.0)
		savefig(plot_dir+'subfield%i_ps_all_good.jpg'%(i))
		close()
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
	im=imshow(img,interpolation='nearest',origin='lower',vmin=vmin,aspect=1,vmax=vmax)
	colorbar()
	title(ititle,fontsize=16)
	savefig(plot_dir+'%s.jpg'%(ititle))
	close()	
	
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
		
		ax2.set_ylim(-0.1,0.1)
		leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
		leg.get_frame().set_visible(False)
		plt.setp(ax.get_xticklabels(), visible=False) 
		plt.subplots_adjust(hspace=0.0)
		
		
		if ps_replaced_with_pk:
			try:
				os.mkdir(plot_dir+'Rbs_peaks')
			except Exception:
				pass
			savefig(plot_dir+'Rbs_peaks/Rbs_smooth%i_emu_peaks_cosmo%02d.jpg'%(ismooth,i))
		elif ps_replaced_by_nicaea:
			savefig(plot_dir+'Rbs_nicaea/Rbs_smooth%i_emu_nicaea_cosmo%02d.jpg'%(ismooth,i))
		elif ps_replaced_by_good:
			savefig(plot_dir+'Rbs_60good/Rbs_smooth%i_emu_good_cosmo%02d.jpg'%(ismooth,i))
		else:
			create_dir_if_nonexist(plot_dir+'RBs')
			savefig(plot_dir+'RBs/Rbs_smooth0_emu_powspec_cosmo%02d.jpg'%(i))
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
labels=('$\Omega_m$',r'${\rm w}$','$\sigma_8$')
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
			create_dir_if_nonexist(plot_dir+'GP_test_powspec')
			savefig(plot_dir+'GP_test_powspec/GP_test_powspec_cosmo%02d.jpg'%(i))
		else:
			create_dir_if_nonexist(plot_dir+'GP_test_peaks')
			savefig(plot_dir+'GP_test_peaks/GP_test_cosmo%02d.jpg'%(i))
		
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
	#if ps_replaced_with_pk:
		#print 'ps_replaced_with_pk'
		#ps_mat = pk_mat
		#ps_avg = pk_avg
		#ps_std = pk_std
		#ell_arr = x
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
				create_dir_if_nonexist(plot_dir+'bad_powspec')
				savefig(plot_dir+'bad_powspec/sigmaG%02d_ps_mat_%i.jpg'%(sigmaG*10,i))
			else:
				create_dir_if_nonexist(plot_dir+'good_powspec')
				savefig(plot_dir+'good_powspec/sigmaG%02d_ps_mat_%i.jpg'%(sigmaG*10,i))
		else:
			savefig(plot_dir+'pk_mat_%i.jpg'%(i))
		close()



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
######################################################################
### build interpolator, a serious one (6/19/2014)
### use only 90 cosmo to build model, cosmo #48 is used as fiducial
### uncomment next 3 lines 
######################################################################
#icosmo = 48
icosmo = int(sys.argv[1])
ps_fidu = ps_mat.copy()[icosmo]
fidu_params = cosmo_params.copy()[icosmo]
fidu_avg = ps_avg.copy()[icosmo]
cosmo_params = delete(cosmo_params, icosmo, 0)
ps_avg = delete(ps_avg, icosmo, 0)
ps_std = delete(ps_std, icosmo, 0)

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
	#fidu_params = (0.26, -1, 0.8)
	
	
	steps = 2000
	burn = 100 # make sure burn < steps
	
	#obs = fidu_avg#ps_CFHT#
	#method = 'multiquadric'#'GP'#
	for obs in (fidu_avg,):#(ps_CFHT,):#(fidu_avg, ps_CFHT):#
	#for k in (10,20,45, 99, 260):#range(500,510):#
		#obs = ps_mat[48,k]
		for method in ('multiquadric',):#('multiquadric','GP'):# ('GP',):#
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
				#Ln = -chisq/2.0/39.0
				return float(Ln)
			
			def lnprob(params, obs):
				lp = lnprior(params)
				if not np.isfinite(lp):
					return -np.inf
				else:
					return lp + lnlike(params, obs)
			
			nll = lambda *args: -lnlike(*args)#minimize chisq
			result = op.minimize(nll, fidu_params, args=(obs,))
			print result
			
			ndim, nwalkers = 3, 100
			#pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
			pos = [array([0.26, -1, 0.798]) + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]
			
			print 'run sampler'
			sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(obs,))

			print 'run mcmc'
			sampler.run_mcmc(pos, steps)
			samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
			
			####################################
			WLanalysis.writeFits(samples,emu_dir+'MCMC_samples_fidu_%isteps_lnchisq.fit'%(steps))
			####################################
			
			errors = array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84],axis=0))))
			print 'Result\t[best fits, lower error, upper error]'
			print 'Omega_m:\t', errors[0]
			print 'w:\t\t', errors[1]
			print 'sigma_8:\t', errors[2]
				
			fig = triangle.corner(samples, labels=["$\Omega_m$", "$w$", "$\sigma_8$"],  plot_datapoints=False)#,truths=fidu_params)
			#fig.savefig(plot_dir+"emu_triangle_MCMC_fidu_%s.jpg"%(method))
			if ps_replaced_with_pk:
				print 'ps_replaced_with_pk'
				if obs[3] == fidu_avg[3]:
					fn = "emu_fidu_%s_peaks"%(method)
				elif obs[3] == ps_CFHT[3]:
					fn = "emu_CFHT_%s_peaks"%(method)
				else:
					fn = "emu_fidu%i_%s_peaks"%(k, method)
			else:
				if obs[3] == fidu_avg[3]:
					fn ='emu_fidu_%s'%(method)
				elif obs[3] == ps_CFHT[3]:
					fn ='emu_CFHT_%s'%(method)
				else:
					fn = "emu_fidu%i_%s"%(k, method)
			title('[%.3f, %.3f, %.3f]'%(errors[0,0], errors[1,0], errors[2,0]))
			fig.savefig(plot_dir+'triangle_'+fn+'_20140804_lnchisq.jpg')
			close()
	

			#best_fit = errors[:,0]
			#ps_interp=interp_cosmo(best_fit,method=method)
			
			#f=figure(figsize=(8,8))
			#ax=f.add_subplot(gs[0])
			#ax2=f.add_subplot(gs[1],sharex=ax)
			#ax.errorbar(ell_arr,obs,fidu_std,color='r',label=fn)
			#ax.plot(ell_arr,ps_interp,'b-',label=method)
			#ax2.plot(ell_arr,zeros(len(ell_arr)),'r-')
			#ax2.plot(ell_arr,ps_interp/obs-1,'b-')

			#if not ps_replaced_with_pk:
				#ax.set_xscale('log')
				#ax2.set_xscale('log')
				#ax.set_yscale('log')
				#ax.set_xlabel('ell')
				#ax.set_ylim(5e-5, 1e-2)
				#ax.set_xlim(ell_arr[0],ell_arr[-1])

			#ax2.set_ylim(-0.1, 0.1)
			#leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
			#leg.get_frame().set_visible(False)
			#plt.setp(ax.get_xticklabels(), visible=False) 
			#plt.subplots_adjust(hspace=0.0)
			#savefig(plot_dir+'frac_diff'+fn+'.jpg')
			#close()


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
			

if build_CFHT_KS_PS_PK:
	ngal_arcmin = 5
	bins = 25
	for sigmaG in sigmaG_arr:
		ps_mat_fn = '/Users/jia/CFHTLenS/CFHTKS/CFHT_powspec_sigma%02d.fit'%(sigmaG*10)
		pk_mat_fn = '/Users/jia/CFHTLenS/CFHTKS/CFHT_peaks_sigma%02d_%03dbins.fit'%(sigmaG*10,bins)
		ps_mat = zeros(shape=(13, 50))
		pk_mat = zeros(shape=(13, bins))
		for i in range(1,14):
			print 'sigmaG, i', sigmaG, i
			KS_fn = '/Users/jia/CFHTLenS/CFHTKS/CFHT_KS_sigma%02d_subfield%02d.fits'%(sigmaG*10,i)
			
			mask_fn = '/Users/jia/CFHTLenS/catalogue/mask/CFHT_mask_ngal%i_sigma%02d_subfield%02d.fits'%(ngal_arcmin,sigmaG*10,i)
			mask = WLanalysis.readFits(mask_fn)
			
			y, x, e1, e2, w, m = WLanalysis.readFits('/Users/jia/CFHTLenS/catalogue/yxewm_subfield%i_zcut0213.fit'%(i)).T
			
			k = array([e1*w, e2*w, (1+m)*w])
			A, galn = WLanalysis.coords2grid(x, y, k)
			Me1, Me2, Mw = A
			
			Me1_smooth = WLanalysis.weighted_smooth(Me1, Mw, PPA=PPA512, sigmaG=sigmaG)
			Me2_smooth = WLanalysis.weighted_smooth(Me2, Mw, PPA=PPA512, sigmaG=sigmaG)
			
			kmap = WLanalysis.KSvw(Me1_smooth, Me2_smooth)
			try:
				plotimshow(kmap, 'CFHT_KS_sigma%02d_sf%02d'%(sigmaG*10,i))
				plotimshow(kmap*mask, 'CFHT_KS_mask_sigma%02d_sf%02d'%(sigmaG*10,i))
				WLanalysis.writeFits(kmap, KS_fn)
				
			except Exception:
				print 'kmap error'
				pass
			#kmap *= mask
			ps = WLanalysis.PowerSpectrum(kmap*mask)[1]
			pk = WLanalysis.peaks_mask_hist(kmap, mask, bins, kmin=kmin, kmax=kmax)
			ps_mat[i-1] = ps
			pk_mat[i-1] = pk
		try:
			WLanalysis.writeFits(pk_mat, pk_mat_fn)
			WLanalysis.writeFits(ps_mat, ps_mat_fn)
		except Exception:
			print 'error'
			pass


if cosmo_params_2D:
	om_arr = linspace(0,1,5)
	w_arr = linspace(-3,0,5)
	si8_arr = linspace(0.1,1.5,5)
	label_tix_arr = [om_arr, w_arr, si8_arr]
	cosmo_params =  genfromtxt('/Users/jia/CFHTLenS/emulator/cosmo_params.txt')
	f=figure(figsize=(9,8))
	k=0
	k_arr = [1,3,4]
	for ij in [[0,1],[0,2], [1,2]]:
		i,j=ij
		ax = f.add_subplot(2,2,k_arr[k])#, projection='3d')
		param1, param2 = cosmo_params.T[ij]
		ax.scatter(param1, param2, color='k', s=20)#, cmap='gray')
		if k_arr[k] in [3,4]:
			ax.set_xlabel(labels[i],fontsize=18)
			ax.set_xticks(label_tix_arr[i])
		if k_arr[k] in [1,3]:
			ax.set_ylabel(labels[j],fontsize=18)
			ax.set_yticks(label_tix_arr[j])
		if k_arr[k] == 1:
			plt.setp(ax.get_xticklabels(), visible=False)
		if k_arr[k] == 4:
			plt.setp(ax.get_yticklabels(), visible=False)
		k+=1
	ax4 = f.add_subplot(2,2,2, projection='3d')
	x, y, z = cosmo_params.T
	ax4.scatter(x, y, z, s=20, color='k')
	ax4.set_xlabel('\n'+labels[0],fontsize=18)#,labelpad=0)
	ax4.set_ylabel('\n$\hspace{1}$'+labels[1],fontsize=18)
	ax4.set_zlabel('\n'+labels[2],fontsize=18)
	#plt.xticks(rotation=50)
	#plt.yticks(rotation=30)
	ax4.set_xticks(label_tix_arr[0][::2])
	ax4.set_yticks(label_tix_arr[1][::2])
	ax4.set_zticks(label_tix_arr[2][::2])
	ax4.tick_params(direction='in', pad=0)
	#plt.subplots_adjust(wspace=0.25,hspace=0.25)
	plt.subplots_adjust(wspace=0.05,hspace=0.05,left=0.09, right=0.9)
	#show()
	
	savefig(plot_dir+'cosmo_params_2D.jpg')
	savefig(plot_dir+'cosmo_params_2D.pdf')
	close()
	
if single_interpolation_fidu99:	
	method = 'multiquadric'
	obs = ps_mat[48,99]
	best_fit = [0.406, -1.513, 0.522]#errors[:,0]
	lo_om = [0.25, -1.513, 0.522]
	hi_om = [0.55, -1.513, 0.522]
	hi_si8= [0.406, -1.513, 0.7]
	lo_si8= [0.406, -1.513, 0.35]
	lo_both = [0.25, -1.513, 0.35]
	variations = [best_fit, hi_om, lo_om, hi_si8, lo_si8, lo_both]
	variation_labels = ['best_fit', 'hi_om', 'lo_om', 'hi_si8', 'lo_si8', 'lo_both']
	seed(69)
	colors=rand(len(variations),3)
	
	f=figure(figsize=(8,8))
	ax=f.add_subplot(gs[0])
	ax2=f.add_subplot(gs[1],sharex=ax)
	ax.errorbar(ell_arr,obs,fidu_std,color='k',linewidth=2)
	ax.plot(ell_arr,obs,color='k',label='true',linewidth=2)
	for i in range(len(variations)):
		best_fit = variations[i]
		ps_interp=interp_cosmo(best_fit,method=method)
		
		del_N = np.mat(ps_interp - obs)
		chisq = float(del_N*cov_inv*del_N.T)/39.0
		ax.plot(ell_arr, ps_interp, color=colors[i], label='%s [%.3f, %.3f, %.3f] $\chi^2$=%.2f' % (variation_labels[i], best_fit[0],best_fit[1],best_fit[2], chisq),linewidth=2)
		ax2.plot(ell_arr,ps_interp/obs-1,color=colors[i],linewidth=2)
		
	ax2.plot(ell_arr,zeros(len(ell_arr)),'k-')
	if not ps_replaced_with_pk:
		ax.set_xscale('log')
		ax2.set_xscale('log')
		ax.set_yscale('log')
		ax.set_xlabel('ell')
		ax.set_ylim(2e-5, 1e-2)
		ax.set_xlim(ell_arr[0],ell_arr[-1])

	ax2.set_ylim(-0.3, 0.15)
	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
	leg.get_frame().set_visible(False)
	plt.setp(ax.get_xticklabels(), visible=False) 
	plt.subplots_adjust(hspace=0.0)
	savefig(plot_dir+'frac_diff_fidu99.jpg')
	close()

if chisq_heat_map:
	print 'sigmaG',sigmaG
	#from multiprocessing import Pool
	#p = Pool(101)
	obs = ps_CFHT
	l = 100
	m = 102
	om_arr = linspace(0,1.2,l)
	si8_arr = linspace(0,1.6,m)
	def plot_heat_map_w (w):
		print 'w=',w	
		#l = 10
		#obs = ps_mat[48,99]
		heatmap = zeros(shape=(l,m))
		for i in range(l):
			for j in range(m):
				best_fit = (om_arr[i], w, si8_arr[j])
				ps_interp = interp_cosmo(best_fit)	
				del_N = np.mat(ps_interp - obs)
				chisq = float(del_N*cov_inv*del_N.T)
				heatmap[i,j] = chisq
				
				#f=figure(figsize=(8,8))
				#ax=f.add_subplot(gs[0])
				#ax2=f.add_subplot(gs[1],sharex=ax)
				#ax.errorbar(ell_arr,obs,fidu_std,color='k',linewidth=2)
				#ax.plot(ell_arr,obs,color='k',label='true',linewidth=2)
				#ax.plot(ell_arr, ps_interp, color='r',linewidth=2)
				#ax.set_title('[%.3f, %.3f, %.3f] $\chi^2$=%.2f' % (best_fit[0],best_fit[1],best_fit[2], chisq))
				#ax2.plot(ell_arr,ps_interp/obs-1,'r',linewidth=2)
					
				#ax2.plot(ell_arr,zeros(len(ell_arr)),'k-')
				
				#ax.set_xscale('log')
				#ax2.set_xscale('log')
				#ax.set_yscale('log')
				#ax.set_xlabel('ell')
				#ax.set_ylim(2e-5, 1e-2)
				#ax.set_xlim(ell_arr[0],ell_arr[-1])

				#ax2.set_ylim(-0.5, 0.5)
				#plt.setp(ax.get_xticklabels(), visible=False) 
				#plt.subplots_adjust(hspace=0.0)
				#savefig(plot_dir+'lowerleftcorner/test_lower_left_corner_%.3f_%.3f_%.3f.jpg' % (best_fit[0],best_fit[1],best_fit[2]))
				#close()
				
				
		#figure(figsize=(6,8))
		#im=imshow(heatmap.T,interpolation='nearest',origin='lower',aspect=1,extent=[0,1.2,0,1.6],vmin=0,vmax=5*len(ps_CFHT))
		#xlabel('Omega_m')
		#ylabel('sigma8')
		#title('w='+str(w))
		#colorbar()
		#if ps_replaced_with_pk:
			#plotfn = plot_dir+'chisq_cube/CFHT_pk_chisq_heat_map_w%.2f_sigmaG%02d.jpg'%(w,sigmaG*10)
		#elif combined_ps_pk:
			#plotfn = plot_dir+'chisq_cube/CFHT_combined_chisq_heat_map_w%.2f_sigmaG%02d.jpg'%(w,sigmaG*10)
			
			#if combined_smoothing_scale:
				#plotfn = plot_dir+'chisq_cube/CFHT_combined_chisq_heat_map_w%.2f_sigmaG1018.jpg'%(w)
		#else:
			#plotfn = plot_dir+'chisq_cube/CFHT_ps_chisq_heat_map_w%.2f.jpg'%(w)
		#savefig(plotfn)
		#close()	
		
		#try:
			#drawContour2D(exp(-heatmap/2), 'cube_slices_%s_w%.2f'%(ellcut,ap,w), xvalues=om_arr, yvalues=si8_arr)
		#except Exception:
			#print 'fail'
			#pass
		
		return heatmap
	chisq_cube = array(map(plot_heat_map_w,linspace(0,-3,101)))#w, om, si8
	if ps_replaced_with_pk:
		cube_fn = emu_dir+'chisq_cube_CFHT_pk_sigmaG%02d.fit'%(sigmaG*10)
		if combined_smoothing_scale:
			cube_fn = emu_dir+'chisq_cube_CFHT_pk_sigmaG1018.fit'
	elif combined_ps_pk:
		#cube_fn = emu_dir+'chisq_cube_CFHT_combined_sigmaG%02d.fit'%(sigmaG*10)
		cube_fn = emu_dir+'chisq_cube_CFHT_combined_sigmaG1018.fit'
	else:
		cube_fn = emu_dir+'chisq_cube_CFHT_ps.fit'
	if ps_remove_4bins:
		cube_fn = cube_fn[:-4]+'ellcut%i_%i.fit'%(i0,i1)
	WLanalysis.writeFits(chisq_cube.reshape(-1), cube_fn)

def findmodes(image, logbins = True, bins = 50):
	"""
	Calculate the azimuthally averaged radial profile.
	Input:
	image = The 2D image
	center = The [x,y] pixel coordinates used as the center. The default is None, which then uses the center of the image (including fracitonal pixels).
	Output:
	ell_arr = the ell's, lower edge
	tbin = power spectrum
	"""
	# Calculate the indices from the image
	y, x = np.indices(image.shape)
	center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
	r = np.hypot(x - center[0], y - center[1])#distance to center pixel, for each pixel

	# Get sorted radii
	ind = np.argsort(r.flat)
	r_sorted = r.flat[ind] # the index to sort by r
	i_sorted = image.flat[ind] # the index of the images sorted by r

	# find index that's corresponding to the lower edge of each bin
	kmin=1.0
	kmax=image.shape[0]/2.0
	edges = logspace(log10(kmin),log10(kmax),bins+1)
	if edges[0] > 0:
		edges = append([0],edges)
		
	hist_ind = np.histogram(r_sorted,bins = edges)[0]
	return hist_ind[12:]

if quick_test_ps_pk_plot:
	## test Lam's theoretical noise
	ell_edges, psd1D = WLanalysis.azimuthalAverage(ones(shape=(512,512)))
	ell_edges *= 360./sqrt(12)
	del_ell = ell_edges[1:]-ell_edges[:-1]
	del_ell = del_ell[11:]
	L = radians(sqrt(13.0*12*sum(fsky)))
	N = (del_ell*L/2/pi)**2	
	theory_std = 1.0/sqrt(N)
	
	N_sim = array([   12,     8,    16,    20,    24,    28,    24,    44,    48,
          76,    76,   116,   124,   164,   200,   252,   296,   408,
         472,   608,   756,   936,  1184,  1460,  1840,  2284,  2860,
        3588,  4444,  5580,  6928,  8680, 10816, 13500, 16864, 21056,
       26276, 32812, 40980]).astype(float)
	
	
	## only 1 subfield, instead of 13 of them
	
	sf1_ps_mat = WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/GoodOnly/powspec_sum/SIM_powspec_sigma05_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_subfield01.fit')[:,11:]/7.6645622253410002
	sf1_avg = mean(sf1_ps_mat,axis=0)
	norm_sf1 = average(fidu_avg[10:25]/sf1_avg[10:25])
	
	sf1_ps_mat *= norm_sf1
	sf1_avg = sf1_avg*norm_sf1
	sf1_std = std(sf1_ps_mat,axis=0)
	
	#best_fit = [0.7, -1.0, 0.3]
	obs=ps_CFHT
	best_fit = [0.26, -1, 0.8]#[1.12, -1.5, 0.27]
	ps_interp = interp_cosmo(best_fit)	
	del_N = np.mat(ps_interp - obs)
	chisq=float(del_N*cov_inv*del_N.T)
	f=figure(figsize=(8,8))
	ax=f.add_subplot(gs[0])
	ax2=f.add_subplot(gs[1],sharex=ax)
	#ax.errorbar(ell_arr,obs,fidu_std,color='k',linewidth=1)
	#ax.errorbar(ell_arr,obs,fidu_std,color='k',label='CFHT',linewidth=1)
	#for r in linspace(0.91,0.98,8)[::2]:
		#seed(int(r*100))
		#obs0=ps_CFHT.copy()
		#obs0[6:11]=r*obs0[6:11]
		#ax.plot(ell_arr,obs0,label='r=%s'%(r),color=rand(3),linewidth=1)
		##ax2.plot(ell_arr,(obs0-obs)/fidu_std,linewidth=1)
		#ax2.plot(ell_arr,(obs0-obs)/obs,color=rand(3),linewidth=1)
	
	ax.errorbar(ell_arr, ps_interp, fidu_std, color='r',label='Interpolation')# [%.2f, %.2f, %.2f] chi^2=%.2f' % (best_fit[0],best_fit[1],best_fit[2], chisq),linewidth=1)# 
	#ax.set_title('[%.3f, %.3f, %.3f] $\chi^2$=%.2f' % (best_fit[0],best_fit[1],best_fit[2], chisq))
	#ax2.plot(ell_arr,ps_interp/obs-1,'r',linewidth=1)
	ax2.errorbar(ell_arr,zeros(len(ell_arr)),fidu_std/fidu_avg,color='r')
	
	
	######### theoretical std
	ax.errorbar(ell_arr, sf1_avg, sf1_std,color='c',label='subfield1',linewidth=1)
	ax2.errorbar(ell_arr, zeros(len(ell_arr)), sf1_std/sf1_avg, color='c',label='subfield1',linewidth=1)
	
	#ax.errorbar(ell_arr, ps_interp, theory_std*obs,color='g',label='Theory',linewidth=1)
	#ax2.errorbar(ell_arr, zeros(len(ell_arr)), theory_std, color='g',label='Theory',linewidth=1)
	
	ax.errorbar(ell_arr, sf1_avg, 1/sqrt(N_sim)*sf1_avg,color='k',label='Theory_sim',linewidth=1)
	ax2.errorbar(ell_arr, zeros(len(ell_arr)), 1/sqrt(N_sim),color='k',label='Theory_sim',linewidth=1)
	
	## next 5 lines for ps only
	if ps_replaced_with_pk == 0:
		ax.set_xscale('log')
		ax2.set_xscale('log')
		ax.set_yscale('log')
		ax.set_xlim(ell_arr[0],ell_arr[-1])
		ax2.set_xlim(ell_arr[0],ell_arr[-1])
		#ax.set_ylim(2e-5, 1e-2)
		#ax.set_ylim(6e-5, 5e-4)
		ax.set_xlabel('ell')
		ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$')
		ax2.set_ylabel(r'$\rm{{\Delta}P/P}$')
		ax2.set_xlabel(r'$\ell$')
	#ax.set_xlim(ell_arr[3],ell_arr[13])
	else:
		ax.set_ylabel('N')
		ax2.set_ylabel('$\Delta$ N/N')
	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
	leg.get_frame().set_visible(False)
	plt.setp(ax.get_xticklabels(), visible=False) 
	plt.subplots_adjust(hspace=0.0)

	savefig(plot_dir+'ps_analytical_err_sim.jpg')	#savefig('/Users/jia/Dropbox/test_peak_powspec/Likelihood_cubes/test_peaks_CFHT_interp_andrea.jpg')#'test_ps_5bins_%.3f_%.3f_%.3f.jpg' % (best_fit[0],best_fit[1],best_fit[2]))
	#savetxt('/Users/jia/Dropbox/test_peak_powspec/Likelihood_cubes/jia/CFHT_peaks_interp_50bins_kappa-0.04_0.12.txt',array([x,obs,ps_interp, fidu_std]).T,header='ell_arr\tCFHT_ps\tInterp_ps\tstd')
	close()
	
	
def drawContour2D (H, ititle, xvalues, yvalues, levels=[0.68,0.955, 0.997], handdrawpatch = 0):
	'''draw a contour for a image for levels, title is the title and filename, x and y values are the values at dimentions 0, 1 bin center.
	if handdrawpatch=1, highlight all the pathces within 1st level, instead of rely on contour routine.'''
	fn = plot_dir+ititle+'.jpg'
	H /= float(sum(H))
	H[isnan(H)]=0
	#find 68%, 95%, 99%
	idx = np.argsort(H.flat)[::-1]
	H_sorted = H.flat[idx]
	H_cumsum = np.cumsum(H_sorted)
	#idx10 = where(abs(H_cumsum-0.1)==amin(abs(H_cumsum-0.1)))[0]
	idx68 = where(abs(H_cumsum-0.683)==amin(abs(H_cumsum-0.683)))[0]	
	idx95 = where(abs(H_cumsum-0.955)==amin(abs(H_cumsum-0.955)))[0]
	idx99 = where(abs(H_cumsum-0.997)==amin(abs(H_cumsum-0.997)))[0]
	#v10 = float(H.flat[idx[idx10]])
	v68 = float(H.flat[idx[idx68]])
	v95 = float(H.flat[idx[idx95]])
	v99 = float(H.flat[idx[idx99]])
	#print 'v68, v95, v99',v68, v95, v99
	X, Y = np.meshgrid(xvalues, yvalues)
	V = [v68, v95, v99]
	figure(figsize=(6,8))
	
	#if handdrawpatch:
		#H[where(H>v68)]=1000
	im=imshow(H.T, interpolation='bilinear', origin='lower', cmap=cm.gray, extent=(xvalues[0], xvalues[-1], yvalues[0], yvalues[-1]))
	CS=plt.contour(X, Y, H.T, levels=V, origin='lower', extent=(xvalues[0], xvalues[-1], yvalues[0], yvalues[-1]), colors=('m', 'r', 'green', 'blue'), linewidths=2)#, (1,1,0), '#afeeee', '0.5'))
	#levels = [0.68, 0.95, 0.99]
	#plt.clabel(CS, levels, inline=1, fmt='%1.2f', fontsize=14)
	xlabel('omega_m')
	ylabel('simga_8')
	title(ititle)
	savefig(fn)
	close()
	
if draw_contour_chisq_map:
	l=100
	ll=102
	cut=86#-1
	si0, si1 = 0,-10
	om_arr = linspace(0,1.2,l)[:cut]
	si8_arr = linspace(0,1.6,ll)[si0:si1]
	w_arr = linspace(0,-3,101)
	
		
		
	if ps_replaced_with_pk:
		if bad_pointings:
			fn = emu_dir+'chisq_cube_CFHT_pk_sigmaG%02d.fit'%(sigmaG*10)
			ap = 'peaks_GoodFileds_sigmaG%02d'%(sigmaG*10)
		else:
			fn = emu_dir+'chisq_cube_peaks_x39.fit'
			ap = 'peaks_AllFields'
		
	elif combined_ps_pk:
		if bad_pointings:
			fn = emu_dir+'chisq_cube_CFHT_combined_sigmaG%02d.fit'%(sigmaG*10)
			ap = 'combined_GoodFields_sigmaG%02d'%(sigmaG*10)
		else:
			fn = emu_dir+'chisq_cube_combined.fit'
			ap = 'combined_AllFields_sigmaG%02d'%(sigmaG*10)
	elif ps_remove_4bins:
		fn = emu_dir+'chisq_cube_ellcut%s.fit'%(ellcut)
		ap = 'ps_AllFields_cut@%iarcmin'%(360*60.0/ell_arr0[ellcut])
	else:
		if bad_pointings:
			fn = emu_dir+'chisq_cube_CFHT_ps.fit'
			ap = 'ps_GoodFileds'
		else:
			fn = emu_dir+'chisq_cube_ps_x39.fit'
			ap = 'ps_AllFields'
		
	chisq_cube = WLanalysis.readFits(fn).reshape(-1,l,ll)
	w0, w1=16,70
	chisq_cube = chisq_cube[w0:w1,:cut,si0:si1]
	if not bad_pointings:
		chisq_cube*=39
	#chisq_cube = 39*chisq_cube[17:67,:cut,:]

	P = sum(exp(-chisq_cube/2),axis=0)
	P /= sum(P)
	
	
	############ draw summed heat map #####################
	#drawContour2D(P, 'contour_3D_%s_w-0.5_-2'%(ap), xvalues=om_arr, yvalues=si8_arr)
	drawContour2D(P, 'contour_3D_%s_wcut'%(ap), xvalues=om_arr, yvalues=si8_arr)
	#for i in range(101):
		#print i
		#try:
			#drawContour2D(exp(-chisq_cube[i]/2), 'contour_2D_%s_w%.2f'%(ap,w_arr[i]), xvalues=om_arr, yvalues=si8_arr)
		#except Exception:
			#print 'fail'
	
	
	### plot sample points from the banana as a function of w
	#P_sample = sort(P.flatten())[-200::30]	
	#f=figure(figsize=(6,8))
	##ax1=f.add_subplot(121)
	#ax2=f.add_subplot(111)
	
	##points = [[21,50],[-1,14]]#,[84,64]]
	#points = [where(P==iP) for iP in P_sample] 
	##ax1.imshow(P.T, interpolation='bilinear', origin='lower', cmap=cm.gray, extent=(0,1.2,0,1.6))
	##ax1.colorbar()	
	#for ipoint in points:
		#i,j = ipoint
		#ichisq = chisq_cube[:,i,j]
		#ax2.plot(w_arr,ichisq,label='[%.2f, %.2f]'%(om_arr[i],si8_arr[j]))
	#leg=ax2.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
	#leg.get_frame().set_visible(False)
	#ax2.set_xlabel('w')
	#ax2.set_ylabel(r'$\chi^2$')
	#savefig(plot_dir+'chisq_vs_w_sample.jpg')
	#close()

if draw_contour_smoothed_MCMC_map:
	
	import matplotlib.cm as cm
	#samples = WLanalysis.readFits(emu_dir+'MCMC_samples_CFHT_10000steps.fit')
	samples = WLanalysis.readFits(emu_dir+'MCMC_samples_CFHT_2000steps.fit')
	om, w, si8 = samples.T
	
	H0, xedges, yedges = histogram2d(om, si8, bins=100)#H shape=(nx,ny)
	
	
	def smoothed_contour (pix):
		H = WLanalysis.smooth(H0, pix)
		H /= float(sum(H))
		#find 68%, 95%, 99%
		idx = np.argsort(H.flat)[::-1]
		H_sorted = H.flat[idx]
		H_cumsum = np.cumsum(H_sorted)
		idx68 = where(abs(H_cumsum-0.683)==amin(abs(H_cumsum-0.683)))[0]	
		idx95 = where(abs(H_cumsum-0.955)==amin(abs(H_cumsum-0.955)))[0]
		idx99 = where(abs(H_cumsum-0.997)==amin(abs(H_cumsum-0.997)))[0]
		v68 = float(H.flat[idx[idx68]])
		v95 = float(H.flat[idx[idx95]])
		v99 = float(H.flat[idx[idx99]])
		X, Y = np.meshgrid(xedges[:-1]+0.5*(xedges[1]-xedges[0]), yedges[:-1]+0.5*(yedges[1]-yedges[0]))
		V = [v68, v95, v99]
		
		figure(figsize=(6,8))
		im=imshow(H.T, interpolation='bilinear', origin='lower', cmap=cm.gray, extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]))

		CS=plt.contour(X, Y, H.T, levels=V, origin='lower', extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]))
		levels = [0.68, 0.95, 0.99]
		#plt.clabel(CS, levels, inline=1, fmt='%1.2f', fontsize=14)
		xlabel('omega_m')
		ylabel('simga_8')
		title('Smoothed over %s pixels'%(pix))
		savefig(plot_dir+'CFHT_MCMC_fixcontour_ps_pix%s.jpg'%(pix))
		close()
	map(smoothed_contour, (0.5, 1, 2, 5, 10))
	
if dC_dp:
	for i in range(len(ps_mat)):
		iom, iw, isi8 = cosmo_params[i]
		#icov_mat = mat(cov(ps_mat[i],rowvar=0))
		#icov_inv = icov_mat.I
		#ifrac = array(icov_inv/cov_inv-1)
		icov_mat = cov(ps_mat[i],rowvar=0)
		ifrac = array(icov_mat/cov_mat-1)
		plotimshow(ifrac, 'dCmat_dp_lim45%.3f_%.3f_%.3f'%(iom,iw,isi8),vmin=-45, vmax=45)

if CFHT_ps_5bins:
	print 'CFHT_ps_5bins'	
	#l = 100
	#m = 102
	l=50
	m=50
	om_arr = linspace(0,1.2,l)
	si8_arr = linspace(0,1.6,m)
	r_arr = linspace(0.91, 0.98, 8)
	def plot_heat_map_r (r):
		print r
		obs = ps_CFHT.copy()
		obs [6:11] = r*obs[6:11]
		heatmap = zeros(shape=(l,m))
		for i in range(l):
			for j in range(m):
				best_fit = (om_arr[i], -1.5, si8_arr[j])
				ps_interp = interp_cosmo(best_fit)	
				del_N = np.mat(ps_interp - obs)
				chisq = float(del_N*cov_inv*del_N.T)/39.0
				heatmap[i,j] = chisq
		P = exp(-heatmap/2)
		P /= sum(P)
		drawContour2D(P, 'test_ps_contour_ratio%s'%(r), xvalues=om_arr, yvalues=si8_arr)
		
	#map(plot_heat_map_r, r_arr)
	

if varying_C:
	l=50
	ll=50
	om_arr = linspace(0,1.2,l)
	si8_arr = linspace(0,1.5,ll)
	
	##########################################
	#interpolate covariance matrix
	#cov_arr=[[],]*91
	#for i in range(91):
		#cov_arr[i] = cov(ps_mat[i],rowvar=0)
	#cov_flatten = array(cov_arr).reshape(91,-1)
	#cov_interps = list()
	#for ibin in range(cov_flatten.shape[-1]):
		#cov_model = cov_flatten[:,ibin]
		#iinterp = interpolate.Rbf(m, w, s, cov_model)
		#cov_interps.append(iinterp)
	#def interp_inv(params):
		#im, wm, sm = params
		#gen_cov = lambda ibin: cov_interps[ibin](im, wm, sm)
		#cov_interp = array(map(gen_cov, range(cov_flatten.shape[-1])))
		#cov_interp = cov_interp.reshape(39,39).squeeze()
		#return cov_interp
	##########################################
	
	##########################################
	##interpolate covariance inverse
	#cov_inv_arr=[[],]*91
	#for i in range(91):
		#cov_inv_arr[i] = mat(cov(ps_mat[i],rowvar=0)).I
	
	#cov_inv_flatten = array(cov_inv_arr).reshape(91,-1)
	#cov_inv_interps = list()
	#for ibin in range(cov_inv_flatten.shape[-1]):
		#cov_model = cov_inv_flatten[:,ibin]
		#iinterp = interpolate.Rbf(m, w, s, cov_model)
		#cov_inv_interps.append(iinterp)
	#def interp_cov_inv(params):
		#im, wm, sm = params
		#gen_cov = lambda ibin: cov_inv_interps[ibin](im, wm, sm)
		#cov_interp = array(map(gen_cov, range(cov_inv_flatten.shape[-1])))
		#cov_interp = cov_interp.reshape(39,39).squeeze()
		#return cov_interp
	##########################################
	
	# find nearest point to interpolation point
	#def find_nearest_cosmo(best_fit):
		#best_fit = array(best_fit)
		#dist = sum((best_fit - cosmo_params)**2,axis=1)
		#return argmin(dist)
	
	# constant covariance inverse
	obs = ps_CFHT.copy()
	#obs = ps_avg[48]
	heatmap = zeros(shape=(l,ll))
	##for idx in range(91):
	for i in range(l):
		for j in range(ll):
			print i,j
			best_fit = (om_arr[i], -1.0, si8_arr[j])
			ps_interp = interp_cosmo(best_fit)#,method='GP')	
			del_N = np.mat(ps_interp - obs)
			##############################################
			#### next 3 lines use varying covariance matrix
			#idx = find_nearest_cosmo(best_fit)#, interp=True)
			#print 'best_fit, idx', best_fit, idx
			#cov_inv = cov_inv_arr[idx]
			##############################################
			
			##############################################
			### next 1 line interpolate covariance matrix ##
			#cov_inv = interp_cov_inv(best_fit) 	
			#cov_mat = interp_inv(best_fit)
			#cov_inv = mat(cov_mat).I
			##############################################
			chisq = float(del_N*cov_inv*del_N.T)#/39.0
			heatmap[i,j] = chisq
	P = exp(-heatmap/2)
	P /= sum(P)
	#drawContour2D(P, 'test_goodfields_contour_fidu_pk_sigmaG%02d'%(sigmaG*10), xvalues=om_arr, yvalues=si8_arr)
	drawContour2D(P, 'pk_BAD_25bins_w-1_mean', xvalues=om_arr, yvalues=si8_arr)

if CFHT2pcf:
	theta,xi_plus,sigma_plus,xi_minus,sigma_minus = genfromtxt('/Users/jia/CFHTLenS/2PCF/CFHT2pcf').T
	#theta to ell
	ell_CFHT = 360.0*60/theta
	
	
	obs = ps_CFHT.copy()
	f=figure(figsize=(8,8))
	ax=f.add_subplot(111)
	
	ax.errorbar(ell_arr,obs,fidu_std,color='k',label='Power Spectrum',linewidth=1)
	ax.errorbar(ell_CFHT, xi_plus, sqrt(sigma_plus),label='xi_+',linewidth=1)
	ax.errorbar(ell_CFHT, xi_minus, sqrt(sigma_minus),label='xi_-',linewidth=1)
	
	## next 5 lines for ps only
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('ell')
	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
	leg.get_frame().set_visible(False)
	show()

	#savefig(plot_dir+'compare_CFHT2pcf.jpg')
	#close()

########################################################################
########################################################################
################ official plots ########################################
if CFHT_fields_and_masks:
	sigmaG=1.0
	centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
	
	kmapGen = lambda sigmaG, Wx: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/W%i_KS_1.3_lo_sigmaG%02d.fit'%(Wx, sigmaG*10))
	
	def maskGen(sigmaG, Wx):
		galn = WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/W%i_galn_1.3_lo_sigmaG%02d.fit'%(Wx, sigmaG*10))
		mask = ones(shape=galn.shape)
		mask_smooth = WLanalysis.smooth(galn.astype(float), sigmaG*PPA512)
		idx = where (mask_smooth < 5/PPA512**2)
		mask[idx] = 0
		Wxsize = mask.shape[0]
		x0, x1 = nonzero(sum(mask,axis=1))[0][[0,-1]]
		y0, y1 = nonzero(sum(mask,axis=0))[0][[0,-1]]
		xcut = int(Wxsize - (x1-x0))
		ycut = int(Wxsize - (y1-y0))
		return mask, x0, y0, x1, y1
	
	def drawSFlines(kmap, Wx,lw=5):
		sfmap = zeros(shape=kmap.shape)
		if Wx == 1:
			sfmap[512:512+lw,:]=nan
			sfmap[512*2:512*2+lw,:]=nan
			sfmap[:,512:512+lw]=nan
			sfmap[:512*2,512*2:512*2+lw]=nan
			sfmap[512*2+lw:,512+lw:]=-100#not used region
		if Wx == 2:
			sfmap[512:512+lw,:]=nan
			sfmap[:,512:512+lw]=nan
			sfmap[512+lw:,512+lw:]=-100#not used region
		if Wx == 3:
			sfmap[kmap.shape[0]/2:kmap.shape[0]/2+lw,:]=nan
			sfmap[:,kmap.shape[1]/2:kmap.shape[1]/2+lw]=nan
		if Wx == 4:
			sfmap[512:512+lw,:512]=nan
			sfmap[:,512:512+lw]=nan
			sfmap[-512-lw:-512,512:]=nan
			#not used region
			sfmap[512+lw:,:512]=-100
			sfmap[:-512-lw,512+lw:]=-100
		return sfmap
			
	def fieldGen (Wx, sigmaG=sigmaG):
		
		kmap0 = kmapGen(sigmaG, Wx)
		print 'W'+str(Wx), kmap0.shape
		mask, x0, y0, x1, y1 = maskGen(sigmaG, Wx)
		kmap0 *= mask
		kmap = kmap0.copy()[x0-1:x1+1,y0-1:y1+1]
		print 'W'+str(Wx), 'after', kmap.shape
		
		sfmap = drawSFlines(kmap, Wx)
		
		kmap[where(kmap==0)]=nan#-1000#
		#kmap += sfmap # turn on to draw devision lines
		return kmap
	
	
	kmaps = map(fieldGen,range(1,5))

	i=0
	f=figure(figsize=(15,12))
	for img in kmaps:
		x0=centers[i][0]+img.shape[1]/2/PPA512/60
		x1=centers[i][0]-img.shape[1]/2/PPA512/60
		y0=centers[i][1]-img.shape[0]/2/PPA512/60
		y1=centers[i][1]+img.shape[0]/2/PPA512/60
		
		ax=f.add_subplot(2,2,i+1)
		#cmap = matplotlib.cm.gnuplot#jet#terrain#hot#cool#gist_earth
		cmap = matplotlib.cm.jet
		cmap.set_bad('w',1.)
		im=imshow(img,cmap=cmap, origin='lower',vmin=-0.05,aspect=1,vmax=0.08,extent=[x0,x1,y0,y1])#interpolation='nearest'
		title(r'$\rm{W%i}$'%(i+1),fontsize=24)
		ax.set_xlabel(r'$\rm{RA [deg]}$',fontsize=24)
		ax.set_ylabel(r'$\rm{Dec [deg]}$',fontsize=24)
		ax.set_aspect('equal', 'datalim')
		ax.tick_params(labelsize=18)
		matplotlib.pyplot.locator_params(nbins=6)
		i+=1
	
	f.subplots_adjust(left=0.08, right=0.83,wspace=0.25,hspace=0.3)
	cbar_ax = f.add_axes([0.88, 0.1, 0.03, 0.8])
	cbar_ax.tick_params(labelsize=18) 
	f.colorbar(im, cax=cbar_ax)
	#savefig(plot_dir+'official/KS_allsubfields_sigmaG%02d.pdf'%(sigmaG*10))
	savefig(plot_dir+'tSZxCFHT/KS_allsubfields_sigmaG%02d.pdf'%(sigmaG*10))
	close()	
		#chop kmap
## contours
l=100
ll=102

om0,om1 = 0, 67#0, 67#
si80,si81 = 20,85#-10#18, 85#0,-10#
w0,w1 = 10,70#int(sys.argv[1]), int(sys.argv[2])#15,70#10,70#4, 70#10,70#4, 70#85

om_arr = linspace(0,1.2,l)[om0:om1]
si8_arr = linspace(0,1.6,ll)[si80:si81]
w_arr = linspace(0,-3,101)[w0:w1]

colors=('r','b','m','c','k','g','r','c','b','g','m','k')
seed(25)
colors2=rand(10,3)
lss =('solid', 'dashed', 'solid', 'dashed', 'dashdot', 'dotted','dashdot', 'dotted')
lss2 = ('dotted', 'dashed','solid','dashdot','dashed','solid','solid')
lss3=('-','-.','--',':','.')
lws = (4,4,2,2,4,4,2,2)
lws2 = (4,4,2,2,4)

def findlevel (H):
	H /= sum(H)
	H /= float(sum(H))
	H[isnan(H)]=0
	#find 68%, 95%, 99%
	idx = np.argsort(H.flat)[::-1]
	H_sorted = H.flat[idx]
	H_cumsum = np.cumsum(H_sorted)
	#idx10 = where(abs(H_cumsum-0.1)==amin(abs(H_cumsum-0.1)))[0]
	idx68 = where(abs(H_cumsum-0.683)==amin(abs(H_cumsum-0.683)))[0]	
	idx95 = where(abs(H_cumsum-0.955)==amin(abs(H_cumsum-0.955)))[0]
	idx99 = where(abs(H_cumsum-0.997)==amin(abs(H_cumsum-0.997)))[0]
	#v10 = float(H.flat[idx[idx10]])
	v68 = float(H.flat[idx[idx68]])
	v95 = float(H.flat[idx[idx95]])
	v99 = float(H.flat[idx[idx99]])
	#print 'v68, v95, v99',v68, v95, v99
	V = [v68, v95, v99]
	return V

def ProbPlan(sigmaG):
	cube_fn = emu_dir+'chisq_cube_CFHT_pk_sigmaG%02d.fit'%(sigmaG*10)
	chisq_cube = WLanalysis.readFits(cube_fn).reshape(-1,l,ll)
	chisq_cube = chisq_cube[w0:w1,om0:om1,si80:si81]
	P = sum(exp(-chisq_cube/2),axis=0)
	P /= sum(P)
	return P

def cube2P(chisq_cube, axis=0):#aixs 0-w, 1-om, 2-si8
	if axis==0:
		chisq_cube = chisq_cube[w0:w1,om0:om1,si80:si81]
	else:
		chisq_cube = chisq_cube[:,om0:om1,si80:si81]
	P = sum(exp(-chisq_cube/2),axis=axis)
	P /= sum(P)
	return P

if contour_peaks_smoothing or contour_peaks_fieldselect:
	if contour_peaks_smoothing:
		Ppk2 = cube2P(WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/chisq_cube_CFHT_pk_sigmaG1018.fit').reshape(-1,l,ll))
		
		sigmaG_arr2 = sigmaG_arr[1:5]
		P_arr = list(map(ProbPlan, sigmaG_arr2))
		P_arr.append(Ppk2)
		labels = [r'$\rm{%.1f\, arcmin}$'%(sigmaG) for sigmaG in sigmaG_arr2]
		labels.append(r'$\rm{1.0+1.8\, arcmin}$')
		
		fn = '/Users/jia/weaklensing/CFHTLenS/plot/official/contour_peaks_smoothing.pdf'
	if contour_peaks_fieldselect:
		Ppk_all = cube2P(WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/chisq_cube_CFHT_pk_sigmaG10.fit').reshape(-1,l,ll))
	
		Ppk_pass = cube2P(WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/GoodOnly/chisq_cube_CFHT_pk_sigmaG10.fit').reshape(-1,l,ll))
		
		P_arr = [Ppk_pass, Ppk_all]
		labels = labels = [r'$\rm{pass\, fields}$', r'$\rm{all\, fields}$']
		fn = '/Users/jia/weaklensing/CFHTLenS/plot/official/contour_peaks_fieldselect.pdf'
	
	f = figure(figsize=(8,8))
	ax=f.add_subplot(111)
	lines=[]
	X, Y = np.meshgrid(om_arr, si8_arr)
	
	for i in arange(len(P_arr)):
		P=P_arr[i]
		V=findlevel(P)
		CS = ax.contour(X, Y, P.T, levels=[V[0],], origin='lower', extent=(om_arr[0], om_arr[-1], si8_arr[0], si8_arr[-1]), colors=colors[i], linewidths=lws2[i+1], linestyles=lss2[i+1])
		lines.append(CS.collections[0])
	
	########## add comb
	##i=0
	##Ppk_10, Ppk_18 = map(ProbPlan, [1.0,1.8])
	##for P in (Ppk2, Ppk_10*Ppk_18):
		##i+=1
		##Vc = findlevel(Ppk2)
		##CS = ax.contour(X, Y, Ppk2.T, levels=[Vc[0],], origin='lower', extent=(om_arr[0], om_arr[-1], si8_arr[0], si8_arr[-1]), colors=colors[i], linewidths=lws[i])
		##lines.append(CS.collections[0])
	##labels = ['With cross term in Cov_mat', 'Without cross term']
	###labels.append('Ppk2')
	###labels.append('Ppk_10*Ppk_18')

	leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':20},loc=0)
	ax.tick_params(labelsize=16)
	ax.set_xlabel(r'$\rm{\Omega_m}$',fontsize=20)
	ax.set_ylabel(r'$\rm{\sigma_8}$',fontsize=20)
	leg.get_frame().set_visible(False)
	#show()
	savefig(fn)
	close()
	
if contour_peaks_powspec:
	sigmaG_arr2=[1.0, 1.8]
	
	# individual probabilities
	#Ppk_10, Ppk_18 = map(ProbPlan, sigmaG_arr2)	
	#Ppk = Ppk_10 * Ppk_18

	#Pps = cube2P(WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/Goodonly/chisq_cube_CFHT_ps.fit').reshape(-1,l,ll))
	
	if include_w:
		w_arr = linspace(0,-3,101)
		axis = 2
		ix,iy = 1, 0#0, 1
		X, Y = np.meshgrid(w_arr, om_arr)
		fn='/Users/jia/weaklensing/CFHTLenS/plot/official/contour_peaks_powspec_w.pdf'
	else:
		axis = 0
		ix,iy = 0,2
		X, Y = np.meshgrid(om_arr, si8_arr)
		fn='/Users/jia/weaklensing/CFHTLenS/plot/official/contour_peaks_powspec.pdf'
	Ppk2 = cube2P(WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/chisq_cube_CFHT_pk_sigmaG1018.fit').reshape(-1,l,ll),axis=axis)
	
	Pps = cube2P(WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/GoodOnly/chisq_cube_CFHT_psellcut0_26.fit').reshape(-1,l,ll),axis=axis)
	
	# combined analysis
	#chisq_cube_comb = WLanalysis.readFits( '/Users/jia/CFHTLenS/emulator/GoodOnly/chisq_cube_CFHT_combined_sigmaG1018.fit').reshape(-1,l,ll)
	#Pc = cube2P(chisq_cube_comb-amin(chisq_cube_comb)+2)
	
	P_arr=[Pps, Ppk2, Pps*Ppk2]#, Pc]#, Ppk]#
	labels = [r'$\rm{power\, spectrum}$', r'$\rm{peaks\, (1.0 + 1.8\,arcmin)}$', r'$\rm{power\, spectrum + peaks}$']#,'actual comb']#, 'Ppk_10 * Ppk_18']#,'actual comb','peaks comb']
	
	f = figure(figsize=(8,8))
	ax=f.add_subplot(111)
	lines=[]
	
	
	for i in range(len(P_arr)):
		print i
		P=P_arr[i]
		V=findlevel(P)
		A = float(P.shape[0]*P.shape[1])
		print 'include_w', bool(include_w), i, len(where(P>V[0])[0])/A, len(where(P>V[1])[0])/A
		
		if i < 2:
			CS = ax.contour(X, Y, P.T, levels=[V[0],], origin='lower', extent=(om_arr[0], om_arr[-1], si8_arr[0], si8_arr[-1]), colors=colors[-3+i], linewidths=lws[i*2], linestyles=lss2[-3+i])
			CS2 = ax.contour(X, Y, P.T, levels=[V[1],], alpha=0.7, origin='lower', extent=(om_arr[0], om_arr[-1], si8_arr[0], si8_arr[-1]), colors=colors[-3+i], linewidths=lws[i*2], linestyles=lss2[-3+i])
		else:
			CS = ax.contourf(X, Y, P.T, levels=V[:-1], origin='lower', extent=(om_arr[0], om_arr[-1], si8_arr[0], si8_arr[-1]), colors=colors[-3+i], linewidths=lws[i*2], linestyles=lss2[-3+i], alpha=0.85)
			CS2 = ax.contour(X, Y, P.T, levels=[V[1],], alpha=0.7, origin='lower', extent=(om_arr[0], om_arr[-1], si8_arr[0], si8_arr[-1]), colors=colors[-3+i], linewidths=lws[i*2], linestyles=lss2[-3+i])
		
		#lines.append(CS.collections[0])

	#leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':20},loc=0)
	#leg.get_frame().set_visible(False)
	ax.tick_params(labelsize=16)
	ax.set_xlabel(cosmo_labels[ix],fontsize=20)
	ax.set_ylabel(cosmo_labels[iy],fontsize=20)
	
	#show()
	savefig(fn)#_wcut%.1f_%.1f_combined.pdf'%(w_arr[0],w_arr[-1]))
	#savefig(plot_dir+'official/contour_peaks_powspec_wcut%.1f_%.1f.pdf'%(w_arr[0],w_arr[-1]))	##savefig(plot_dir+'official/contour_peaks_powspec_allsky_%iarcmin.pdf'%(360*60.0/ell_arr0[18]))
	close()

if sample_interpolation:

	lw=3
	ps_interp_GP = interp_cosmo(fidu_params, method = 'GP')
	ps_interp_spline = interp_cosmo(fidu_params)
	
	f=figure(figsize=(10,8))
	ax=f.add_subplot(gs[0])
	ax2=f.add_subplot(gs[1],sharex=ax)
		
	ax.errorbar(ell_arr, fidu_avg, fidu_std, color='k', linewidth=lw)
	ax.plot(ell_arr, fidu_avg, 'k-', label=r'$\rm{ True}$',linewidth=lw)
	ax.plot(ell_arr, ps_interp_spline, 'm--', label=r'$\rm{ RBF}$',linewidth=lw)
	ax.plot(ell_arr, ps_interp_GP, 'b:', label=r'$\rm{ GP}$',linewidth=lw)

	ax2.errorbar(ell_arr, zeros(len(ell_arr)),fidu_std/fidu_avg, color='k',label=r'$\rm{ True}$',linewidth=lw)
	ax2.plot(ell_arr, ps_interp_spline/fidu_avg-1, 'm--', label=r'$\rm{ RBF}$',linewidth=lw)
	ax2.plot(ell_arr, ps_interp_GP/fidu_avg-1, 'b:', label=r'$\rm{ GP}$',linewidth=lw)

	ax2.set_ylim(-0.1, 0.1)
	ax2.set_yticks(linspace(-0.05,0.05,3))

	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':20},loc=2)
	leg.get_frame().set_visible(False)

	ax.set_title(r'$\rm{[\Omega_m,\,{\rm w},\,\sigma_8] = [%.3f,\, %.3f,\, %.3f]}$' % (fidu_params[0],fidu_params[1],fidu_params[2])+'\n', fontsize=20)
	plt.setp(ax.get_xticklabels(), visible=False) 
	plt.subplots_adjust(hspace=0.0,left=0.15)#, top=0.8)
	ax.set_xlim(ell_arr[0],ell_arr[-1])
	ax.tick_params(labelsize=16)
	ax2.tick_params(labelsize=16)
	if not ps_replaced_with_pk:
		ax2.set_xlabel(r'$\ell$',fontsize=20)
		ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$',fontsize=20)
		ax2.set_ylabel(r'$\rm{{\Delta}P/P}$',fontsize=20)
		ax2.set_xscale('log')
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.set_ylim(1e-5, 1e-2)
		#show()
		savefig(plot_dir+'sample_interpolation_ps_sigmaG%02d_%i.jpg'%(sigmaG*10,icosmo))
		#savefig(plot_dir+'official/sample_interpolation_ps_sigmaG%02d.pdf'%(sigmaG*10))
		
	else:
		ax.set_ylabel(r'$\rm{peak\, counts\, N(\kappa)}$',fontsize=20)
		ax2.set_ylabel(r'$\rm{{\Delta}N/N}$',fontsize=20)
		ax2.set_xlabel('$\kappa$',fontsize=20)
		savefig(plot_dir+'sample_interpolation_pk_sigmaG%02d_%i.jpg'%(sigmaG*10,icosmo))
		#savefig(plot_dir+'official/sample_interpolation_pk_sigmaG%02d.pdf'%(sigmaG*10))
	close()

labels=('\n'+r'$\rm{\Omega_m}$','\n'+r'$\rm{w}$','\n'+r'$\rm{\sigma_8}$')
cmaps = ('spring','summer','winter','hot')
if interp_2D_plane:
	iside=40
	ibin = int(sys.argv[1])#-20
	#ibins = [5, 10, 12, 19] 
	w_dummy = -ones(shape=(iside,iside))
	s_dummy = 0.8*ones(shape=(iside,iside))
	m_dummy = 0.26*ones(shape=(iside,iside))
	ijs = [[1,0],[1,2],[0,2]]#k=2,0,1
	#zlims = [[8,11.5], [8,16],[8,12.5]]#zlim for powspec
	dummys = [m_dummy,w_dummy,s_dummy]
	fs=24#fontsize
	seed(99)
	ll = 1
	fig = figure(figsize=(22,7))
	for ij in ijs:
		print ij
		i, j = ij#[::-1]
		X, Y = meshgrid(linspace(params_min[i],params_max[i],iside), linspace(params_min[j],params_max[j],iside))	
		#ps = ps_avg.T[ibin]
		#ps_interp = interpolate.Rbf(m, w, s, ps, smooth = 0)
		ps_interp = lambda ibin: interpolate.Rbf(m, w, s, ps_avg.T[ibin], smooth = 0)
		Xarr = [0,]*3
		k = delete(range(3),ij)[0]
		Xarr[i] = X
		Xarr[j] = Y
		Xarr[k] = dummys[k]
		x, y, z = Xarr
		#print 'x, y, z',x.shape, y.shape, z.shape
		ell = ell_arr[ibin]
		if ps_replaced_with_pk:
			Z = ps_interp(ibin)(x, y, z)/100.0
		else:
			Z = ps_interp(ibin)(x, y, z)*2*pi/ell/(ell+1)*1e10
		
		
		ax = fig.add_subplot(1,3,ll, projection='3d')
		ll+=1
		#ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, cmap='hot')#color=rand(3)
		offset=amin(Z)*0.9
		ax.contourf(X, Y, Z, zdir='z', cmap=cmaps[k], offset=offset)
		ax.plot_surface(X, Y, Z, cmap=cmaps[k],alpha=0.7, rstride=int(0.1*iside), cstride=int(0.1*iside))
		ax.set_zlim(offset,)
		#z0,z1=zlims[k]
		#ax.set_zlim(z0,z1)
		ax.set_xlabel(labels[i],fontsize=fs)
		ax.set_ylabel(labels[j],fontsize=fs)
		
		ax.tick_params(labelsize=fs-4)
		#ax.zaxis.tick_top()
		#ax.ticklabel_format(axis='z', style='sci', scilimits=(-2,2))
		matplotlib.pyplot.locator_params(nbins=4)
		
		if ps_replaced_with_pk:
			ax.tick_params(axis='z', direction='out', pad=5)
			ax.set_zlabel('\n'+r'$\rm{N}(\kappa=%.2f)/100$'%(ell_arr[ibin]),fontsize=fs)
			#savefig(plot_dir+'official/interp2D_peaks_%02dbin_%i.pdf'%(ibin,k))
			figfn=plot_dir+'official/interp2D_peaks_%02dbin.pdf'%(ibin)
		else:
			
			ax.set_zlabel('\n'+r'$10^{10}\times\rm{P}(\ell=%i,000)$'%(ell_arr[ibin]/1000.0),fontsize=fs)
			#savefig(plot_dir+'official/interp2D_powspec_%02dbin_%i.pdf'%(ibin,k))
			#figfn=plot_dir+'official/interp2D_powspec_%02dbin_nicaea.pdf'%(ibin)
			figfn='/Users/jia/weaklensing/CFHTLenS/plot/official/interp2D_powspec_%02dbin.pdf'%(ibin)
		plt.subplots_adjust(left=0.0, right=0.95, hspace=0.0, wspace=0.1)
	savefig(figfn)
	close()


if good_bad_powspec:
	#need to turn on bad_pointings for this to plot
	
	if sample_points:
		if ps_replaced_with_pk:
			# need to turn on ps_remove_4bins
			ell_arr_labels = array(['%.2f'%(i) for i in list(ell_arr.copy())*2])[idx999][::5]
			ell_arr = range(len(ps_CFHT))
			#ell_arr_labels = ell_arr
			Pc = exp(-0.5*WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/chisq_cube_CFHT_pk_sigmaG1018.fit').reshape(-1,l,ll))[w0:w1,om0:om1,si80:si81]
			#Pc = Ppk2
		else:
			Pc = exp(-0.5*WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/GoodOnly/chisq_cube_CFHT_psellcut0_26.fit').reshape(-1,l,ll))[w0:w1,om0:om1,si80:si81]
		V = findlevel(Pc)
		
		idx = array(where(Pc>V[0]))
		idxmax = array(where(Pc==amax(Pc))).T
		seed(5)
		best_idx_arr0 = idx.T[[randint(0,len(idx[0])-1,size=2)]]
		best_idx_arr = concatenate((idxmax,best_idx_arr0))
		best_fit_arr = [[om_arr[iidx[1]],w_arr[iidx[0]], si8_arr[iidx[2]] ] for iidx in best_idx_arr]
	else:
		ii=48
		#ps_mat_all = WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/powspec_sum/ps_mat_sigma05.fit').reshape(91,1000,-1)
		#ps_CFHT = ps_avg[ii]#fidu_avg
		#ps_CFHT_all = mean(ps_mat_all[ii],axis=0)
		#ps_CFHT_all = ps_CFHT_all/ps_CFHT_all[20]*ps_CFHT[20]
	###################################
	lw=4
	
	f=figure(figsize=(10,8))
	ax=f.add_subplot(gs[0])
	ax2=f.add_subplot(gs[1],sharex=ax)
	
	if sample_points:
		ax.errorbar(ell_arr, ps_CFHT, fidu_std, color='k', linewidth=lw)
		ax.plot(ell_arr, ps_CFHT, 'k-', label=r'$\rm{CFHTLenS}$',linewidth=lw)	
		ax2.errorbar(ell_arr, zeros(len(ell_arr)),fidu_std/fidu_avg, color='k', linewidth=lw)
		ax2.plot(ell_arr, zeros(len(ell_arr)), 'k-', label=r'$\rm{pass\, fields}$',linewidth=lw)
	else:
		ax.plot(ell_arr, ps_CFHT, 'g-', label=r'$\rm{pass\, fields}$',linewidth=lw)	
		ax.errorbar(ell_arr, ps_CFHT, fidu_std, color='g', linewidth=lw)
		ax.plot(ell_arr, ps_CFHT_all, 'm--', label=r'$\rm{all\, fields}$',linewidth=lw)
		ax2.plot(ell_arr, zeros(len(ell_arr)), 'g-', label=r'$\rm{pass\, fields}$',linewidth=lw)
		ax2.errorbar(ell_arr, zeros(len(ell_arr)),fidu_std/fidu_avg, color='g', linewidth=lw)
		ax2.plot(ell_arr, ps_CFHT_all/ps_CFHT-1,'m--', label=r'$\rm{all\, fields}$',linewidth=lw)
	
	######## sample points ########
	if sample_points:
		i=0
		for best_fit in best_fit_arr:
			ps_interp = interp_cosmo(best_fit)
			del_N = np.mat(ps_interp - ps_CFHT)
			chisq = del_N*cov_inv*del_N.T
			chisq /= (len(ps_interp)-4)
			ilabel= r'$[{\rm\Omega_m},\,\rm{w},\,{\rm\sigma_8}] = [%.2f,\,%.2f,\,%.2f],\,\chi^2/dof=%.2f$' % (best_fit[0],best_fit[1],best_fit[2], chisq)
			ax.plot(ell_arr, ps_interp, lss3[i],color=colors[i], label=ilabel, linewidth=2)
			ax2.plot(ell_arr, ps_interp/ps_CFHT-1, lss3[i], color=colors[i], linewidth=2)
			i+=1
	###########################
	
	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=0)
	leg.get_frame().set_visible(False)

	plt.subplots_adjust(hspace=0.0,left=0.15)
	plt.setp(ax.get_xticklabels(), visible=False) 
	ax.set_xlim(ell_arr[0],ell_arr[-1])
	ax.tick_params(labelsize=16)
	ax2.tick_params(labelsize=16)
	
	if sample_points:
		if ps_replaced_with_pk:
			fn='/Users/jia/weaklensing/CFHTLenS/plot/official/peaks_fit.pdf'
			ax.set_ylabel(r'$\rm{peak\, counts\, N(\kappa)}$',fontsize=20)
			ax2.set_ylabel(r'$\rm{{\Delta}N/N}$',fontsize=20)
			ax2.set_xlabel('$\kappa$',fontsize=20)
			ax2.set_ylim(-0.3, 0.3)
			ax2.set_yticks(linspace(-0.2,0.2,3))
			ax.set_ylim(-1,2700)
			ax.set_xticklabels(ell_arr_labels)
			ax.text(9.2,400,r'$[\rm{1.0\, arcmin}]$',color='k',fontsize=16)
			ax.text(28,1100,r'$[\rm{1.8\, arcmin}]$',color='k',fontsize=16)
		else:
			ax2.set_ylim(-0.1, 0.1)
			ax2.set_yticks(linspace(-0.05,0.05,3))
			fn='/Users/jia/weaklensing/CFHTLenS/plot/official/powspec_fit.pdf'
			ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$',fontsize=20)
			ax.set_xscale('log')
			ax2.set_xscale('log')
			ax2.set_xlabel(r'$\ell$')
			ax.set_yscale('log')
			ax2.set_ylabel(r'$\rm{{\Delta}P/P}$',fontsize=20)
			ax.set_ylim(1e-5, 1e-2)
		#savefig(plot_dir+'official/sample_interpolation_ps_sigmaG%02d.jpg'%(sigmaG*10))
		ax.set_title(r'$\rm{CFHTLenS}$',fontsize=24)
		#savefig('/Users/jia/weaklensing/CFHTLenS/plot/official/good_bad_powspec_simulation%i.pdf'%(ii))
		
		
	else:
		ax2.set_ylim(-0.1, 0.1)
		ax2.set_yticks(linspace(-0.05,0.05,3))
		fn='/Users/jia/weaklensing/CFHTLenS/plot/official/good_bad_powspec_CFHT.pdf'
		ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$',fontsize=20)
		ax.set_xscale('log')
		ax2.set_xscale('log')
		ax2.set_xlabel(r'$\ell$')
		ax.set_yscale('log')
		ax2.set_ylabel(r'$\rm{{\Delta}P/P}$',fontsize=20)
		ax.set_ylim(1e-5, 1e-2)
	savefig(fn)	
	close()

if contour_ps_fieldselect:
	Pps_all = cube2P(39*WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/chisq_cube_ps_x39.fit').reshape(-1,l,ll))
	
	Pps_good = cube2P(WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/GoodOnly/chisq_cube_CFHT_ps.fit').reshape(-1,l,ll))
	
	Pps_all_cut10 = cube2P(39*WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/chisq_cube_ellcut15.fit').reshape(-1,l,ll))
	
	Pps_all_cut7 = cube2P(39*WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/chisq_cube_ellcut18.fit').reshape(-1,l,ll))
	
	Pps_all_cuttil26 = cube2P(WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/chisq_cube_CFHT_psellcut0_26.fit').reshape(-1,l,ll))
	
	Pps_good_cuttil26 = cube2P(WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/GoodOnly/chisq_cube_CFHT_psellcut0_26.fit').reshape(-1,l,ll))

	
	P_arr = (Pps_good, Pps_all, Pps_good_cuttil26, Pps_all_cuttil26)#, Pps_all_cut7))

	f = figure(figsize=(8,8))
	
	ax=f.add_subplot(111)
	
	lines=[]
	X, Y = np.meshgrid(om_arr, si8_arr)

	for i in range(len(P_arr)):
		P=P_arr[i]
		V=findlevel(P)
		CS = ax.contour(X, Y, P.T, levels=[V[0],], origin='lower', extent=(om_arr[0], om_arr[-1], si8_arr[0], si8_arr[-1]), colors=colors[i], linewidths=lws[i], linestyles=lss[i])
		lines.append(CS.collections[0])
	labels = [r'$\rm{pass\, fields}$', r'$\rm{all\, fields}$', r'$\rm{pass\, fields(\ell<7,000)}$', r'$\rm{all\, fields(\ell<7,000)}$']#, r'$\rm{all\, fields(\ell>3,000)}$']

	########## add comb
	##i=0
	##Ppk_10, Ppk_18 = map(ProbPlan, [1.0,1.8])
	##for P in (Ppk2, Ppk_10*Ppk_18):
		##i+=1
		##Vc = findlevel(Ppk2)
		##CS = ax.contour(X, Y, Ppk2.T, levels=[Vc[0],], origin='lower', extent=(om_arr[0], om_arr[-1], si8_arr[0], si8_arr[-1]), colors=colors[i], linewidths=lws[i])
		##lines.append(CS.collections[0])
	##labels = ['With cross term in Cov_mat', 'Without cross term']
	###labels.append('Ppk2')
	###labels.append('Ppk_10*Ppk_18')

	leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':20},loc=0)
	ax.tick_params(labelsize=16)
	ax.set_xlabel(r'$\rm{\Omega_m}$',fontsize=20)
	ax.set_ylabel(r'$\rm{\sigma_8}$',fontsize=20)
	leg.get_frame().set_visible(False)
	#show()
	savefig('/Users/jia/weaklensing/CFHTLenS/plot/official/contour_powespec_fieldselection.pdf')#_w_%s_%s.pdf'%(w0,w1))
	close()

if good_bad_peaks:
	sigmaG = 1.0
	
	## CFHTLenS
	ps_CFHT_mat_fn = '/Users/jia/CFHTLenS/emulator/goodonly/CFHT_peaks_sigma%02d.fit'%(sigmaG*10)
	ps_CFHT_all_mat_fn = '/Users/jia/CFHTLenS/CFHTKS/CFHT_peaks_sigma%02d_025bins.fit'%(sigmaG*10)
	ps_CFHT = sum(WLanalysis.readFits(ps_CFHT_mat_fn),axis=0)
	ps_CFHT_all = sum(WLanalysis.readFits(ps_CFHT_all_mat_fn),axis=0)*sum(fsky)/sum(fsky_all)
	
	## Simulation
	#pk_mat_all_fn = '/Users/jia/CFHTLenS/emulator/peaks_sum/pk_mat_sigma%02d_%02dbins.fit'%(sigmaG*10,bins)
	#pk_mat = WLanalysis.readFits(pk_mat_all_fn).reshape(91,1000,-1)
	#ps_CFHT_all = mean(pk_mat,axis=1)[48]*sum(fsky)/sum(fsky_all)
	#ps_CFHT = ps_avg[48]
	
	lw=3
	
	f=figure(figsize=(10,8))
	ax=f.add_subplot(gs[0])
	ax2=f.add_subplot(gs[1],sharex=ax)
	
	ax.errorbar(ell_arr, ps_CFHT, fidu_std, color='g', linewidth=lw)
	ax.plot(ell_arr, ps_CFHT, 'g-', label=r'$\rm{pass\, fields}$',linewidth=lw)
	
	ax.plot(ell_arr, ps_CFHT_all, 'm--', label=r'$\rm{all\, fields}$',linewidth=lw)
	
	ax2.errorbar(ell_arr, zeros(len(ell_arr)),fidu_std/fidu_avg, color='g', linewidth=lw)
	ax2.plot(ell_arr, zeros(len(ell_arr)), 'g-', label=r'$\rm{pass\, fields}$',linewidth=lw)
	ax2.plot(ell_arr, ps_CFHT_all/ps_CFHT-1,'m--', label=r'$\rm{all\, fields}$',linewidth=lw)
	
	ax2.set_ylim(-0.15, 0.15)
	ax2.set_yticks(linspace(-0.1,0.1,3))
	
	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':20},loc=2)
	leg.get_frame().set_visible(False)

	plt.subplots_adjust(hspace=0.0,left=0.15)
	plt.setp(ax.get_xticklabels(), visible=False) 
	ax.set_xlim(ell_arr[0],ell_arr[-1])
	ax.tick_params(labelsize=16)
	ax2.tick_params(labelsize=16)
	ax.set_ylabel(r'$\rm{peak\, counts\, N(\kappa)}$',fontsize=20)
	ax2.set_ylabel(r'$\rm{{\Delta}N/N}$',fontsize=20)
	ax2.set_xlabel('$\kappa$',fontsize=20)
	#ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$',fontsize=20)
	#ax.set_xscale('log')
	#ax2.set_xscale('log')
	#ax2.set_xlabel(r'$\ell$')
	#ax.set_yscale('log')
	#ax2.set_ylabel(r'$\rm{{\Delta}P/P}$',fontsize=20)
	#ax.set_ylim(1e-5, 1e-2)
	#savefig(plot_dir+'official/sample_interpolation_ps_sigmaG%02d.jpg'%(sigmaG*10))
	
	#ax.set_title(r'$\rm{Simulation}$',fontsize=24)
	#savefig('/Users/jia/weaklensing/CFHTLenS/plot/official/good_bad_peaks_fidu.pdf')
	
	ax.set_title(r'$\rm{CFHTLenS}$',fontsize=24)
	savefig('/Users/jia/weaklensing/CFHTLenS/plot/official/good_bad_peaks_CFHT.pdf')
	close()

if contour_including_w:
	w_arr = linspace(0,-3,101)
	
	cube_pk =WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/chisq_cube_CFHT_pk_sigmaG1018.fit').reshape(-1,l,ll)
	cube_ps = WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/GoodOnly/chisq_cube_CFHT_psellcut0_26.fit').reshape(-1,l,ll)
	
	#Ppk2 = lambda axis: cube2P(WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/chisq_cube_CFHT_pk_sigmaG1018.fit').reshape(-1,l,ll), axis=axis)
	
		
	#, Pc]#, Ppk]#
	labels = [r'$\rm{power\, spectrum}$', r'$\rm{peaks\, (1.0 + 1.8\,arcmin)}$', r'$\rm{power\, spectrum + peaks}$']
	
	
	
	lines=[]

	param_arr = [om_arr, w_arr, si8_arr]
	xylabels=((0,1),(0,2),(1,2))#need to marginalize over si8, w, om
	axis_arr=[2,0,1]#cube axis are[w, om, si8]
	f = figure(figsize=(10,8))
	for k in (1,2,3):
		j1,j2=xylabels[k-1]
		if k == 1:
			ax=f.add_subplot(2,2,1)
		else:
			ax=f.add_subplot(2,2,k+1)
		Pps = cube2P(cube_ps, axis=axis_arr[k-1])
		Ppk = cube2P(cube_pk, axis=axis_arr[k-1])
		
		X, Y = np.meshgrid(param_arr[j1], param_arr[j2])
		print k, 'Pps.shape, X.shape', Pps.shape, X.shape
		if Pps.shape[0] == X.shape[0]:
			Pps = Pps.T
			Ppk = Ppk.T
		P_arr=[Pps, Ppk, Pps*Ppk]
		for i in range(len(P_arr)):
			print i
			P=P_arr[i]
			V=findlevel(P)
			CS = ax.contour(X, Y, P.T, levels=[V[0],], origin='lower', extent=(om_arr[0], om_arr[-1], si8_arr[0], si8_arr[-1]), colors=colors[-3+i], linewidths=lws[i*2], linestyles=lss2[i])
			if k == 2:
				lines.append(CS.collections[0])
		ax.tick_params(labelsize=14)
		if k>0:
			ax.set_xlabel(cosmo_labels[j1],fontsize=16)
		if k<3:
			ax.set_ylabel(cosmo_labels[j2],fontsize=16)
		matplotlib.pyplot.locator_params(nbins=5)

	ax=f.add_subplot(2,2,3)
	leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':14},loc=0)
	leg.get_frame().set_visible(False)
	plt.subplots_adjust(hspace=0.0,wspace=0)
	plt.setp(subplot(221).get_xticklabels(), visible=False)
	plt.setp(subplot(224).get_yticklabels(), visible=False)
	savefig('/Users/jia/weaklensing/CFHTLenS/plot/official/contour3_peaks_powspec.pdf')#_wcut%.1f_%.1f_combined.pd
	close()
	
if compare_pk_contour_andrea:
	
	#w_arr_j = linspace(0,-3,101)[51]
	
	chisq_cube_andrea = np.load('/Users/jia/Dropbox/test_peak_powspec/Likelihood_cubes/andrea/likelihood_peaks--1.0.npy').reshape(100,100,100)
	
	chisq_cube_jia = WLanalysis.readFits('/Users/jia/Dropbox/test_peak_powspec/Likelihood_cubes/jia/BAD_chisq_cube_CFHT_pk_sigmaG10.fit').reshape(-1,100,102)[:51,:,:]
	
	om_arr_a = linspace(0.05,1.2,100)
	si8_arr_a = linspace(0.1,1.6,100)
	om_arr = linspace(0,1.2,l)
	si8_arr = linspace(0,1.6,ll)
	
	Pa = sum(exp(-chisq_cube_andrea/2),axis=1)
	Pa /= sum(Pa)
	Xa, Ya = np.meshgrid(om_arr_a, si8_arr_a)
	
	Pj = sum(exp(-chisq_cube_jia/2),axis=0)
	Pj /= sum(Pj)
	Xj, Yj = np.meshgrid(om_arr, si8_arr)
	
	f = figure(figsize=(8,8))
	labels = ['andrea','jia']
	ax=f.add_subplot(111)
	lines=[]
	
	Va=findlevel(Pa)
	CSa = ax.contour(Xa, Ya, Pa.T, levels=Va[:-1], origin='lower', extent=(om_arr_a[0], om_arr_a[-1], si8_arr_a[0], si8_arr_a[-1]), colors=colors[0], linewidths=lws2[0], linestyles=lss2[0])
	lines.append(CSa.collections[0])
	
	Vj=findlevel(Pj)
	CSj = ax.contour(Xj, Yj, Pj.T, levels=Vj[:-1], origin='lower', extent=(om_arr[0], om_arr[-1], si8_arr[0], si8_arr[-1]), colors=colors[1], linewidths=lws2[1], linestyles=lss2[1])
	lines.append(CSj.collections[0])
	
	leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':20},loc=0)
	ax.tick_params(labelsize=16)
	ax.set_xlabel(r'$\rm{\Omega_m}$',fontsize=20)
	ax.set_ylabel(r'$\rm{\sigma_8}$',fontsize=20)
	leg.get_frame().set_visible(False)
	#show()
	savefig(plot_dir+'contour_peaks_1arcmin_andrea.jpg')
	close()

if m_correction:
	sigmaG=1.0
	m_center = (0.057,-0.37)
	m_cov = matrix([[ 7.16613211e-05, -4.88602145e-04],
		 [ -4.88602145e-04, 3.53233868e-03]]) 
	rab = lambda:np.random.multivariate_normal(m_center,m_cov)
	def mfun (snr, r): 
		alpha, beta = rab()
		m = beta/log10(snr)*exp(-alpha*r*snr)
		return m
	
	################prepare neccesarry files
	#zarr = genfromtxt('/Users/jia/CFHTLenS/catalogue/raytrace_subfield1',usecols=[2,3,4])
	#zidx = where((amax(zarr,axis=1)<1.3)&(amin(zarr,axis=1)>0.2))
	#WLanalysis.writeFits(zidx[0],'/Users/jia/CFHTLenS/catalogue/raytrace_subfield.fit')
	#zidx=WLanalysis.readFits('/Users/jia/CFHTLenS/catalogue/raytrace_subfield.fit')
	
	#fullfield1 = genfromtxt('/Users/jia/CFHTLenS/catalogue/full_subfields/full_subfield1',usecols=[0,1,9,10,11,13,14,16,17])[zidx].T # see chopCFHT.py see column descriptions
	#WLanalysis.writeFits(fullfield1.T,'/Users/jia/CFHTLenS/catalogue/full_subfield_zcut0213.fit')
	################################
	
	y, x, e1, e2, w, r, snr, m, c2 = WLanalysis.readFits('/Users/jia/CFHTLenS/catalogue/full_subfield_zcut0213.fit').T
	
	k, s1, s2 = WLanalysis.readFits('/Users/jia/CFHTLenS/catalogue/emulator_subfield1_WL-only_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_4096xy_0999r.fit').T[[0,1,2]]
	mask_ps = WLanalysis.readFits('/Users/jia/CFHTLenS/catalogue/mask/BAD_CFHT_mask_ngal5_sigma10_subfield01.fits')
	mask_pk = WLanalysis.readFits('/Users/jia/CFHTLenS/catalogue/mask/CFHT_mask_ngal5_sigma10_subfield01.fits')
	def m_test (iseed=1):
		print iseed
		seed(iseed)
		mx = mfun (snr, r)		
		e1, e2 = (1+mx)*s1, (1+mx)*s2
		A, galn = WLanalysis.coords2grid(x, y, array([e1*w, e2*w, (1+mx)*w, (1+m)*w, k*w]))
		Me1, Me2, Mmxw, Mmw, Mk = A
		
		Me1_smooth_MC = WLanalysis.weighted_smooth(Me1, Mmxw, PPA=PPA512, sigmaG=sigmaG)
		Me2_smooth_MC = WLanalysis.weighted_smooth(Me2, Mmxw, PPA=PPA512, sigmaG=sigmaG)
		
		Me1_smooth = WLanalysis.weighted_smooth(Me1, Mmw, PPA=PPA512, sigmaG=sigmaG)
		Me2_smooth = WLanalysis.weighted_smooth(Me2, Mmw, PPA=PPA512, sigmaG=sigmaG)

		Mk_smooth = WLanalysis.weighted_smooth(Mk, Mmw, PPA=PPA512, sigmaG=sigmaG)
		
		kmap_MC = WLanalysis.KSvw(Me1_smooth_MC, Me2_smooth_MC)
		kmap = WLanalysis.KSvw(Me1_smooth, Me2_smooth)
		
		#plotimshow(Mk_smooth, 'mcorrection_Mk_smooth')
		#plotimshow(kmap_MC, 'mcorrection_kmapMC_smooth')
		#plotimshow(kmap, 'mcorrection_kmap_smooth')
		ps = 1/fsky[0]*WLanalysis.PowerSpectrum(kmap*mask_ps, sizedeg=12.0)[-1][11:]
		ps_MC = 1/fsky[0]*WLanalysis.PowerSpectrum(kmap_MC*mask_ps, sizedeg=12.0)[-1][11:]
		pk = WLanalysis.peaks_mask_hist(kmap*mask_pk, mask_pk, bins, kmin = kmin, kmax = kmax)
		pk_MC = WLanalysis.peaks_mask_hist(kmap_MC*mask_pk, mask_pk, bins, kmin = kmin, kmax = kmax)
		return kmap, kmap_MC, ps, ps_MC, pk, pk_MC
	#a=m_test()
	#Mtest_arr = array(map(m_test,range(100)))
	#ps_arr = array([Mtest_arr[i][2] for i in range(100)])
	#psMC_arr = array([Mtest_arr[i][3] for i in range(100)])
	#pk_arr = array([Mtest_arr[i][4] for i in range(100)])
	#pkMC_arr = array([Mtest_arr[i][5] for i in range(100)])
	
	#WLanalysis.writeFits(ps_arr,   emu_dir+'mtest_ps_arr.fit')
	#WLanalysis.writeFits(psMC_arr, emu_dir+'mtest_psMC_arr.fit')
	#WLanalysis.writeFits(pk_arr,   emu_dir+'mtest_pk_arr.fit')
	#WLanalysis.writeFits(pkMC_arr, emu_dir+'mtest_pkMC_arr.fit')
	ps_arr = WLanalysis.readFits(emu_dir+'mtest_ps_arr.fit')
	psMC_arr=WLanalysis.readFits(emu_dir+'mtest_psMC_arr.fit')
	pk_arr = WLanalysis.readFits(emu_dir+'mtest_pk_arr.fit')
	pkMC_arr=WLanalysis.readFits(emu_dir+'mtest_pkMC_arr.fit')
	
	lw=3
	if ps_replaced_with_pk:
		ps_const = mean(pk_arr,axis=0)
		ps_MC = mean(pkMC_arr, axis=0)
		ps_const_std = std(pk_arr,axis=0)
		ps_MC_std = std(pkMC_arr, axis=0)
	else:
		ps_const = mean(ps_arr,axis=0)
		ps_MC = mean(psMC_arr,axis=0)
		ps_const_std = std(ps_arr, axis=0)
		ps_MC_std = std(psMC_arr,axis=0)
		
	f=figure(figsize=(10,8))
	ax=f.add_subplot(gs[0])
	ax2=f.add_subplot(gs[1],sharex=ax)

	ax.errorbar(ell_arr, ps_MC, ps_MC_std, color='m', linewidth=lw)
	ax.plot(ell_arr, ps_MC, 'm-', label=r'$\rm{true}$',linewidth=lw)
	ax.errorbar(ell_arr, ps_const, ps_const_std, color='m', linewidth=lw)
	ax.plot(ell_arr, ps_const, 'g--', label=r'$\rm{fit}$',linewidth=lw)
		
	ax2.errorbar(ell_arr, zeros(len(ell_arr)),ps_MC_std/ps_MC, color='m',linewidth=lw)
	ax2.errorbar(ell_arr, ps_const/ps_MC-1, ps_const_std/ps_const, color='g', linewidth=lw)

	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=0)
	leg.get_frame().set_visible(False)

	plt.subplots_adjust(hspace=0.0,left=0.15)
	plt.setp(ax.get_xticklabels(), visible=False) 
	ax.set_xlim(ell_arr[0],ell_arr[-1])
	ax.tick_params(labelsize=16)

	if ps_replaced_with_pk:
		fn='/Users/jia/weaklensing/CFHTLenS/plot/official/m_correction_peaks.pdf'
		ax.set_ylabel(r'$\rm{peak\, counts\, N(\kappa)}$',fontsize=20)
		ax2.set_ylabel(r'$\rm{{\Delta}N/N}$',fontsize=20)
		ax2.set_xlabel('$\kappa$',fontsize=20)
		ax2.set_ylim(-0.005, 0.005)
		#ax2.set_yticks(linspace(-0.2,0.2,3))
		#ax.set_ylim(-1,2700)
	else:
		#ax2.set_ylim(-0.1, 0.1)
		#ax2.set_yticks(linspace(-0.05,0.05,3))
		fn='/Users/jia/weaklensing/CFHTLenS/plot/official/m_correction_powspec.pdf'
		ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$',fontsize=20)
		ax.set_xscale('log')
		ax2.set_xscale('log')
		ax2.set_xlabel(r'$\ell$')
		ax.set_yscale('log')
		ax2.set_ylabel(r'$\rm{{\Delta}P/P}$',fontsize=20)
		#ax.set_ylim(1e-5, 1e-2)

	savefig(fn)
	close()
	
	
if correlation_matrix:
	std_fidu = std(ps_fidu,axis=0)
	X, Y = np.meshgrid(std_fidu, std_fidu)
	corr_mat = cov_mat / (X*Y)
	plotimshow(corr_mat, 'corr_matrix_combined_01',vmin=0,vmax=0.1)
	
	
if ps_from_2pcf:
	from scipy.integrate import quad
	r, xi_plus = genfromtxt('/Users/jia/CFHTLenS/2PCF/CFHT2pcf').T[:2]#,:15] # r in arcmin
	r = radians(r/60.0)#convert from degrees to radians
	xi_interp = lambda x: interpolate.interp1d(r,xi_plus)(x)
	bessel = lambda theta, k, r: exp(-1j*k*r*cos(theta))
	#r xi(r)  d theta  exp(i k r cos(theta))
	def integrand (ir, k):
		return float(ir*xi_interp(ir)*quad(bessel, 0, 2*pi, (k, ir))[0])
	
	ps_2pcf = zeros(len(ell_arr))
	i = 0
	for k in ell_arr:
		print i
		ps_2pcf[i] = quad(integrand,r[0],r[-1],(k,))[0]
		i+=1
	f=figure()
	ax=f.add_subplot(111)

	#ax.plot(ell_arr, ps_2pcf*ell_arr*(ell_arr+1)/2/pi,'ro',label='Kilbinger+ 2013')
	ax.plot(ell_arr, ps_2pcf*ell_arr*(ell_arr+1),'ro',label='Kilbinger+ 2013')
	ax.plot(ell_arr, fidu_avg,'b-',label='Power Spectrum')
	ax.set_ylim(5e-6, 5e-2)
	ax.set_xlim(ell_arr[0],ell_arr[-1])
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set_xlabel('ell')
	ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$')
	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
	leg.get_frame().set_visible(False)
	savefig(plot_dir+'ps_2pcf_2pi.jpg')
	close()

if std_converge:
	randstd = lambda n: std(ps_fidu[randint(0,999,n)],axis=0)
	n_arr = arange(50,1000,20)
	std_arr = zeros(shape=(len(n_arr),2))
	i = 0
	for n in n_arr:
		all_std = array(map(randstd, [n,]*100))
		std_arr[i,0] = mean(mean(all_std,axis=0))
		std_arr[i,1] = mean(std(all_std, axis=0))
		i+=1
	f=figure()
	ax=f.add_subplot(111)
	#ax.plot(n_arr, std_arr[:,0])
	ax.errorbar(n_arr, std_arr[:,0], std_arr[:,1])
	savefig(plot_dir+'errorbar_ps_test_20steps_err.jpg')
	close()

if theory_powspec_err:	
	ell_edges, psd1D = WLanalysis.azimuthalAverage(ones(shape=(512,512)))
	ell_edges *= 360./sqrt(12)
	del_ell = ell_edges[1:]-ell_edges[:-1]
	del_ell = del_ell[11:]
	
	N_sim = array([   12,     8,    16,    20,    24,    28,    24,    44,    48,
          76,    76,   116,   124,   164,   200,   252,   296,   408,
         472,   608,   756,   936,  1184,  1460,  1840,  2284,  2860,
        3588,  4444,  5580,  6928,  8680, 10816, 13500, 16864, 21056,
       26276, 32812, 40980]).astype(float)
	
	sf1_ps_mat = WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/GoodOnly/powspec_sum/SIM_powspec_sigma05_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_subfield01.fit')[:,11:]/7.6645622253410002
	
	sf1_noiseless_ps_mat = WLanalysis.readFits('/Users/jia/CFHTLenS/emulator/ps_mat_kappa_noiselss_sf1.fit')[:,11:]
	delpp_noiseless = std(sf1_noiseless_ps_mat,axis=0)/mean(sf1_noiseless_ps_mat,axis=0)
	
	delpp_sf1 = std(sf1_ps_mat,axis=0)/mean(sf1_ps_mat,axis=0)
	plankerr = 1/sqrt(fsky[0]*12.0/41253.0*(2*ell_arr+1)*del_ell)
	#plankerr = 1/sqrt(fsky_all[0]*12.0/41253.0*(2*ell_arr+1)*del_ell)
	
	f=figure(figsize=(8,6))
	ax=f.add_subplot(111)
	ax.plot(ell_arr,delpp_sf1,'-k',linewidth=2,label=r'$\rm{Simulation}$')
	ax.plot(ell_arr,delpp_noiseless,'-g',linewidth=2,label=r'$\rm{Sim_noiseless}$')
	ax.plot(ell_arr,1/sqrt(N_sim),'--m',linewidth=2,label=r'$1/\sqrt{N}$')
	ax.plot(ell_arr,1/sqrt(N_sim/2.0),'-.g',linewidth=2,label=r'$2/\sqrt{N}$')
	
	ax.plot(ell_arr,plankerr,'-r',linewidth=1,label=r'$PlanckXVIII$')
	
	ax.set_xscale('log')
	ax.set_xlim(ell_arr[0],ell_arr[-5])
	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=0)
	leg.get_frame().set_visible(False)
	ax.set_ylabel(r'$\rm{{\Delta}P/P}$',fontsize=16)
	ax.set_xlabel(r'$\ell$',fontsize=16)
	savefig(plot_dir+'official/variance_sim_theory.jpg')
	close()