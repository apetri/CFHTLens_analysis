# this would be a messy code, plot out whatever needs to be plotted

import WLanalysis
import os
import numpy as np
from scipy import *
import scipy.ndimage as snd
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.gridspec as gridspec 
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
import triangle
########## knobs ################
plot_sample_KS_maps = 0 # pass - make sure the maps look reasonable
peaks_4cosmo_CFHT = 0 # pass - plot out the peak counts for simulation and CFHT
peaks_varians_13fields = 0 # pass - varians among 13 fields
CFHT_cosmo_fit = 0 # pass - fit to CFHT and countour (using simulation for contour)
shear_asymmetry = 0 # test Jan's ray tracing if e1, e2 are asymmetric
test_gaussianity = 0
model_vs_CFHT = 0
emcee_MCMC = 0
mask_collect = 0
mask_ngal = 0 #demonstrate how smoothed ngal can be used as mask
test_bins = 0 # official plot
test_sigmaG = 0 # official plot
powspec_vs_nicaea = 1
########## knobs end ############

dp = array([0.03, 0.2, 0.05])
fidu_params = array([0.26, -1.0, 0.8])
zg_arr = ('pz','rz1','rz2') # rz1 for model building 
bins_arr = arange(10, 110, 15)
sigmaG_arr = (0.5, 1, 1.8, 3.5, 5.3, 8.9)
i_arr=arange(1,14)
R_arr=arange(1,1001)
Rtol = len(R_arr)

plot_dir = '/Users/jia/weaklensing/CFHTLenS/plot/'
KSsim_dir = '/Users/jia/weaklensing/CFHTLenS/KSsim/'
CFHT_dir = '/Users/jia/weaklensing/CFHTLenS/CFHTKS/'
fidu='mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800'
hi_w='mQ3-512b240_Om0.260_Ol0.740_w-0.800_ns0.960_si0.800'
hi_s='mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.850'
hi_m='mQ3-512b240_Om0.290_Ol0.710_w-1.000_ns0.960_si0.800'

SIMfn= lambda i, cosmo, R: KSsim_dir+'test_ells_asymm/raytrace_subfield%i_WL-only_%s_4096xy_%04dr.fit'%(i, cosmo, R) 

KSsim_fn = lambda i, cosmo, R, sigmaG, zg: KSsim_dir+'%s/SIM_KS_sigma%02d_subfield%i_%s_%s_%04dr.fit'%(cosmo, sigmaG*10, i, zg, cosmo,R)

peaks_fn = lambda i, cosmo, Rtol, sigmaG, zg, bins: KSsim_dir+'peaks/SIM_peaks_sigma%02d_subfield%i_%s_%s_%04dR_%03dbins.fit'%(sigmaG*10, i, zg, cosmo, Rtol, bins)

powspec_fn = lambda i, cosmo, Rtol, sigmaG, zg: KSsim_dir+'powspec/SIM_powspec_sigma%02d_subfield%i_%s_%s_%04dR.fit'%(sigmaG*10, i, zg, cosmo, Rtol)


CFHT_sum_fn = lambda sigmaG, bins: KSsim_dir+'peaks_sum13fields/CFHT_peaks_sigma%02d_%03dbins.fits'%(sigmaG*10, bins)

peaks_sum_fn = lambda cosmo, Rtol, sigmaG, zg, bins: KSsim_dir+'peaks_sum13fields/SIM_peaks_sigma%02d_%s_%s_%04dR_%03dbins.fit'%(sigmaG*10, zg, cosmo, Rtol, bins)

cosmo_arr=(fidu,hi_m,hi_w,hi_s)

cosmolabels=('$Fiducial$','$High-\Omega_m$', '$High-w$', '$High-\sigma_8$')

labels=('$\Omega_m$','$w$','$\sigma_8$')

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


def plotEllipse(pos, P, edge, ilabel):
	U, s, Vh = svd(P)
	orient = math.atan2(U[1,0],U[0,0])*180/pi
	#print pos, sqrt(s[0]), sqrt(s[1]),orient
	ellipsePlot = Ellipse(xy=pos, width=2.0*math.sqrt(s[0]), height=2.0*math.sqrt(s[1]), angle=orient,linewidth=2, fill = False, edgecolor=edge, label=ilabel)
	ax = gca()
	#ax.add_patch(ellipsePlot)
	ax.add_artist(ellipsePlot)
	return ellipsePlot
############################################

if plot_sample_KS_maps:

	sigmaG = 3.5
	for i in i_arr:
		
		fn = KSsim_fn (i, fidu, 1000, sigmaG, 'rz1')
		img = WLanalysis.readFits(fn)
		plotimshow (img, 'fidu_SF%i_%i'%(i,sigmaG*10))
		
		fn = KSsim_fn (i, hi_s, 1000, sigmaG, 'rz1')
		img = WLanalysis.readFits(fn)
		plotimshow (img, 'hi_Si8_SF%i_%i'%(i,sigmaG*10))

if peaks_4cosmo_CFHT:
	
	#peaks_fn = lambda i, cosmo, Rtol, sigmaG, zg, bins
	x = linspace(-0.04, 0.12, 26)
	x = x[:-1]+0.5*(x[1]-x[0])
	
	peaks_mat=zeros(shape=(4, 1000, 25))
	CFHT_peak=zeros(25)
	for j in arange(1,14):
		print 'adding up subfield,',j
		i=0
		CFHT_peak+=WLanalysis.readFits (CFHT_dir+'CFHT_peaks_sigma35_subfield%02d_025bins.fits'%(j))
		for cosmo in cosmo_arr:
			peaks_mat[i] += WLanalysis.readFits(peaks_fn(j,cosmo,1000,3.5,'rz1',25))
			i+=1

	peaks = average(peaks_mat,axis=1)
	stdp = std(peaks_mat,axis=1)

	seed(11)
	colors=rand(8,3)
	gs = gridspec.GridSpec(2,1,height_ratios=[3,1]) 
	
	f=figure(figsize=(8,8))
	ax=f.add_subplot(gs[0])
	ax2=f.add_subplot(gs[1],sharex=ax)

	for i in range(4):
		ax.plot(x, peaks[i], label=cosmolabels[i],color=colors[i],linewidth=1.5)
		ax2.plot(x, peaks[i]/peaks[0]-1,color=colors[i],linewidth=1.5)
		
	ax.plot(x,CFHT_peak,color=colors[-2],label='$CFHT$')
	ax2.plot(x,CFHT_peak/peaks[0]-1,color=colors[-2],linewidth=1.5)

	ax.plot(x,N_model,color=colors[-1],label='$N_{model}$')
	ax2.plot(x,N_model/peaks[0]-1,color=colors[-1],linewidth=1.5)
	
	leg=ax.legend(ncol=1, labelspacing=.2, prop={'size':14})
	leg.get_frame().set_visible(False)

	ax.errorbar(x, peaks[0], yerr = stdp[0], fmt=None, ecolor=colors[0])

	#ax.set_yscale('log') 
	ax2.set_xlabel(r'$\kappa$',fontsize=14)
	plt.setp(ax.get_xticklabels(), visible=False) 
	plt.subplots_adjust(hspace=0.0)
	ax.set_ylim(0,600)
	ax.set_title('Peak Counts CFHT vs SIM, 1000r, 13 subfields')
	ax2.set_xlim(-0.02, 0.07) 
	ax2.set_ylim(-0.5, 0.5) 
	ax2.set_yticks(np.linspace(-0.45,0.45,4)) 
	ax.set_ylabel('N (peak)', fontsize=14)
	ax2.set_ylabel('frac diff from fiducial', fontsize=14)
	#savefig(plot_dir+'Peaks_sigma35_13fields_1000r_log.jpg')
	savefig(plot_dir+'Peaks_sigma35_13fields_1000r.jpg')
	close()   
	
if peaks_varians_13fields:
	x = linspace(-0.04, 0.12, 26)
	x = x[:-1]+0.5*(x[1]-x[0])
	CFHT_peak=zeros(shape=(13, 25))
	f=figure(figsize=(8,6))
	ax=f.add_subplot(111)
	seed(16)
	colors=rand(20,3)
	for j in arange(1,14):
		print 'CFHT subfield,',j
		iCFHT_peak=WLanalysis.readFits (CFHT_dir+'CFHT_peaks_sigma35_subfield%02d_025bins.fits'%(j))
		CFHT_peak [j-1] = iCFHT_peak
		ax.plot(x, iCFHT_peak/float(sum(iCFHT_peak)),color=colors[j],label='%i (%i)'%(j,sum(iCFHT_peak)))
		#ax.plot(x, iCFHT_peak,color=colors[j],label='%i (%i)'%(j,sum(iCFHT_peak)))
	leg=ax.legend(ncol=2, labelspacing=.2, prop={'size':14},title='subfield')
	leg.get_frame().set_visible(False)
	ax.set_xlabel(r'$\kappa$',fontsize=14)
	ax.set_ylabel('N (peak)', fontsize=14)
	ax.set_xlim(-0.02, 0.07)
	title('CFHT 13 fields, peak counts (Normed)')
	savefig(plot_dir+'Peaks_CFHT_13fields_normed.jpg')
	close()

if CFHT_cosmo_fit:
	
	fitpz=genfromtxt(KSsim_dir+'fit/fit_rz2_config_13subfields_1000R_021bins')[:,1:]
	fitrz2=genfromtxt(KSsim_dir+'fit/fit_pz_config_13subfields_1000R_021bins')[:,1:]
	fitCFHT=genfromtxt(KSsim_dir+'fit/fit_CFHT_13subfields_1000R_021bins')[1:]
	

	#### ellipse
	centerpz=average(fitpz,axis=0)
	centerrz2=average(fitrz2,axis=0)

	#xylabels=((0,2),(1,2),(0,1))
	xylabels=((0,1),(0,2),(1,2))
	f=figure(figsize=(8,6))
	for k in (1,2,3):
		if k == 1:
			subplot(2,2,1)
		else:
			subplot(2,2,k+1)
		i,j=xylabels[k-1]
		Ppz = cov(fitpz.T[[i,j]])
		Prz2 = cov(fitrz2.T[[i,j]])
		
		plotEllipse(centerpz[[i,j]],Ppz,'b','peak z')
		plotEllipse(centerrz2[[i,j]],Prz2,'r','random z')
		plot(centerpz[i],centerpz[j],'bo')
		plot(centerrz2[i],centerrz2[j],'ro')
		scatter(fitrz2.T[i],fitrz2.T[j]) # added 5/7/2014
		
		xlim(amin(fitrz2.T[i]),amax(fitrz2.T[i]))
		ylim(amin(fitrz2.T[j]),amax(fitrz2.T[j]))
		
		legend()
		xlabel(labels[i],fontsize=16,labelpad=15)
		ylabel(labels[j],fontsize=16)
		plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.97, wspace=0.3,hspace=0.3)
	
	#ax=f.add_subplot(223)
	#ax.text(0.8,2.0,'peak z',color='b',fontsize=14)
	#ax.text(0.8,1.5,'random z',color='r',fontsize=14)
	
	#plt.subplots_adjust(left=0.18,bottom=0.05,right=0.95,top=0.97,wspace=0.15,hspace=0.25)
	#plt.subplots_adjust(left=0.07,bottom=0.15,right=0.97,top=0.91,wspace=0.27,hspace=0.2)
	#show()
	savefig(plot_dir+'SIM_rand_peak_redshift_points.jpg')
	close()
	########## CFHT contour
	#f=figure(figsize=(8,6))
	#for k in (1,2,3):
		#subplot(2,2,k+1)
		#i,j=xylabels[k-1]
		#Prz2 = cov(fitrz2.T[[i,j]])
		
		#plotEllipse(fitCFHT[[i,j]],Prz2,'r','random z')
		#plot(fitCFHT[i],fitCFHT[j],'ro')
		
		#xlim(fitCFHT[i]-centerpz[i]*2.0,fitCFHT[i]+centerpz[i]*2.0)
		#ylim(fitCFHT[j]-centerpz[j]*2.0,fitCFHT[j]+centerpz[j]*2.0)
		
		#xlabel(labels[i],fontsize=16,labelpad=15)
		#ylabel(labels[j],fontsize=16)
		#plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.97, wspace=0.3,hspace=0.3)
	#savefig(plot_dir+'SIM_CFHTfit.jpg')
	#close()
	
if shear_asymmetry:
	#SIMfn= lambda i, cosmo, R
	f=figure(figsize=(12,12))
	j=1
	ellim = 0.05
	lwt = 2.0
	for cosmo in cosmo_arr:
		
		zcut_idx = WLanalysis.readFits(KSsim_dir+'test_ells_asymm/zcut0213_idx_subfield1.fit')
		#raytrace_cat=WLanalysis.readFits(SIMfn(1,cosmo,1000)).T
		raytrace_cat=WLanalysis.readFits(SIMfn(1,cosmo,1000))[zcut_idx].T
		e1, e2, e1_b, e2_b, e1_c, e2_c = raytrace_cat[[1,2,5,6,9,10]]
		
		k = (j-1)*3+1
		subplot(4,3,k)
		hist(e1,range=(-ellim,ellim),bins=20,normed=True, histtype='step',linewidth=lwt,label='e1 peak z')
		hist(e2,range=(-ellim,ellim),bins=20,normed=True,histtype='step',linewidth=lwt,label='e2 peak z')
		legend(fontsize=10,title=cosmolabels[j-1])
		
		subplot(4,3,k+1)
		hist(e1_b,range=(-ellim,ellim),bins=20,normed=True,histtype='step',linewidth=lwt,label='e1 rnd z')
		hist(e2_b,range=(-ellim,ellim),bins=20,normed=True,histtype='step',linewidth=lwt,label='e2 rnd z')
		legend(fontsize=10,title=cosmolabels[j-1])
		
		subplot(4,3,k+2)
		hist(e1_c,range=(-ellim,ellim),bins=20,normed=True,histtype='step',linewidth=lwt,label='e1 rnd z2')
		hist(e2_c,range=(-ellim,ellim),bins=20,normed=True,histtype='step',linewidth=lwt,label='e2 rnd z2')
		legend(fontsize=10,title=cosmolabels[j-1])
		#if j == 1:
			#leg = ax.legend(ncol=1, labelspacing=.2, prop={'size':10})
			#leg.get_frame().set_visible(False)
		
		j+=1
	
	savefig(plot_dir+'test_ells_asymm_zcut.jpg')
	close()

if test_gaussianity:
	######### chisquares
	chisqs = genfromtxt(KSsim_dir+'fit/fit_rz2_config_13subfields_1000R_021bins')[:,0]
	
	f=figure(figsize=(8,6))
	ax=f.add_subplot(111)
	ax.hist(chisqs,histtype='step',linewidth=2,bins=20,label='chisq histogram')
	title('chi-sq distribution of 1000 fits')
	savefig(plot_dir+'chisqs_test.jpg')
	close()
	
	x = linspace(-0.04, 0.12, 26)
	x = x[:-1]+0.5*(x[1]-x[0])
	
	######## bins
	peaks_mat=zeros(shape=(4, 1000, 25))
	f=figure(figsize=(8,6))
	i=0
	for cosmo in cosmo_arr:
		
		for j in arange(1,14):
			ipeaks = WLanalysis.readFits(peaks_fn(j,cosmo,1000,3.5,'rz1',25))
			peaks_mat[i] += ipeaks
		
		ax=f.add_subplot(2,2,i+1)
		for ibin in (11, 13, 15):
			hist(ipeaks[:,ibin], histtype='step',label='%i'%(ibin))
		leg = ax.legend(ncol=2, labelspacing=.2, prop={'size':10},title='bin #')
		leg.get_frame().set_visible(False)
		ax.set_title(cosmolabels[i])
		i+=1
	savefig(plot_dir+'individual_bins_gaussianity.jpg')
	close()
	
if model_vs_CFHT:
	### model 5/3/2014
	fitCFHT=genfromtxt(KSsim_dir+'fit/fit_CFHT_13subfields_1000R_021bins')[1:]
	cosmo_mat=(genfromtxt(KSsim_dir+'fit/cosmo_mat_13subfields_1000R_021bins')).reshape(4,1000,-1)
	dp = array([0.03, 0.2, 0.05])
	fidu_params = array([0.26, -1.0, 0.8])
	cov_mat = cov(cosmo_mat[0], rowvar = 0)#rowvar is the row contaning observations, aka 128R
	cov_inv = np.mat(cov_mat).I
	fidu_avg = mean(cosmo_mat[0], axis = 0)
	him_avg, hiw_avg, his_avg = mean(cosmo_mat[1:], axis = 1)
	dNdm = (him_avg - fidu_avg)/dp[0]
	dNdw =(hiw_avg - fidu_avg)/dp[1] 
	dNds = (his_avg - fidu_avg)/dp[2]
	X = np.mat([dNdm, dNdw, dNds])

	
	config_2_21 = array([[1.8, 25, 3, 17],
			  [3.5, 25, 5, 12]])
	
	CFHT_peak = zeros(21)
	for i in range(1,14):
		print 'adding CFHTobs', i
		CFHT_peak[:17-3] += WLanalysis.readFits(CFHT_dir+'CFHT_peaks_sigma18_subfield%02d_025bins.fits'%(i))[3:17]
		CFHT_peak[17-3:] += WLanalysis.readFits(CFHT_dir+'CFHT_peaks_sigma35_subfield%02d_025bins.fits'%(i))[5:12]
	
	
	Y = np.mat(CFHT_peak-fidu_avg)
	del_p = ((X*cov_inv*X.T).I)*(X*cov_inv*Y.T)
	N_model = array(fidu_avg+del_p.T*X).squeeze()
	del_N = Y-del_p.T*X
	y = linspace(-0.04, 0.12, 26)
	y = y[:-1]+0.5*(y[1]-y[0])
	y = concatenate((y[3:17],y[5:12]))
	
	x=range(21)
	
	peaks_mat = cosmo_mat
	peaks = average(peaks_mat,axis=1)
	stdp = std(peaks_mat,axis=1)

	seed(11)
	colors=rand(7,3)
	gs = gridspec.GridSpec(2,1,height_ratios=[3,1]) 
	
	f=figure(figsize=(8,8))
	ax=f.add_subplot(gs[0])
	ax2=f.add_subplot(gs[1],sharex=ax)

	for i in range(4):
		ax.plot(x, peaks[i], label=cosmolabels[i],color=colors[i],linewidth=1.5)
		ax2.plot(x, peaks[i]/peaks[0]-1,color=colors[i],linewidth=1.5)
		
	ax.plot(x,CFHT_peak,color=colors[-2],label='$CFHT$')
	ax2.plot(x,CFHT_peak/peaks[0]-1,color=colors[-2],linewidth=1.5)

	ax.plot(x,N_model,color=colors[-1],label='$N_{model}$')
	ax2.plot(x,N_model/peaks[0]-1,color=colors[-1],linewidth=1.5)
	
	leg=ax.legend(ncol=1, labelspacing=.2, prop={'size':14})
	leg.get_frame().set_visible(False)

	ax.errorbar(x, peaks[0], yerr = stdp[0], fmt=None, ecolor=colors[0])

	#ax.set_yscale('log') 
	ax2.set_xlabel(r'$\kappa$',fontsize=14)
	plt.setp(ax.get_xticklabels(), visible=False) 
	plt.subplots_adjust(hspace=0.0)
	#ax.set_ylim(0,600)
	ax.set_title('Peak Counts, 1000r, 1.8+3.5arcmin')
	#ax2.set_xlim(-0.02, 0.07) 
	ax2.set_ylim(-0.15, 0.15) 
	ax2.set_yticks(np.linspace(-0.1,0.1,3)) 
	#ax2.set_xticklabels(y[::len(y)/5.0]) 
	
	ax.set_ylabel('N (peak)', fontsize=14)
	ax2.set_ylabel('frac diff from fiducial', fontsize=14)
	#savefig(plot_dir+'Peaks_sigma35_13fields_1000r_log.jpg')
	savefig(plot_dir+'Peaks_sigma18and35_13fields_1000r.jpg')
	close()   

if emcee_MCMC:
	Rtol, bintol = 1000, 21
	fitCFHT=genfromtxt(KSsim_dir+'fit/fit_CFHT_13subfields_1000R_021bins')[1:]
	#fitSIM = genfromtxt('/Users/jia/weaklensing/CFHTLenS/KSsim/fit/fit_rz2_config_13subfields_1000R_021bins')

	cosmo_mat=(genfromtxt(KSsim_dir+'fit/cosmo_mat_13subfields_1000R_021bins')).reshape(4,1000,-1)
	dp = array([0.03, 0.2, 0.05])
	fidu_params = array([0.26, -1.0, 0.8])
	cov_mat = np.cov(cosmo_mat[0], rowvar = 0)#rowvar is the row contaning observations, aka 128R
	cov_inv = np.mat(cov_mat).I
	fidu_avg = mean(cosmo_mat[0], axis = 0)
	him_avg, hiw_avg, his_avg = mean(cosmo_mat[1:], axis = 1)
	dNdm = (him_avg - fidu_avg)/dp[0]
	dNdw =(hiw_avg - fidu_avg)/dp[1] 
	dNds = (his_avg - fidu_avg)/dp[2]
	X = np.mat([dNdm, dNdw, dNds])
	
	def cosmo_fit (obs):
		Y = np.mat(obs-fidu_avg)
		del_p = ((X*cov_inv*X.T).I)*(X*cov_inv*Y.T)
		m, w, s = np.squeeze(np.array(del_p.T))+fidu_params
		del_N = Y-del_p.T*X
		chisq = float((Rtol-bintol-2.0)/(Rtol-1.0)*del_N*cov_inv*del_N.T)
		return chisq, m, w, s
	fitSIM = array(map(cosmo_fit,cosmo_mat[0]))

	#obs = cosmo_mat[0,1]#fidu_avg
	obs = fidu_avg

	def lnprior(params):
		# flat prior
		m, w, s = params
		if -0.23 < m < 0.78 and -5.36 < w < 3.15 and -0.1 < s < 1.49:
			return 0.0
		else:
			return -np.inf

	def lnlike (params, obs):
		### Jia 5/12 change
		model = fidu_avg + mat(array(params)-fidu_params)*X
		del_N = np.mat(model - obs)
		ichisq = del_N*cov_inv*del_N.T #*float((Rtol-bintol-2.0)/(Rtol-1.0))
		Ln = -0.5*ichisq
		print params, Ln
		return float(Ln)
	
	def lnprob(params, obs):
		lp = lnprior(params)
		if not np.isfinite(lp):
			return -np.inf
		else:
			return lp + lnlike(params, obs)
	
	
	import scipy.optimize as op
	import emcee
	nll = lambda *args: -lnlike(*args)
	result = op.minimize(nll, fidu_params*1.1, args=(obs,), method='L-BFGS-B')
	
	ndim, nwalkers = 3, 100
	steps = 2000
	#pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
	pos = [fidu_params + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
	
	print 'run sampler'
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(obs,))
	
	print 'run mcmc'
	sampler.run_mcmc(pos, steps)
	samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
	
	import triangle
	
	#fig = triangle.corner(fitSIM[:,1:], labels=["$\Omega_m$", "$w$", "$\sigma_8$"],
			#truths=fidu_params, truths_color='#FF0000')

	#fig.savefig(plot_dir+"triangle_analytical_rz1.jpg")
	#close()

	fig = triangle.corner(samples, labels=["$\Omega_m$", "$w$", "$\sigma_8$"],
			truths=fidu_params)
	fig.savefig(plot_dir+"triangle_MCMC_%isteps_corrected.jpg"%(steps))
	close()

	####### chains plot out	
	#for i in range(3):
		#subplot(3,1,i+1)
		#plot(arange(samples.shape[0]),samples[:,i])
		#ylabel(labels[i])
		
	#savefig(plot_dir+'MC_steps_%isteps.jpg'%(steps))
	#close()
	
	### Fisher error 5/10/2014 ###
	F = zeros(shape=(3,3))
	for i in arange(3):
		for j in arange(3):
			F[i,j] += trace(cov_inv * X[i].T * X[j])
	
	###### plot error ellipse like mine
	
	fitrz2=genfromtxt(KSsim_dir+'fit/fit_pz_config_13subfields_1000R_021bins')[:,1:]
	#fitrz2=fitSIM[:,1:]
	centerrz2=array(cosmo_fit(obs)[1:])#average(fitrz2,axis=0)

	fitpz = samples## sneakily change fitpz to samples
	#centerpz=average(samples,axis=0)
	centerpz=np.percentile(samples,50,axis=0)
	

	xylabels=((0,1),(0,2),(1,2))
	f=figure(figsize=(8,6))
	for k in (1,2,3):
		if k == 1:
			subplot(2,2,1)
		else:
			subplot(2,2,k+1)
		i,j=xylabels[k-1]
		Ppz = cov(fitpz.T[[i,j]])
		Prz2 = cov(fitrz2.T[[i,j]])
		
		plotEllipse(centerpz[[i,j]],Ppz,'b','MCMC %i steps'%(steps*nwalkers))
		plotEllipse(centerrz2[[i,j]],Prz2,'r','analytical')
		# Fisher ellipse
		plotEllipse(centerrz2[[i,j]],(mat(F).I)[:,[i,j]][[i,j]],'m','Fisher')
		
		legend()
		
		plot(centerpz[i],centerpz[j],'bo')
		plot(centerrz2[i],centerrz2[j],'ro')
		
		xlim(amin(fitrz2.T[i]),amax(fitrz2.T[i]))
		ylim(amin(fitrz2.T[j]),amax(fitrz2.T[j]))
		
		
		xlabel(labels[i],fontsize=16,labelpad=15)
		ylabel(labels[j],fontsize=16)
		plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.97, wspace=0.3,hspace=0.3)
		

	ax=f.add_subplot(223)
	#ax.text(-0.18,0.3,'MCMC %.0e steps'%(100*nwalkers),color='g',fontsize=10)
	ax.text(-0.18,0.3,'Fisher',color='m',fontsize=10)
	ax.text(-0.18,0.15,'MCMC %.0e steps'%(steps*nwalkers),color='b',fontsize=10)
	ax.text(-0.18,0.0,'Analytical',color='r',fontsize=10)
	savefig(plot_dir+'MCMC_vs_analytial_%isteps_%i_corrected.jpg'%(steps,nwalkers))
	close()

if mask_collect:
	mask_bin_dir = '/Users/jia/weaklensing/mask/'
	for i in range(4):
		print i
		#imask_Wx = WLanalysis.readFits(mask_bin_dir+'Mask_W%i_fix05082014.fits'%(i+1))
		imask_Wx = genfromtxt(mask_bin_dir+'Mask_W%i_fix05082014.txt'%(i+1))
		imshow(imask_Wx[:,::-1],origin='lower')
		title('W'+str(i+1))
		savefig(plot_dir+'Mask_W%i_fix05082014_b.jpg'%(i+1))
		close()
		
if mask_ngal:
	size = 100
	a = 10*ones(shape=(size,size))

	y, x = np.indices((size,size),dtype=float)
	center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
	x -= center[0]
	y -= center[1]

	r = np.hypot(x, y)
	a[r<20] = 0
	
	asmooth = snd.filters.gaussian_filter(a,8)
	
	imshow(a,interpolation='nearest',vmin=0,vmax=10,cmap='binary')
	colorbar()
	savefig(plot_dir+'fakemask_original.jpg')
	close()
	
	imshow(asmooth,interpolation='nearest',vmin=0,vmax=10,cmap='binary')
	colorbar()
	CS=contour(asmooth, linewidth=4)
	plt.clabel(CS)
	savefig(plot_dir+'fakemask_cuts.jpg')
	close()

def cosmo_fit (obs, cosmo_mat):
	'''input obs and cosmo_mat (shape=(4,1000,-1))
	return 3 fitted params, and cov=F^-1
	'''
# get rid of the 0 bins
	idx = where (average(cosmo_mat[0],axis=0)>0.5)
	if len(idx[0]) < cosmo_mat.shape[-1]:
		print '----->',cosmo_mat.shape[-1]-len(idx[0]), '0-bins'
	cosmo_mat = cosmo_mat[:,:,idx].squeeze()
	obs=obs[idx].squeeze()

	cov_mat = np.cov(cosmo_mat[0], rowvar = 0)#rowvar is the row contaning observations, aka 128R
	
	cov_inv = np.mat(cov_mat).I
	fidu_avg, him_avg, hiw_avg, his_avg = mean(cosmo_mat, axis = 1)
	
	dNdm = (him_avg - fidu_avg)/dp[0]
	dNdw =(hiw_avg - fidu_avg)/dp[1] 
	dNds = (his_avg - fidu_avg)/dp[2]
	X = np.mat([dNdm, dNdw, dNds])
	
	Y = np.mat(obs-fidu_avg)
	del_p = ((X*cov_inv*X.T).I)*(X*cov_inv*Y.T)
	m, w, s = np.squeeze(np.array(del_p.T))+fidu_params
	del_N = Y-del_p.T*X
	#chisq = float((Rtol-bintol-2.0)/(Rtol-1.0)*del_N*cov_inv*del_N.T)
	
	F = zeros(shape=(3,3))
	for i in arange(3):
		for j in arange(3):
			F[i,j] += trace(cov_inv * X[i].T * X[j])
	return array([m, w, s]), mat(F).I
	
def cosmo_errors(sigmaG, bins, obs=None, zg='rz1'):
	cosmo_mat = zeros(shape=(4,1000,bins))
	k = 0
	for cosmo in cosmo_arr:
		fn = peaks_sum_fn(cosmo, Rtol, sigmaG, zg, bins)
		cosmo_mat[k] = WLanalysis.readFits(fn)
		k += 1
	# take only the bins with count >=5
	if obs == None:
		obs = average(cosmo_mat[0],axis = 0)
	centers, F_inv = cosmo_fit (obs, cosmo_mat)
	return centers, F_inv

def adjust_spines(ax,spines):
	for loc, spine in ax.spines.items():
		if loc in spines:
			spine.set_position(('outward',10)) # outward by 10 points
			spine.set_smart_bounds(True)
		else:
			spine.set_color('none') # don't draw spine

	# turn off ticks where there is no spine
	if 'left' in spines:
		ax.yaxis.set_ticks_position('left')
	else:
		# no yaxis ticks
		ax.yaxis.set_ticks([])

	if 'bottom' in spines:
		ax.xaxis.set_ticks_position('bottom')
	else:
		# no xaxis ticks
		ax.xaxis.set_ticks([])
if test_bins:
	seed(222)
	colors=rand(10,3)
	zg = 'rz1'
	xylabels=((0,1),(0,2),(1,2))
	def plot_bins (sigmaG):
		f = figure(figsize=(8,6))
		ic=0
		for bins in bins_arr:
			print
			fn_rz2 = peaks_sum_fn(fidu, Rtol, sigmaG, 'rz2', bins)
			#obs = average(WLanalysis.readFits(fn_rz2),axis=0)
			print bins, obs
			centers, F_inv = cosmo_errors(sigmaG,bins)
			
			for k in range(1,4):
				if k == 1:
					ax=subplot(2,2,1)
				else:
					ax=subplot(2,2,k+1)
				i, j = xylabels[k-1]
				plot(centers[i],centers[j],'o')
				triangle.error_ellipse(centers[[i,j]],F_inv[:,[i,j]][[i,j]], color=colors[ic], label=str(bins))
				
				xlabel(labels[i],fontsize=16,labelpad=15)
				if k <3:
					ylabel(labels[j],fontsize=16)
				plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.97, wspace=0.3,hspace=0.3)
				#if k == 1:
					#leg=legend(ncol=2, labelspacing=.1, prop={'size':12},loc=2)
					#leg.get_frame().set_visible(False)
				ax.xaxis.set_major_locator(MaxNLocator(6))
				ax.yaxis.set_major_locator(MaxNLocator(6))

			ic+=1
		for i in (1,3,4):
			ax=subplot(2,2,i)
			ax.set_xticks(ax.get_xticks()[1:-1])
			ax.set_yticks(ax.get_yticks()[1:-1])
		
		ihandles, ilabels = ax.get_legend_handles_labels()
		ax=subplot(222)
		
		leg=ax.legend(ihandles, ilabels, ncol=1, labelspacing=.3, title='Number of bins', prop={'size':14},loc=10)
		#leg.get_frame().set_visible(False)
		plt.setp(subplot(222).get_xticklabels(), visible=False)
		plt.setp(subplot(222).get_yticklabels(), visible=False)
		#ax.set_frame_on(False)
		adjust_spines(ax,[])
		leg.get_title().set_fontsize('14')
		#plt.setp(ax.legend.get_title(),fontsize=16)
		
		plt.subplots_adjust(wspace=0, hspace=0)
		plt.setp(subplot(221).get_xticklabels(), visible=False)
		plt.setp(subplot(224).get_yticklabels(), visible=False)
		
			
		savefig(plot_dir+"official/bins_sigmaG%s.pdf"%(sigmaG))
		savefig(plot_dir+"bins_sigmaG%s.jpg"%(sigmaG))
		close()
	map(plot_bins,sigmaG_arr)

if test_sigmaG:
	seed(222)
	colors=rand(10,3)
	zg = 'rz1'
	xylabels=((0,1),(0,2),(1,2))
	def plot_sigmaG (bins):
		f = figure(figsize=(8,6))
		ic=0
		for sigmaG in sigmaG_arr[::-1]:
			print bins, sigmaG
			centers, F_inv = cosmo_errors(sigmaG, bins)
			for k in range(1,4):
				if k == 1:
					ax=subplot(2,2,1)
				else:
					ax=subplot(2,2,k+1)
				i, j = xylabels[k-1]
				plot(centers[i],centers[j],'ko')
				triangle.error_ellipse(centers[[i,j]],F_inv[:,[i,j]][[i,j]], color=colors[ic], linewidth=2,label=str(sigmaG))
				
				xlabel(labels[i],fontsize=16,labelpad=15)
				if k <3:
					ylabel(labels[j],fontsize=16)
				plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.97, wspace=0.3,hspace=0.3)
				#if k == 1:
					#leg=legend(ncol=2, labelspacing=.1, prop={'size':12},loc=2)
					#leg.get_frame().set_visible(False)
				ax.xaxis.set_major_locator(MaxNLocator(6))
				ax.yaxis.set_major_locator(MaxNLocator(6))

			ic+=1
		for i in (1,3,4):
			ax=subplot(2,2,i)
			ax.set_xticks(ax.get_xticks()[1:-1])
			ax.set_yticks(ax.get_yticks()[1:-1])
		
		ihandles, ilabels = ax.get_legend_handles_labels()
		ax=subplot(222)
		
		leg=ax.legend(ihandles, ilabels, ncol=1, labelspacing=.3, title=r'$\theta_G$ (arcmin)', prop={'size':14},loc=10)
		#leg.get_frame().set_visible(False)
		plt.setp(subplot(222).get_xticklabels(), visible=False)
		plt.setp(subplot(222).get_yticklabels(), visible=False)
		#ax.set_frame_on(False)
		adjust_spines(ax,[])
		leg.get_title().set_fontsize('14')
		#plt.setp(ax.legend.get_title(),fontsize=16)
		
		plt.subplots_adjust(wspace=0, hspace=0)
		plt.setp(subplot(221).get_xticklabels(), visible=False)
		plt.setp(subplot(224).get_yticklabels(), visible=False)
		
			
		savefig(plot_dir+"official/sigmaG_bins%s.pdf"%(bins))
		savefig(plot_dir+"sigamG_bins%s.jpg"%(bins))
		close()
	map(plot_sigmaG,bins_arr)

if powspec_vs_nicaea:
	cosmo = fidu
	Nicaea_omegab001_ps = genfromtxt('/Users/jia/Documents/code/nicaea_2.4/Demo/P_kappa').T[:-1]
		
	Nicaea_omegab045_ps = genfromtxt('/Users/jia/Documents/code/nicaea_2.4/Demo/P_kappa_omegab045').T[:-1]
	#my ps
	powspec_CFHT_fn = lambda i, sigmaG: CFHT_dir+'CFHT_powspec_sigma%02d_subfield%02d.fits'%(sigmaG*10, i)
	ell_arr = logspace(log10(110.01746692),log10(25207.90813028),50)
	ix = Nicaea_omegab045_ps[0]
	loglog(ix, Nicaea_omegab001_ps[1],label='nicae sf1')
	loglog(ix, Nicaea_omegab045_ps[1],label='nicae omegab045')
	sigmaG = 0.5
	for i in (1,):#arange(1,14):
		print i
		ps = WLanalysis.readFits(powspec_CFHT_fn(i, sigmaG))
		loglog(ell_arr, ps[1], label='CFHT')
		

		SIMpw = WLanalysis.readFits(powspec_fn(i, fidu, 1000, sigmaG, 'rz1'))
		SIMpw_avg = average(SIMpw,axis=0)
		loglog(ell_arr, SIMpw_avg, label='SIM fidu')
		
		SIMpwm = WLanalysis.readFits(powspec_fn(i, hi_m, 1000, sigmaG, 'rz1'))
		SIMpwm_avg = average(SIMpw,axis=0)
		loglog(ell_arr, SIMpwm_avg, label='SIM hi_m')
	
	## get no noise maps
	zcut_idx = WLanalysis.readFits(KSsim_dir+'test_ells_asymm/zcut0213_idx_subfield1.fit')
	raytrace_cat=WLanalysis.readFits(SIMfn(1,cosmo,1000))[zcut_idx].T		
	k1, s1, s2= raytrace_cat[[0, 1, 2]]
	x, y, e1, e2, w, m = WLanalysis.readFits(KSsim_dir+'yxewm_subfield1_zcut0213.fit').T
	A, galn = WLanalysis.coords2grid(x,y, array([k1,s1,s2]))
	Mk, Ms1, Ms2 = A
	
	idx = where(galn>0)
	#Mk_norm = Mk.copy()
	#Mk_norm[idx] /= galn[idx]
	for sigmaG in sigmaG_arr[:3]:
		Mk_norm_smooth = WLanalysis.weighted_smooth(Mk, galn, sigmaG)
		ps_nonoise = WLanalysis.PowerSpectrum(Mk_norm_smooth)	
		loglog(ell_arr,ps_nonoise[1], label='SIM fidu %s'%(sigmaG))
	legend(fontsize=10,loc=0)
	show()
	