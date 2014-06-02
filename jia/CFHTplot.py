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
powspec_vs_nicaea = 0
noise_powspec_sigmaG = 0
powspec_one_map = 0 # see very noisy power spectrum, then wonder what went wrong with the maps
noise_redshift_relation = 0
cosmo_ps = 0
emu_checkps = 0
emu_checkpeaks = 0
emu_checkcheckps = 1
#emu_interpolate = 1
x = linspace(-0.04, 0.12, 26)
x = x[:-1]+0.5*(x[1]-x[0])#the kappa_arr
######## knobs end ############
#colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
lss = ('r-','g--','b-.','m:','c*','y,','ks','1','2','3','4','D')
markers = ('-','--','-.',':','*')

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
emu_dir = '/Users/jia/CFHTLenS/emulator/'
fidu='mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800'
hi_w='mQ3-512b240_Om0.260_Ol0.740_w-0.800_ns0.960_si0.800'
hi_s='mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.850'
hi_m='mQ3-512b240_Om0.290_Ol0.710_w-1.000_ns0.960_si0.800'

SIMfn= lambda i, cosmo, R: KSsim_dir+'test_ells_asymm/raytrace_subfield%i_WL-only_%s_4096xy_%04dr.fit'%(i, cosmo, R) 

KSsim_fn = lambda i, cosmo, R, sigmaG, zg: KSsim_dir+'%s/SIM_KS_sigma%02d_subfield%i_%s_%s_%04dr.fit'%(cosmo, sigmaG*10, i, zg, cosmo,R)

peaks_fn = lambda i, cosmo, Rtol, sigmaG, zg, bins: KSsim_dir+'peaks/SIM_peaks_sigma%02d_subfield%i_%s_%s_%04dR_%03dbins.fit'%(sigmaG*10, i, zg, cosmo, Rtol, bins)

powspec_fn = lambda i, cosmo, Rtol, sigmaG, zg: KSsim_dir+'powspec/SIM_powspec_sigma%02d_subfield%i_%s_%s_%04dR.fit'%(sigmaG*10, i, zg, cosmo, Rtol)

powspec_CFHT_fn = lambda i, sigmaG: CFHT_dir+'CFHT_powspec_sigma%02d_subfield%02d.fits'%(sigmaG*10, i)

CFHT_sum_fn = lambda sigmaG, bins: KSsim_dir+'peaks_sum13fields/CFHT_peaks_sigma%02d_%03dbins.fits'%(sigmaG*10, bins)

peaks_sum_fn = lambda cosmo, Rtol, sigmaG, zg, bins: KSsim_dir+'peaks_sum13fields/SIM_peaks_sigma%02d_%s_%s_%04dR_%03dbins.fit'%(sigmaG*10, zg, cosmo, Rtol, bins)

powspec_sum_fn = lambda cosmo, Rtol, sigmaG, zg: KSsim_dir+'powspec_sum13fields/SIM_powspec_sigma%02d_%s_%s_%04dR.fit'%(sigmaG*10, zg, cosmo, Rtol)

cosmo_arr=(fidu,hi_m,hi_w,hi_s)

cosmolabels=('Fiducial','High-$\Omega_m$', 'High-$w$', 'High-$\sigma_8$')

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
	idx = where (average(cosmo_mat[0],axis=0)>0.0)
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
	
def cosmo_errors(sigmaG, bins, obs=None, zg='rz1', powspec = False):
	if powspec:
		cosmo_mat = zeros(shape=(4,1000,50))
	else:
		cosmo_mat = zeros(shape=(4,1000,bins))
	k = 0
	for cosmo in cosmo_arr:
		if powspec:
			fn = powspec_sum_fn(cosmo, Rtol, sigmaG, zg)
		else: # peaks
			fn = peaks_sum_fn(cosmo, Rtol, sigmaG, zg, bins)
		cosmo_mat[k] = WLanalysis.readFits(fn)
		k += 1
	if powspec:
		cosmo_mat = cosmo_mat[:,:,11:]
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
	map(plot_sigmaG, bins_arr)

powspecMk_fn = lambda i, cosmo: KSsim_dir+'powspec_Mk/SIM_powspec_sigma05_subfield%i_rz1_%s_1000R.fit'%(i,cosmo)

gen_z_hist = 0
if gen_z_hist:
	def z_hist (i):
		'''return the histogramed redshift distribution for subfield i'''
		fn = '/Users/jia/weaklensing/CFHTLenS/catalogue/emulator_galpos_zcut0213/emulator_subfield%i_zcut0213.fit'%(i)
		z = WLanalysis.readFits(fn).T[2]
		zhist = histogram(z, range=(0.2,1.3), bins=16)
		print i, len(z)
		return zhist[0]

	zbinedges = linspace(0.2,1.3,17)
	all_hist = array(map(z_hist, range(1,14)))
	savetxt(KSsim_dir+'all_hist', all_hist)

ell_arr = logspace(log10(110.01746692),log10(25207.90813028),50)

def average_powspec_nonoise (cosmo):
	ps = zeros(shape=(1000,50))
	weights = (genfromtxt(KSsim_dir+'galn.txt').T[1]).astype(float)
	weights /= sum(weights)
	fn = KSsim_dir+'powspec_Mk_sum13fields/SIM_powspec_sigma05_rz1_%s_1000R.fit'%(cosmo)
	if os.path.isfile(fn):
		return WLanalysis.readFits(fn)
	else:
		for i in range(1,14):
			ips=weights[i-1]*WLanalysis.readFits(powspecMk_fn(i, cosmo))
			ps += ips
		WLanalysis.writeFits(ps,fn)
		return ps

def average_powspec_withnoise (cosmo, sigmaG, zg='rz1', CFHT=None):
	weights = (genfromtxt(KSsim_dir+'galn.txt').T[1]).astype(float)
	weights /= sum(weights)
	
	if CFHT:
		ps = zeros(50)
		fn = KSsim_dir+'powspec_sum13fields/CFHT_powspec_sigma%02d.fit'%(sigmaG*10)
	else:
		ps = zeros(shape=(1000,50))
		fn = KSsim_dir+'powspec_sum13fields/SIM_powspec_sigma%02d_%s_%s_%04dR.fit'%(sigmaG*10, zg, cosmo, Rtol)
	if os.path.isfile(fn):
		return WLanalysis.readFits(fn)
	else:
		for i in range(1,14):
			if CFHT:
				ips=weights[i-1]*WLanalysis.readFits(powspec_CFHT_fn(i, sigmaG))[-1]
			else:
				ips=weights[i-1]*WLanalysis.readFits(powspec_fn(i, cosmo, 1000, sigmaG, zg))
			ps += ips
		WLanalysis.writeFits(ps,fn)
		return ps
	
if powspec_vs_nicaea:
	sigmaG = 0.5
	cosmo = fidu
	
	Nicaea_ps = genfromtxt('/Users/jia/Documents/code/nicaea_2.4/Demo/P_kappa').T[:-1]
	
	loglog(Nicaea_ps[0], Nicaea_ps[1],'k.',label='Smith el al. 2003',linewidth=2)
	
	
	a=map(average_powspec_nonoise,cosmo_arr)
	for i in range(4):
		loglog(ell_arr, average(a[i], axis=0), lss[i], label=cosmolabels[i],linewidth=2)
	
	xlim(ell_arr[12],ell_arr[-1])
	ylim(8e-6,8e-5)
	leg=legend(ncol=1, labelspacing=0.3, prop={'size':14},loc=2)
	leg.get_frame().set_visible(False)
	savefig(plot_dir+'official/nicaea.pdf')
	savefig(plot_dir+'vs_nicaea.jpg')
	close()
	
	#for i in (1, 5, 8):#arange(1,14):
		#print i
		##ps = WLanalysis.readFits(powspec_CFHT_fn(i, sigmaG))
		##loglog(ell_arr, ps[1], label='CFHT')

		#SIM_ps = WLanalysis.readFits(powspecMk_fn(i, fidu))
		#SIM_ps_avg = average(SIM_ps,axis=0)
		#loglog(ell_arr, SIM_ps_avg, label='SIM fidu sf%i'%(i))

		#SIM_ps = WLanalysis.readFits(powspecMk_fn(i, hi_m))
		#SIM_ps_avg = average(SIM_ps,axis=0)
		#loglog(ell_arr, SIM_ps_avg, '--', label='SIM hi_m sf%i'%(i))
		
	#### no noise maps on the spot ###
	#zcut_idx = WLanalysis.readFits(KSsim_dir+'test_ells_asymm/zcut0213_idx_subfield1.fit')
	#raytrace_cat=WLanalysis.readFits(SIMfn(1,cosmo,1000))[zcut_idx].T		
	#k1, s1, s2= raytrace_cat[[0, 1, 2]]
	#x, y, e1, e2, w, m = WLanalysis.readFits(KSsim_dir+'yxewm_subfield1_zcut0213.fit').T
	#A, galn = WLanalysis.coords2grid(x,y, array([k1,s1,s2]))
	#Mk, Ms1, Ms2 = A
	
	#idx = where(galn>0)
	#for sigmaG in sigmaG_arr[:3]:
		#Mk_norm_smooth = WLanalysis.weighted_smooth(Mk, galn, sigmaG)
		#ps_nonoise = WLanalysis.PowerSpectrum(Mk_norm_smooth)	
		#loglog(ell_arr,ps_nonoise[1], label='SIM fidu %s'%(sigmaG))
	####################################
	
if noise_powspec_sigmaG:
	cosmo = fidu
	nonoise_fidu = average_powspec_nonoise(fidu)
	powspec_CFHT = average_powspec_withnoise (cosmo, 0.5, zg='rz1', CFHT=True)
	loglog(ell_arr, powspec_CFHT,label='CFHT')
	#Nicaea_ps = genfromtxt('/Users/jia/Documents/code/nicaea_2.4/Demo/P_kappa').T[:-1]
	#loglog(Nicaea_ps[0], Nicaea_ps[1],'k.',label='Smith el al. 2003',linewidth=2)
	loglog(ell_arr, average(nonoise_fidu,axis=0), label='fidu no noise')
	
	for sigmaG in sigmaG_arr:
		i=0
		for cosmo in cosmo_arr:
			psmat = average_powspec_withnoise (cosmo, sigmaG, zg='rz1', CFHT=None)
			ps = average(psmat, axis=0)
			loglog(ell_arr, ps, markers[i], label='%s %s'%(sigmaG,cosmolabels[i]))
			i+=1

	xlim(ell_arr[12],ell_arr[-1])
	#ylim(8e-6,8e-5)
	leg=legend(ncol=3, labelspacing=0.1, prop={'size':10},loc=2)
	leg.get_frame().set_visible(False)
	#savefig(plot_dir+'official/powspec_sigmaG.pdf')
	savefig(plot_dir+'powspec_sigmaG.pdf')
	close()

def eobs_fun (g1, g2, k, e1, e2):
	'''van wearbeke 2013 eq 5-6, get unbiased estimator for shear.
	Input:
	g1, g2: shear
	k: convergence
	e1, e2: galaxy intrinsic ellipticity
	Output:
	e_obs1, e_obs2
	'''
	g = (g1+1j*g2)/(1-k)
	eint = e1+1j*e2
	eobs = (g+eint)/(1-g*eint)
	return real(eobs), imag(eobs)

def eobs_analytical (z):
	sigmaz = 0.15 + 0.035*z
	erand = np.random.normal(0, sigmaz)
	return erand

if powspec_one_map:
	cosmo = fidu 
	
	zcut_idx = WLanalysis.readFits(KSsim_dir+'test_ells_asymm/zcut0213_idx_subfield1.fit')
	
	raytrace_cat=WLanalysis.readFits(SIMfn(1,cosmo,1000))[zcut_idx].T
	kappa, s1o, s2o= raytrace_cat[[0, 1, 2]]
	
	y, x, e1, e2, w, m = WLanalysis.readFits(KSsim_dir+'test_ells_asymm/yxewm_subfield1_zcut0213.fit').T
	
	z = WLanalysis.readFits('/Users/jia/weaklensing/CFHTLenS/catalogue/emulator_galpos_zcut0213/emulator_subfield1_zcut0213.fit').T[-1]
	
	znoise1 = eobs_analytical (z)
	znoise2 = eobs_analytical (z)
	znoise27 = np.random.normal(0, 0.29,size=len(z))
	
	s1 = s1o*(1+m)
	s2 = s2o*(1+m)
	
	eint1, eint2 = WLanalysis.rndrot(e1, e2, iseed=1000)
	eint1_45, eint2_45 = WLanalysis.rndrot(e1, e2, iseed=1000,deg=45.0)

	e1_reduce, e2_reduce = eobs_fun(s1, s2, kappa, eint1, eint2)
	e1_add, e2_add = s1+eint1, s2+eint2
	
	A, galn = WLanalysis.coords2grid(x, y, array([e1_reduce*w, e2_reduce*w, w*(1+m), s1o, s2o, kappa, e1_add*w, e2_add*w, s1+znoise1, s2+znoise2, kappa+znoise1]))
	#Mk, Ms1, Ms2 = A
	Me1, Me2, Mw, Ms1, Ms2, Mk, Me1add, Me2add, Ms1n, Ms2n, Mkn = A
	
	B, galn = WLanalysis.coords2grid(x, y, array([znoise1, eint1, eint2, eint1_45, eint2_45, znoise27]))
	Mn, Men1, Men2, Men1b, Men2b, Mn27 = B
	### pure analytical noise ######
	Mn_smooth =  WLanalysis.weighted_smooth(Mn, galn, sigmaG=0.5)
	Mn27_smooth =  WLanalysis.weighted_smooth(Mn27, galn, sigmaG=0.5)
	Men1_smooth =  WLanalysis.weighted_smooth(Men1, galn, sigmaG=0.5)
	Men2_smooth =  WLanalysis.weighted_smooth(Men2, galn, sigmaG=0.5)
	Men1b_smooth =  WLanalysis.weighted_smooth(Men1b, galn, sigmaG=0.5)
	Men2b_smooth =  WLanalysis.weighted_smooth(Men2b, galn, sigmaG=0.5)
	Nmap = WLanalysis.KSvw(Men1_smooth, Men2_smooth)
	Nmap45 = WLanalysis.KSvw(Men1b_smooth, Men2b_smooth)
	
	ps_nconv27 = WLanalysis.PowerSpectrum(Mn27_smooth)[-1]
	ps_nconv = WLanalysis.PowerSpectrum(Mn_smooth)[-1]
	ps_nrand = WLanalysis.PowerSpectrum(Nmap)[-1]
	ps_n45 = WLanalysis.PowerSpectrum(Nmap45)[-1]
	
	loglog(ell_arr, ps_nconv27, '.',label='Conv pure noise rms=0.29')
	loglog(ell_arr, ps_nconv, '.', label='Conv pure noise rms=0.15+0.035z')
	loglog(ell_arr, ps_nrand, '*', label='Shear pure noise(rand)')
	loglog(ell_arr, ps_n45, '*',label='Shear pure noise(45 deg)')
	

	
	###### shear ps w/ noise ######
	print 'shear ps w/ noise'
	Me1_smooth = WLanalysis.weighted_smooth(Me1, Mw, sigmaG=0.5)
	Me2_smooth = WLanalysis.weighted_smooth(Me2, Mw, sigmaG=0.5)
	emap = WLanalysis.KSvw(Me1_smooth, Me2_smooth)	
	ps_e = WLanalysis.PowerSpectrum(emap)
	loglog(ps_e[0],ps_e[1],label='Shear w/ noise')


	
	###### shear ps w/ linear noise ######
	#print 'shear ps w/ linear noise'
	#Me1a_smooth = WLanalysis.weighted_smooth(Me1add, Mw, sigmaG=0.5)
	#Me2a_smooth = WLanalysis.weighted_smooth(Me2add, Mw, sigmaG=0.5)
	#emapa = WLanalysis.KSvw(Me1a_smooth, Me2a_smooth)	
	#ps_ea = WLanalysis.PowerSpectrum(emapa)
	#loglog(ps_ea[0],ps_ea[1],label='Shear w/ linear noise')
	
	###### shear ps, no noise ######
	print 'shear ps w/out noise'
	Ms1_smooth = WLanalysis.weighted_smooth(Ms1, galn, sigmaG=0.25)
	Ms2_smooth = WLanalysis.weighted_smooth(Ms2, galn, sigmaG=0.25)
	smap = WLanalysis.KSvw(Ms1_smooth, Ms2_smooth)
	ps_s = WLanalysis.PowerSpectrum(smap)
	loglog(ps_s[0],ps_s[1],':',label='Shear w/out noise',linewidth=2)
	
	###### conv ps, no noise #######
	print 'conv ps w/out noise'
	kmap = WLanalysis.weighted_smooth(Mk, galn, sigmaG=0.25)
	ps_k = WLanalysis.PowerSpectrum(kmap)
	loglog(ps_k[0],ps_k[1],':',label='Conv w/out noise',linewidth=2)

	###### shear ps w/ noise ######
	loglog(ell_arr,ps_s[1]+ps_nconv27,label='Shear + rms0.29 noise')
	
	###### shear ps w/ Analytical noise ######
	print 'shear ps w/ analytical noise'
	Ms1n_smooth = WLanalysis.weighted_smooth(Ms1n, galn, sigmaG=0.5)
	Ms2n_smooth = WLanalysis.weighted_smooth(Ms2n, galn, sigmaG=0.5)
	smapn = WLanalysis.KSvw(Ms1n_smooth, Ms2n_smooth)	
	ps_sn = WLanalysis.PowerSpectrum(smapn)
	loglog(ps_sn[0],ps_sn[1],'--',label='Shear w/ rms=0.15+0.035z')
	
	###### conv ps w/ analytical noise  ######
	print 'conv ps w/ analytical noise'
	kmapn = WLanalysis.weighted_smooth(Mkn, galn, sigmaG=0.5)
	ps_e = WLanalysis.PowerSpectrum(kmapn)
	loglog(ps_e[0],ps_e[1],'--',label='Conv w/ rms=0.15+0.035z')
	
	leg=legend(loc=2, fontsize=12)
	leg.get_frame().set_visible(False)
	#setp(linewidth=2)
	xlim(ell_arr[4],ell_arr[-1])
	show()
	
	
if noise_redshift_relation:
	nbins = 20
	y, x, e1, e2, w, m = WLanalysis.readFits(KSsim_dir+'test_ells_asymm/yxewm_subfield1_zcut0213.fit').T
	e1 /= (1+m)
	e2 /= (1+m)
	z = WLanalysis.readFits('/Users/jia/weaklensing/CFHTLenS/catalogue/emulator_galpos_zcut0213/emulator_subfield1_zcut0213.fit').T[-1]
	
	z_arr = histogram(z,bins=nbins)[-1]
	e_std = zeros(nbins)
	
	for i in range(nbins):
		z0, z1 = z_arr[i:i+2]
		idx = where( (z0<z) & (z<z1) )
		if len(idx[0]) > 0:
			ie = concatenate([e1[idx],e2[idx]])
			e_std[i] = std(ie)
	eidx = where(e_std>0)
	plot(z_arr[eidx], e_std[eidx],label='CFHT')
	zz = linspace(z_arr[0],z_arr[-1],100)
	plot(zz, 0.15 + zz*0.035,label='Song&Knox 2004')
	leg=legend(fontsize=14,loc=0)
	leg.get_frame().set_visible(False)
	xlabel('z')
	ylabel('r.m.s. ellipticity')
	show()
	
if cosmo_ps:
	seed(222)
	colors=rand(10,3)
	zg = 'rz1'
	xylabels=((0,1),(0,2),(1,2))
	f = figure(figsize=(8,6))
	ic=0
	for sigmaG in sigmaG_arr[::-1]:
		#print bins, sigmaG
		centers, F_inv = cosmo_errors(sigmaG, 0, powspec = True)
		print centers, F_inv
		for k in range(1,4):
			if k == 1:
				ax=subplot(2,2,1)
			else:
				ax=subplot(2,2,k+1)
			i, j = xylabels[k-1]
			plot(centers[i],centers[j],'ko')
			triangle.error_ellipse(centers[[i,j]],F_inv[:,[i,j]][[i,j]], edgecolor=colors[ic], linewidth=2,label=str(sigmaG))
			
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
	
		
	#savefig(plot_dir+"official/sigmaG_bins%s.pdf"%(bins))
	savefig(plot_dir+"powspec_sigamG.jpg")
	close()	

if emu_checkps:
	ps_fn_arr = os.listdir(emu_dir+'powspec_sum/sigma05/')
	getps = lambda ps_fn: WLanalysis.readFits(emu_ps_dir+ps_fn)
	ps_mat = array(map(getps, ps_fn_arr))
	ps_avg = mean(ps_mat,axis=1)
	ps_std = std(ps_mat, axis=1)
	ps_CFHT = WLanalysis.readFits('/Users/jia/CFHTLenS/KSsim/powspec_sum13fields/CFHT_powspec_sigma05.fit')
	for i in range(len(ps_avg)):
		loglog(ell_arr, ps_avg[i])
	loglog(ell_arr, ps_CFHT, '--', linewidth = 2)
	show()

if emu_checkpeaks:
	#pk_CFHT = WLanalysis.readFits('/Users/jia/CFHTLenS/KSsim/peaks_sum13fields/CFHT_peaks_sigma05_025bins.fits')
	CFHT_peak = zeros(25)
	for j in arange(1,14):
		print 'adding up subfield,',j
		CFHT_peak+=WLanalysis.readFits (CFHT_dir+'CFHT_peaks_sigma18_subfield%02d_025bins.fits'%(j))
		
	pk_fn_arr = os.listdir(emu_dir+'peaks_sum/sigma05/')
	
	def getpk (pk_fn, bins = 25):
		pk600bins = WLanalysis.readFits(emu_dir+'peaks_sum/sigma05/'+pk_fn)
		pk = pk600bins.reshape(1000, -1, 600/bins)
		pk = sum(pk, axis = -1)
		return pk
	
	pk_mat = array(map(getpk, pk_fn_arr))
	pk_avg = mean(pk_mat,axis=0)
	pk_std = std(pk_mat, axis=0)
	
	for i in range(len(pk_avg)):
		plot(x, pk_avg[i])
	plot(x, CFHT_peak, '--', linewidth = 2)
	show()

ps_CFHT = WLanalysis.readFits('/Users/jia/CFHTLenS/KSsim/powspec_sum13fields/CFHT_powspec_sigma05.fit')	

if emu_checkcheckps:
	os.chdir(emu_dir+'test_cat/')
	getps = lambda i: WLanalysis.readFits('SIM_powspec_sigma05_subfield%i_emu1-512b240_Om0.136_Ol0.864_w-2.484_ns0.960_si1.034_1000r.fit'%(i))
	
	galcount = array([342966,365597,322606,380838,
		  263748,317088,344887,309647,
		  333731,310101,273951,291234,
		  308864]).astype(float) # galaxy counts for subfields, prepare for weighte sum powspec
	galcount /= sum(galcount)
	ps_arr = array(map(getps,i_arr))
	xtot = zeros(50)
	j=0
	for x in ps_arr:
		loglog(ell_arr, x)
		xtot += galcount[j]*x
		j+=1
		
	loglog(ell_arr,ps_CFHT, '--', linewidth=2)
	loglog(ell_arr,xtot, '.', linewidth=2)
	show() #look OK
#if emu_interpolate:
	