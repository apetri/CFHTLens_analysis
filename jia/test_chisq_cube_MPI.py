# Jia Liu 11/14/2014
# after found bug in power spectrum computation, this code is to 
# compute chisq cube

import numpy as np
from scipy import *
from scipy import interpolate
import os
import WLanalysis
import sys

stampede = 1
if stampede:
	from emcee.utils import MPIPool
	nn = int(sys.argv[1])
	test_dir = '/work/02977/jialiu/chisq_cube/'

else:
	test_dir = '/Users/jia/Documents/weaklensing/CFHTLenS/emulator/test_ps_bug/'
	#range from 0 to 10 for idx_arr
	#print nn # nn=0 (2pk+ps), 1 (2pk), 8 (ps, pass & ell cut)


fsky_all = 10.010646820070001
fsky_pass= 7.6645622253410002

cosmo_params = genfromtxt(test_dir+'cosmo_params.txt')
im, iw, s = cosmo_params.T

w_arr = linspace(0,-3,101)
l,ll =  100,102
om_arr = linspace(0,1.2,l)

# for SIGMA8, change params to [om, w, SIGMA]
si8_arr = linspace(0,1.6,ll)#original
#si8_arr = linspace(0.4, 1.2, ll)
#if nn==0:
	#alpha = 0.63#ps+2pk
#elif nn==1:
	#alpha = 0.6#2pk
#elif nn==8:
	#alpha = 0.64#ps
#s=(im/0.27)**alpha*s


ps_CFHT0 = concatenate(array([np.load(test_dir+fn) for fn in ('PASS_ps_CFHT.npy', 'ALL_pk_CFHT_sigmaG10.npy', 'ALL_pk_CFHT_sigmaG18.npy', 'ALL_pk_CFHT_sigmaG35.npy', 'ALL_pk_CFHT_sigmaG53.npy', 'ALL_ps_CFHT.npy','PASS_pk_CFHT_sigmaG10.npy', 'Corr_counts_sigma10_CFHT.npy', 'Corr_counts_sigma18_CFHT.npy ', 'Corr_kappa_sigma10_CFHT.npy', 'Corr_kappa_sigma18_CFHT.npy ')]),axis=-1)
ps_fidu_mat0 = concatenate([np.load(test_dir+fn) for fn in ('SIM_powspec_sigma05_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_PASS.npy',
 'SIM_peaks_sigma10_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_025bins_ALL.npy',
 'SIM_peaks_sigma18_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_025bins_ALL.npy',
 'SIM_peaks_sigma35_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_025bins_ALL.npy',
 'SIM_peaks_sigma53_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_025bins_ALL.npy',
 'SIM_powspec_sigma05_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_ALL.npy',
 'SIM_peaks_sigma10_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_025bins_PASS.npy', 'Corr_counts_sigma10_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765.npy', 'Corr_counts_sigma18_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765.npy',
 'Corr_kappa_sigma10_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765.npy',
 'Corr_kappa_sigma18_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765.npy')],axis=-1)
ps_avg0 = concatenate([np.load(test_dir+fn) for fn in ('PASS_ps_avg.npy','ALL_pk_avg_sigmaG10.npy',
'ALL_pk_avg_sigmaG18.npy', 'ALL_pk_avg_sigmaG35.npy','ALL_pk_avg_sigmaG53.npy','ALL_ps_avg.npy','PASS_pk_avg_sigmaG10.npy','peakpeak_counts_avg_10.npy','peakpeak_counts_avg_18.npy', 'peakpeak_kappa_avg_10.npy', 'peakpeak_kappa_avg_18.npy')], axis=-1)

####### 0:50 pass ps; 
####### 50:75 peak 10; 
####### 75:100 peak 18;
####### 100:125 peak 35;
####### 125:150 peak 53;
####### 150:200 all powspec
#### 1) 2 smoothings + powspec
#idx_full = arange(11,100)
idx_full = delete(arange(11,100), arange(50-12,50)-11)
#### 2) 2 smoothing
idx_pk2 = arange(50,100)
#### 3) 1.0 smoothing
idx_pk10 = arange(50,75)
#### 4) 1.8 smoothing
idx_pk18 = arange(75,100)
#### 5) 3.5 smoothing
idx_pk35 = arange(100,125)
#### 6) 5.3 smoothing
idx_pk53 = arange(125,150)
#### 7) ps pass
idx_psPass = arange(11,50)
#### 8) ps all
idx_psAll = arange(161,200)
#### 9) ps pass ell26
idx_psPass7000 = arange(11,50-12)
#### 10) ps all ell26
idx_psAll7000 = arange(161,200-12)
#### 11) referee test: pk_10 PASS (1/15/2015)
idx_pk10PASS = arange(200,225)
#### 11) referee test: pk_10 PASS (1/15/2015)
idx_ps_cut = idx_psPass[:-2]
#### 12) idx
idx_peakpeak_counts10 = arange(225, 250)
idx_peakpeak_counts18 = arange(250, 275)
idx_peakpeak_kappa10 = arange(275, 300)
idx_peakpeak_kappa18 = arange(300, 325)


idx_arr = [idx_full, idx_pk2, idx_pk10, idx_pk18, idx_pk35, idx_pk53, idx_psPass, idx_psAll, idx_psPass7000, idx_psAll7000, idx_pk10PASS, idx_ps_cut, idx_peakpeak_counts10, idx_peakpeak_counts18, idx_peakpeak_kappa10 , idx_peakpeak_kappa18 ]
fn_arr = ['idx_psPass7000_pk2smoothing', 'pk2smoothing', 'pk10', 'pk18', 'pk35', 'pk53', 'psPass', 'psAll', 'psPass7000', 'psAll7000','idx_pk10PASS', 'idx_ps_cut', 'idx_peakpeak_counts10', 'idx_peakpeak_counts18', 'idx_peakpeak_kappa10' , 'idx_peakpeak_kappa18']

def return_interp_cosmo_for_idx (idx):
	ps_CFHT_test = ps_CFHT0[idx]
	idx = idx[where(ps_CFHT_test>0)[0]]#rid of zero bins
	
	ps_CFHT = ps_CFHT0[idx]
	ps_fidu_mat = ps_fidu_mat0[:,idx]
	ps_avg = ps_avg0[:,idx]
	cov_mat = cov(ps_fidu_mat,rowvar=0)
	cov_inv = mat(cov_mat).I

	spline_interps = list()
	for ibin in range(ps_avg.shape[-1]):
		ps_model = ps_avg[:,ibin]
		iinterp = interpolate.Rbf(im, iw, s, ps_model)
		spline_interps.append(iinterp)

	def interp_cosmo (params, method = 'multiquadric'):
		'''Interpolate the powspec for certain param.
		Params: list of 3 parameters = (om, w, si8)
		Method: "multiquadric" for spline (default), and "GP" for Gaussian process.
		'''
		mm, wm, sm = params
		gen_ps = lambda ibin: spline_interps[ibin](mm, wm, sm)
		ps_interp = array(map(gen_ps, range(ps_avg.shape[-1])))
		ps_interp = ps_interp.reshape(-1,1).squeeze()
		return ps_interp
	return interp_cosmo, cov_mat, cov_inv, ps_CFHT

def plot_heat_map_w (values):
	w, idx, interp_cosmo, cov_inv, ps_CFHT = values
	#fn = test_dir+'test/chisqcube_SIGMA_%s_w%s.npy'%(fn_arr[nn], w)
	#fn = test_dir+'test/chisqcube_%s_w%s.npy'%(fn_arr[nn], w)
	#if os.path.isfile(fn) == False:
	print 'w=',w 
	heatmap = zeros(shape=(l,ll))
	for i in range(l):
		for j in range(ll):
			best_fit = (om_arr[i], w, si8_arr[j])
			ps_interp = interp_cosmo(best_fit)
			del_N = np.mat(ps_interp - ps_CFHT)
			chisq = float(del_N*cov_inv*del_N.T)
			heatmap[i,j] = chisq
		#save(fn, heatmap)
	#else:
		#print 'w=', w, 'done'
	return heatmap

#cov_mat = cov(ps_fidu_mat0[:,arange(10,100)],rowvar=0)
#X, Y = np.meshgrid(sqrt(diag(cov_mat)), sqrt(diag(cov_mat)))

#imshow(cov_mat / (X*Y), origin='lower', interpolation='nearest',vmin=-0.25,vmax=0.3,extent=(1,90,1,90))
#colorbar()
##xlabel(r'${\rm i}$',fontsize=20)
##ylabel(r'${\rm j}$',fontsize=20)
#savefig('/Users/jia/Documents/weaklensing/CFHTLenS/paper_all/CFHTpaper/CorrCoeff.pdf')
#close()
#####################01/15/2014 ############
#### referee continued, get cube for Sigma8
############################################
#pool=MPIPool()
#idx = idx_arr[nn]
#interp_cosmo, cov_mat, cov_inv, ps_CFHT = return_interp_cosmo_for_idx (idx)
#values = [[w, idx, interp_cosmo, cov_inv, ps_CFHT] for w in w_arr]


#pool.map(plot_heat_map_w, values)

#cube = array([load('/home1/02977/jialiu/chisq_cube/test/chisqcube_SIGMA_%s_w%s.npy'%(fn_arr[nn], w)) for w in w_arr])

#save(test_dir+'covmat_%s_SIGMA.npy'%(fn_arr[nn]), cov_mat)
#save(test_dir+'chisqcube_%s_SIGMA.npy'%(fn_arr[nn]), cube)

#####################01/15/2014 ############
#### referee report tests ##################
############################################
#def idx2P (idx):
	#'''quick test, for certain idx, return probability plan (2D), with w=-1'''
	#interp_cosmo, cov_mat, cov_inv, ps_CFHT = return_interp_cosmo_for_idx (idx)
	#values = [-1.0, idx, interp_cosmo, cov_inv, ps_CFHT]
	#heatmap = plot_heat_map_w (values)
	#P = exp(-heatmap/2)
	#P /= sum(P)
	#V = WLanalysis.findlevel(P)
	#return P, V

## 1) pass pk
#P_pk10ALL, V_pk10ALL = idx2P (idx_pk10)
#P_pk10PAS, V_pk10PAS = idx2P (idx_pk10PASS)

## 2) ell cut at 2e5
##P_psPASS, V_psPASS = idx2P (idx_psPass)
##P_psCut,  V_psCut = idx2P (idx_ps_cut)

#X, Y = np.meshgrid(om_arr, si8_arr)

#f=figure()
#ax=f.add_subplot(111)
#lines=[]
############# peaks
#CS=ax.contour(X, Y, P_pk10ALL.T, levels=V_pk10ALL[:-1], origin='lower', colors='r')
#lines.append(CS.collections[0])
#CS=ax.contour(X, Y, P_pk10PAS.T, levels=V_pk10PAS[:-1], origin='lower', colors='b')
#lines.append(CS.collections[0])
#labels=('peaks 1 arcmin (all)', 'peaks 1 arcmin (pass)')
############# power spectrum
##CS=ax.contour(X, Y, P_psPASS.T, levels=V_psPASS[:-1], origin='lower', colors='r')
##lines.append(CS.collections[0])
##CS=ax.contour(X, Y, P_psCut.T, levels=V_psCut[:-1], origin='lower', colors='b')
##lines.append(CS.collections[0])
##labels=('all ell', 'ell < 20193')

#leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':20},loc=0)
#leg.get_frame().set_visible(False)
#xlim(0,0.8)
#ylim(0.3,1.3)
#xlabel(r'$\rm{\Omega_m}$',fontsize=20)
#ylabel(r'$\rm{\sigma_8}$',fontsize=20)
#show()

#####################12/25/2014 ############
#### test alpha as function of SNR cut #####
############################################
#kappa_arr = linspace(-0.0368,0.1168,25)
#repeat_elem = lambda aP: (repeat(aP[0], aP[1]*1e5).reshape(2,-1)).T
#X, Y = np.meshgrid(om_arr, si8_arr)
#all_points = array([X.flatten(),Y.flatten()]).T
#alpha_arr = linspace(0.4,0.8,301)
#def findalpha(idxcut=20):
	#idx_pk10high = arange(50+idxcut,75)# 16th bin = 0.066, 20th bin = 0.091
	#interp_cosmo, cov_mat, cov_inv, ps_CFHT = return_interp_cosmo_for_idx (idx_pk10high)
	#w = -1
	#values = (w, idx_pk10high, interp_cosmo, cov_inv, ps_CFHT)
	#heatmap = plot_heat_map_w (values)
	#P = exp(-heatmap/2.0)
	#P[67:,85:]=0
	#P[:,:20]=0
	#P/=sum(P)
	
	#all_prob0 = (P.T).flatten()#/amax(P)
	#idx = where(all_prob0*1e4>2)[0]#only care about points with larger prob.
	#iall_points, all_prob = all_points[idx], all_prob0[idx]
	#samples = concatenate(map(repeat_elem, array([iall_points,all_prob]).T),axis=0)
	#Sigma8 = lambda alpha: std((samples.T[0]/0.27)**alpha*samples.T[1])
	#Sigma8_arr = array(map(Sigma8, alpha_arr))
	#alpha = alpha_arr[argmin(Sigma8_arr)]
	#SNR = kappa_arr[idxcut]/0.033
	#print idxcut, SNR, alpha
	#return P, SNR, alpha
#a=map(findalpha,(0,16, 18, 20))
#f = figure(figsize=(8,6))
#ax=f.add_subplot(111)
#lines=[]
#labels=['SNR > %.1f, alpha = %.2f'%(a[i][1], a[i][2]) for i in arange(len(a))]
#extents = (om_arr[0], om_arr[-1], si8_arr[0], si8_arr[-1])
#i=0
#icolors = ('r','b','m','g','k','g','r','c','b','g','m','k')
#for ia in a:
	#iP, inu, ialpha = ia
	#V = WLanalysis.findlevel(iP)
	#CS = ax.contour(X[20:85, 0:67], Y[20:85, 0:67], iP[0:67,20:85].T, levels=[V[0],], colors=icolors[i], origin='lower', extent=extents, linewidths=2)
	#lines.append(CS.collections[0])
	#i+=1
#leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
#ax.set_xlabel('Omega_m',fontsize=20)
#ax.set_ylabel('sigma_8',fontsize=20)
#show()
############################################


###########################################################
############ operation ####################################
##########################################################
pool=MPIPool()
print 'boo'
idx = idx_arr[nn]

interp_cosmo, cov_mat, cov_inv, ps_CFHT = return_interp_cosmo_for_idx (idx)
print 'boo2'
values = [[w, idx, interp_cosmo, cov_inv, ps_CFHT] for w in w_arr]

pool.map(plot_heat_map_w, values)

cube = array([load('/work/02977/jialiu/chisq_cube/test/chisqcube_%s_w%s.npy'%(fn_arr[nn], w)) for w in w_arr])

save(test_dir+'covmat_%s.npy'%(fn_arr[nn]), cov_mat)
save(test_dir+'chisqcube_%s.npy'%(fn_arr[nn]), cube)

############### junk below
#def chisq2P(chisq_mat):#(idx=idx_full,w=-1):#aixs 0-w, 1-om, 2-si8
	##chisq_mat = plot_heat_map_w (idx=idx_full,w=-1)
	#P = exp(-(chisq_mat-amin(chisq_mat))/2)
	#P /= sum(P)
	#V = WLanalysis.findlevel(P)
	#return P, V

#idx_arr = [idx_full, idx_ps, idx_pk2]#, idx_pk10ps, idx_pk18ps, idx_pk10, idx_pk18]
#chisq_arr = map(plot_heat_map_w, idx_arr)
#P_arr0 = map(chisq2P, chisq_arr)
#labels0 = ['full', 'ps', 'pk2', 'pk10ps', 'pk18ps', 'pk10', 'pk18']
#P_arr = (P_arr0[0][0], P_arr0[1][0]*P_arr0[2][0])
#labels = ['full', 'ps*pk2']

#X, Y = np.meshgrid(om_arr, si8_arr)

#from pylab import *
#f = figure(figsize=(8,8))
#ax=f.add_subplot(111)
#lines=[]
#colors = ('r','b','g','m','y','k')
#for i in arange(len(P_arr)):
	#P = P_arr[i]
	#V=WLanalysis.findlevel(P)
	#CS = ax.contour(X, Y, P.T, levels=V[:-1], origin='lower', extent=(om_arr[0], om_arr[-1], si8_arr[0], si8_arr[-1]), colors=colors[i])#, linewidths=lws2[i+1], linestyles=lss2[i+1])
	#lines.append(CS.collections[0])
#leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':20},loc=0)
#ax.tick_params(labelsize=16)
#ax.set_xlabel(r'$\rm{\Omega_m}$',fontsize=20)
#ax.set_ylabel(r'$\rm{\sigma_8}$',fontsize=20)
#leg.get_frame().set_visible(False)
##show()
#savefig(test_dir+'plot/contour_test_w-1_bugfix.jpg')
#close()

######## chisq cube ###############
#pool = MPIPool()
#chisq_cube = map(plot_heat_map_w, w_arr)
#if cut7000 <39:
	#np.save(test_dir+'%s_chisqcube_ps_ell7000'%(BG), array(chisq_cube).reshape(-1))
#elif BG == 'COMB_pk':
	#np.save(test_dir+'COMB_chisqcube_cutPeaks', array(chisq_cube).reshape(-1))
#else:
	#np.save(test_dir+'%s_chisqcube_ps'%(BG), array(chisq_cube).reshape(-1))

######### test for w=1 ###########


print 'done'
