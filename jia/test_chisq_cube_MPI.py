# Jia Liu 11/14/2014
# after found bug in power spectrum computation, this code is to 
# compute chisq cube

import numpy as np
from scipy import *
from scipy import interpolate
import os
import WLanalysis
from emcee.utils import MPIPool
import sys

cut7000 = 100 #int(sys.argv[2])

fsky_all = 10.010646820070001
fsky_pass= 7.6645622253410002

#test_dir = '/Users/jia/Documents/weaklensing/CFHTLenS/emulator/test_ps_bug/'
test_dir = '/home1/02977/jialiu/chisq_cube/'
cosmo_params = genfromtxt(test_dir+'cosmo_params.txt')
im, iw, s = cosmo_params.T

w_arr = linspace(0,-3, 3)
l, ll = 5, 5
#w_arr = linspace(0,-3,101)
#l,ll =  100,102
om_arr = linspace(0,1.2,l)
si8_arr = linspace(0,1.6,ll)

ps_CFHT0 = concatenate(array([np.load(test_dir+fn) for fn in ('PASS_ps_CFHT.npy', 'ALL_pk_CFHT_sigmaG10.npy', 'ALL_pk_CFHT_sigmaG18.npy', 'ALL_pk_CFHT_sigmaG35.npy', 'ALL_pk_CFHT_sigmaG53.npy', 'ALL_ps_CFHT.npy')]),axis=-1)
ps_fidu_mat0 = concatenate([np.load(test_dir+fn) for fn in ('SIM_powspec_sigma05_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_PASS.npy',
 'SIM_peaks_sigma10_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_025bins_ALL.npy', 'SIM_peaks_sigma18_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_025bins_ALL.npy',
 'SIM_peaks_sigma35_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_025bins_ALL.npy',
 'SIM_peaks_sigma53_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_025bins_ALL.npy',
 'SIM_powspec_sigma05_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_ALL.npy')],axis=-1)
ps_avg0 = concatenate([np.load(test_dir+fn) for fn in ('PASS_ps_avg.npy','ALL_pk_avg_sigmaG10.npy',
'ALL_pk_avg_sigmaG18.npy', 'ALL_pk_avg_sigmaG35.npy','ALL_pk_avg_sigmaG53.npy','ALL_ps_avg.npy',)], axis=-1)

####### 0:50 pass ps; 
####### 50:75 peak 10; 
####### 75:100 peak 18;
####### 100:125 peak 35;
####### 125:150 peak 53;
####### 150:200 all powspec
#### 1) 2 smoothings + powspec
idx_full = delete(arange(11,100), arange(50-12,50))
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

idx_arr = [idx_full, idx_pk2, idx_pk10, idx_pk18, idx_pk35, idx_pk53, idx_psPass, idx_psAll, idx_psPass7000, idx_psAll7000]
fn_arr = ['idx_psPass7000_pk2smoothing', 'pk2moothing', 'pk10', 'pk18', 'pk35', 'pk53', 'psPass', 'psAll', 'psPass7000', 'psAll7000']

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
		im, wm, sm = params
		gen_ps = lambda ibin: spline_interps[ibin](im, wm, sm)
		ps_interp = array(map(gen_ps, range(ps_avg.shape[-1])))
		ps_interp = ps_interp.reshape(-1,1).squeeze()
		return ps_interp
	return interp_cosmo, cov_mat, cov_inv, ps_CFHT

def plot_heat_map_w (values):
	w, idx, interp_cosmo, cov_inv, ps_CFHT = values
	print 'w=',w
	heatmap = zeros(shape=(l,ll))
	for i in range(l):
		for j in range(ll):
			best_fit = (om_arr[i], w, si8_arr[j])
			
			ps_interp = interp_cosmo(best_fit)
			print 'idx, len(ps_interp), len(ps_CFHT)', len(ps_interp), len(ps_CFHT)
			del_N = np.mat(ps_interp - ps_CFHT)
			chisq = float(del_N*cov_inv*del_N.T)
			heatmap[i,j] = chisq
	return heatmap

###########################################################
############ operation ####################################
###########################################################
pool=MPIPool()
i = 0
for idx in idx_arr[:2]:
	print fn_arr[i]
	interp_cosmo, cov_mat, cov_inv, ps_CFHT = return_interp_cosmo_for_idx (idx)
	values = [[w, idx, interp_cosmo, cov_inv, ps_CFHT] for w in w_arr]
	cube = array(pool.map(plot_heat_map_w, values))
	#cube = array(map(plot_heat_map_w, values))
	save(test_dir+'test_chisqcube_%s.npy'%(fn_arr[i]), cube)
	save(test_dir+'test_covmat_%s.npy'%(fn_arr[i]), cov_mat)
	i+=1


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
