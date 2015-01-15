# Jia Liu 11/28/2014
# This code is a cleaner versino of emu_spline.py


import numpy as np
import triangle
from scipy import *
import scipy.optimize as op
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

test_dir = '/Users/jia/CFHTLenS/emulator/test_ps_bug/'
plot_dir = test_dir+'plot/'

contour_peaks_smoothing = 0
contour_ps_fieldselect = 0
contour_peaks_powspec = 0
contour_including_w = 0
contour_peaks_powspec_covariance = 0

good_bad_powspec = 0
interp_2D_plane = 0
good_bad_peaks = 0
sample_interpolation = 0
sample_points = 0
SIGMA_contour = 1

########### constants ####################
l=100
ll=102
om0,om1 = 0, 67
si80,si81 = 20,85
w0,w1 = 10,70
om_arr = linspace(0,1.2,l)[om0:om1]
si8_arr = linspace(0,1.6,ll)[si80:si81]
w_arr = linspace(0,-3,101)#[w0:w1]
sigmaG_arr = [1.0, 1.8, 3.5, 5.3]
fn_arr = ['idx_psPass7000_pk2smoothing', 'pk2smoothing', 'pk10', 'pk18', 'pk35', 'pk53', 'psPass', 'psAll', 'psPass7000', 'psAll7000', 'idx_psPass7000_pk2smoothing_SIMGA', 'pk2smoothing_SIMGA', 'psPass7000_SIMGA']

cube_arr = array([load(test_dir+'chisqcube_%s.npy'%(fn)) for fn in fn_arr])

colors=('r','b','m','c','k','g','r','c','b','g','m','k')
lss =('solid', 'dashed', 'solid', 'dashed', 'dashdot', 'dotted','dashdot', 'dotted')
lss3=('-','-.','--',':','.')
lws = (4,4,2,2,4,4,2,2,4,4,2,2)

############################################

def cube2P(chisq_cube, axis=0, nocut = 0):#aixs 0-w, 1-om, 2-si8
	if nocut==0:	
		if axis==0:
			chisq_cube = chisq_cube[w0:w1,om0:om1,si80:si81]
		else:
			chisq_cube = chisq_cube[:,om0:om1,si80:si81]
	P = sum(exp(-chisq_cube/2),axis=axis)
	P /= sum(P)
	return P

extents = (om_arr[0], om_arr[-1], si8_arr[0], si8_arr[-1])
def official_contour (cube_arr, labels, nlev, fn, colors, lws, lss, include_w = 0):
	f = figure(figsize=(8,8))
	ax=f.add_subplot(111)
	lines=[]
	X, Y = np.meshgrid(om_arr, si8_arr)
	
	for i in arange(len(cube_arr)):
		cube = cube_arr[i]	
		if include_w:
			ix,iy = 1, 0
			P=cube2P(cube, axis=2)
			X, Y = np.meshgrid(w_arr, om_arr)
		else:
			P=cube2P(cube)
		V=WLanalysis.findlevel(P)
		print fn,'include_w %s\t 1sigma: %.5f\t 2sigma: %.5f'%(bool(include_w), 100*len(where(P>V[0])[0])/512.**2, 100*len(where(P>V[1])[0])/512.**2)
		if include_w and i ==2:
			CS = ax.contourf(X, Y, P.T, levels=[V[0],V[1]], origin='lower', extent=extents, colors=colors[i], linewidths=lws[i], linestyles=lss[i],alpha=0.85)
		else:
			CS = ax.contour(X, Y, P.T, levels=[V[0],], origin='lower', extent=extents, colors=colors[i], linewidths=lws[i], linestyles=lss[i])
			lines.append(CS.collections[0])
		if nlev == 2:
			CS2 = ax.contour(X, Y, P.T, levels=[V[1],], alpha=0.7, origin='lower', extent=extents, colors=colors[i], linewidths=lws[i], linestyles=lss[i])

	
	if include_w:
		ax.set_ylabel(r'$\rm{\Omega_m}$',fontsize=20)
		ax.set_xlabel(r'$\rm{w}$',fontsize=20)
	else:
		leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':20},loc=0)
		ax.tick_params(labelsize=16)
		ax.set_xlabel(r'$\rm{\Omega_m}$',fontsize=20)
		ax.set_ylabel(r'$\rm{\sigma_8}$',fontsize=20)
		leg.get_frame().set_visible(False)
	savefig(plot_dir+fn+'.pdf')
	close()

if contour_peaks_smoothing:
	colors = ['r','b','g','m','k']
	lws = [4,4,2,2,4]
	lss =('dotted', 'dashed', 'solid', 'dashed', 'solid')
	labels = [r'$\rm{%.1f\, arcmin}$'%(sigmaG) for sigmaG in sigmaG_arr]
	labels.append(r'$\rm{1.0+1.8\, arcmin}$')
	icube_arr = cube_arr[[2,3,4,5,1]]
	fn = 'contour_peaks_smoothing'
	nlev = 1
	official_contour (icube_arr, labels, nlev, fn, colors, lws, lss)
	
if contour_ps_fieldselect:
	colors = ['c','b','g','r','k']
	lws = [2,2,4,4]
	lss =( 'dashed', 'solid', 'dashed', 'solid')
	labels = [r'$\rm{pass\, fields}$', r'$\rm{all\, fields}$', r'$\rm{pass\, fields(\ell<7,000)}$', r'$\rm{all\, fields(\ell<7,000)}$']
	icube_arr = cube_arr[[6, 7, 8, 9]]
	fn = 'contour_powspec_fieldselect'
	nlev = 1
	official_contour (icube_arr, labels, nlev, fn, colors, lws, lss)
	
if contour_peaks_powspec:
	colors = ['g','m','k']
	lws = [4,2,4]
	lss =('dashed', 'solid', 'solid')
	labels = [r'$\rm{power\, spectrum}$', r'$\rm{peaks\, (1.0 + 1.8\,arcmin)}$', r'$\rm{power\, spectrum + peaks}$']
	icube_arr = cube_arr[[8, 1, 0]]
	fn = 'contour_peaks_powspec'
	nlev = 2
	if contour_including_w:
		fn = fn+'_w'
	official_contour (icube_arr, labels, nlev, fn, colors, lws, lss, include_w = contour_including_w)
	
if contour_peaks_powspec_covariance:
	colors = ['k','r']
	lws = [2,4]
	lss =('solid', 'dashed')
	labels = [r'$\rm{with\, covariance}$', r'$\rm{without\, covariance}$']
	icube_arr = [cube_arr[0], cube_arr[1]+cube_arr[-2]]
	fn = 'contour_peaks_powspec_covariance'
	nlev = 2
	official_contour (icube_arr, labels, nlev, fn, colors, lws, lss)
	
fsky_all = 10.010646820070001
fsky_pass= 7.6645622253410002
gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
ell_arr = logspace(log10(110.01746692),log10(25207.90813028),50)[11:]
if good_bad_powspec:
	
	#ps_CFHT_all = np.load(test_dir+'ALL_ps_CFHT.npy')[11:]/fsky_all
	#ps_CFHT = np.load(test_dir+'PASS_ps_CFHT.npy')[11:]/fsky_pass
	
	ps_fidu = np.load(test_dir+'SIM_powspec_sigma05_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_PASS.npy')[:,11:]/fsky_pass
	
	ps_fidu_all = np.load(test_dir+'SIM_powspec_sigma05_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_ALL.npy')[:,11:]/fsky_all
	
	
	fidu_std = std(ps_fidu, axis=0)
	fidu_avg = mean(ps_fidu, axis=0)
	
	ps_CFHT_all = mean(ps_fidu_all, axis=0)
	ps_CFHT = fidu_avg
	#ps_CFHT_all = mean(ps_fidu_all, axis=0)/ps_CFHT_all[20]*ps_CFHT[20]
	
	lw=4
	
	f=figure(figsize=(10,8))
	ax=f.add_subplot(gs[0])
	ax2=f.add_subplot(gs[1],sharex=ax)
	
	ax.plot(ell_arr, ps_CFHT, 'g-', label=r'$\rm{pass\, fields}$',linewidth=lw)	
	ax.errorbar(ell_arr, ps_CFHT, fidu_std, color='g', linewidth=lw)
	ax.plot(ell_arr, ps_CFHT_all, 'm--', label=r'$\rm{all\, fields}$',linewidth=lw)
	ax2.plot(ell_arr, zeros(len(ell_arr)), 'g-', label=r'$\rm{pass\, fields}$',linewidth=lw)
	ax2.errorbar(ell_arr, zeros(len(ell_arr)),fidu_std/fidu_avg, color='g', linewidth=lw)
	ax2.plot(ell_arr, ps_CFHT_all/ps_CFHT-1,'m--', label=r'$\rm{all\, fields}$',linewidth=lw)
	
	
	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=0)
	leg.get_frame().set_visible(False)
	plt.subplots_adjust(hspace=0.0,left=0.15)
	plt.setp(ax.get_xticklabels(), visible=False) 
	ax.set_xlim(ell_arr[0],ell_arr[-1])
	ax.tick_params(labelsize=16)
	ax2.tick_params(labelsize=16)
					
	#ax.set_title(r'$\rm{CFHTLenS}$',fontsize=24)
	ax.set_title(r'$\rm{Simulation}$',fontsize=24)
	ax2.set_ylim(-0.1, 0.1)
	ax2.set_yticks(linspace(-0.05,0.05,3))
	
	fn=plot_dir+'good_bad_powspec_fidu.pdf'
	ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$',fontsize=20)
	ax.set_xscale('log')
	ax2.set_xscale('log')
	ax2.set_xlabel(r'$\ell$')
	ax.set_yscale('log')
	ax2.set_ylabel(r'$\rm{{\Delta}P/P}$',fontsize=20)
	ax.set_ylim(1e-5, 1e-2)
	savefig(fn)	
	close()

if SIGMA_contour:
	#cube_ps, cube_pk, cube_pspk = cube_arr[-3:]
	Pcomb, Ppk, Pps = [cube2P(cube, nocut=1) for cube in cube_arr[-3:]]
	
	Pps_marg = sum(Pps, axis=0)/sum(Pps)
	Ppk_marg = sum(Ppk, axis=0)/sum(Ppk)	
	Pcomb_marg = sum(Pcomb, axis=0)/sum(Pcomb)
	S_arr = linspace(0.4, 1.2, ll)
	def findlevel1D (prob, xvalues):
		idx = argmax(prob)
		bestfit = xvalues[idx]
		sortprob = sort(prob)[::-1]
		tolprob = cumsum(sortprob)
		idx = where(abs(tolprob-0.68) == sort(abs(tolprob-0.68))[0])
		val = sortprob[idx]
		left,right=xvalues[where(prob>val)][[0,-1]]-bestfit
		print 'bestfit, left, right', bestfit, left, right
		return bestfit, left, right

	findlevel1D(Pps_marg, S_arr)
	findlevel1D(Ppk_marg, S_arr)
	findlevel1D(Pcomb_marg, S_arr)
	
	f = figure(figsize=(8,6))
	ax=f.add_subplot(111)
	ax.plot(S_arr, Pps_marg,'g--', label=r'$\rm{power\,spectrum}\,(\alpha=0.64)$',linewidth=2)
	ax.plot(S_arr, Ppk_marg,'m-',label=r'$\rm{peaks}\,(\alpha=0.60)$',linewidth=1)
	ax.plot(S_arr, Pcomb_marg, 'k-',label=r'$\rm{power\, spectrum + peaks}\,(\alpha=0.63)$',linewidth=2)
	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=0)
	leg.get_frame().set_visible(False)
	ax.set_xlabel(r'$\rm{\Sigma_8=\sigma_8(\Omega_m/0.27)^\alpha}$',fontsize=20)
	ax.set_ylabel(r'$\rm{Probability}$',fontsize=20)
	ax.set_xlim(0.5, 1.2)
	ax.set_ylim(0.0, 0.05)
	savefig(plot_dir+'SIGMA_marg_prob.pdf')
	close()

SIGMA_contour_junk = 0
if SIGMA_contour_junk:
	# quote delta Sigma = simga_8*(omega_m/0.26)^0.6
	# steps: 
	# 1) find alpha
	# 2) interpolator for Sigma
	# 3) marginalized over w, and fixed w
	print 'SIGMA_contour'
	
	######### the following block is to find alpha for ps, pk ######
	P_ps = cube2P(cube_arr[-2])
	P_pk = cube2P(cube_arr[1])
	P_comb = cube2P(cube_arr[0])
	X, Y = np.meshgrid(om_arr, si8_arr)
	all_points = array([X.flatten(),Y.flatten()]).T
	repeat_elem = lambda aP: (repeat(aP[0], aP[1]*1e5).reshape(2,-1)).T
	alpha_arr = linspace(0.4,0.8,301)
	def findalpha (P, fn=None):
		all_prob0 = (P.T).flatten()#/amax(P)
		idx = where(all_prob0*1e4>2)[0]#only care about points with larger prob.
		iall_points, all_prob = all_points[idx], all_prob0[idx]
		samples = concatenate(map(repeat_elem, array([iall_points,all_prob]).T),axis=0)
		Sigma8 = lambda alpha: std((samples.T[0]/0.27)**alpha*samples.T[1])
		Sigma8_arr = array(map(Sigma8, alpha_arr))
		alpha = alpha_arr[argmin(Sigma8_arr)]
		print fn, alpha
		return alpha
	#findalpha(P_ps,'ps')
	#findalpha(P_pk, 'pk')
	#findalpha(P_comb, 'comb')
	#######################################################
	
	######### build interpolator for 2D array ###
	######### alpha = comb:0.63, pk:0.60, ps:0.64
	import test_chisq_cube_MPI as tcs
	nn_arr = [[tcs.idx_psPass7000, 0.64, 'ps'],[tcs.idx_pk2, 0.60, 'pk'],[tcs.idx_full, 0.63, 'combfix']]
	S_arr = linspace(0.5, 1.1, 150)
	w_arr = linspace(-2.2, -0.2, 70)
	def heatmap_sigma8 (nn):
		idx, alpha, fn = nn
		print 'fn, alpha:', fn, alpha
		if os.path.isfile(test_dir+'Sigma8_chisq_%s.npy'%(fn))==False:
			interp_cosmo, cov_mat, cov_inv, ps_CFHT = tcs.return_interp_cosmo_for_idx (idx, alpha=alpha)
			heatmap = zeros(shape=(len(S_arr),len(w_arr)))
			for i in range(len(S_arr)):
				print S_arr[i]
				for j in range(len(w_arr)):
					best_fit = (S_arr[i],w_arr[j])
					ps_interp = interp_cosmo(best_fit)
					del_N = np.mat(ps_interp - ps_CFHT)
					chisq = float(del_N*cov_inv*del_N.T)
					heatmap[i,j] = chisq
			save(test_dir+'Sigma8_chisq_%s.npy'%(fn), heatmap)
		else:
			heatmap = load(test_dir+'Sigma8_chisq_%s.npy'%(fn))
		return heatmap
	#heatps, heatpk, heatcomb = map(heatmap_sigma8, nn_arr)
	
	############ final plots to find SIGMA########
	Pps, Ppk, Pcomb = exp(-array(map(heatmap_sigma8, nn_arr))/2)
	
	Pps_marg = sum(Pps, axis=1)/sum(Pps)
	Ppk_marg = sum(Ppk, axis=1)/sum(Ppk)	
	Pcomb_marg = sum(Pcomb, axis=1)/sum(Pcomb)
	
	def findlevel1D (prob, xvalues):
		idx = argmax(prob)
		bestfit = xvalues[idx]
		sortprob = sort(prob)[::-1]
		tolprob = cumsum(sortprob)
		idx = where(abs(tolprob-0.68) == sort(abs(tolprob-0.68))[0])
		val = sortprob[idx]
		left,right=xvalues[where(prob>val)][[0,-1]]-bestfit
		print 'bestfit, left, right', bestfit, left, right
		return bestfit, left, right

	findlevel1D(Pps_marg, S_arr)
	findlevel1D(Ppk_marg, S_arr)
	findlevel1D(Pcomb_marg, S_arr)
	
	f = figure(figsize=(8,6))
	ax=f.add_subplot(111)
	ax.plot(S_arr, Pps_marg,'g--', label=r'$\rm{power\,spectrum}\,(\alpha=0.64)$',linewidth=2)
	ax.plot(S_arr, Ppk_marg,'m-',label=r'$\rm{peaks}\,(\alpha=0.60)$',linewidth=1)
	ax.plot(S_arr, Pcomb_marg, 'k-',label=r'$\rm{power\, spectrum + peaks}\,(\alpha=0.63)$',linewidth=2)
	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=0)
	leg.get_frame().set_visible(False)
	ax.set_xlabel(r'$\rm{\Sigma_8=\sigma_8(\Omega_m/0.27)^\alpha}$',fontsize=20)
	ax.set_ylabel(r'$\rm{Probability}$',fontsize=20)
	ax.set_xlim(0.5, 1.2)
	ax.set_ylim(0.0, 0.05)
	savefig(plot_dir+'SIGMA_marg_prob.pdf')
	close()
	