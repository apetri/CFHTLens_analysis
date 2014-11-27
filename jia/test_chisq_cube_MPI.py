# Jia Liu 11/14/2014
# after found bug in power spectrum computation, this code is to 
# compute chisq cube
# run with:
# ibrun python test_chisq_cube_MPI.py ALL 100 x
# ibrun python test_chisq_cube_MPI.py ALL 27
# ibrun python test_chisq_cube_MPI.py BAD 27 x
# ibrun python test_chisq_cube_MPI.py BAD 100 x
# ibrun python test_chisq_cube_MPI.py COMB_pk 100

import numpy as np
from scipy import *
from scipy import interpolate
import os
import WLanalysis
from emcee.utils import MPIPool
import sys

BG = 'COMB_pk'#str(sys.argv[1]) #bad or good
cut7000 = 100 #int(sys.argv[2])
#test_dir = '/home1/02977/jialiu/ps_chisq_compute/'

test_dir = '/Users/jia/Documents/weaklensing/CFHTLenS/emulator/test_ps_bug/'
cosmo_params = genfromtxt(test_dir+'cosmo_params.txt')
m, w, s = cosmo_params.T

print BG, cut7000
#w_arr = linspace(0,-3,3)
#l,ll=5,5
w_arr = linspace(0,-3,101)
l,ll =  20,20#100,102
om_arr = linspace(0,1.2,l)
si8_arr = linspace(0,1.6,ll)
#best_fit = (om_arr[10],-1,si8_arr[24])
#ps_CFHT = np.load(test_dir+'%s_ps_CFHT.npy'%(BG))[:cut7000]
#ps_fidu_mat = np.load(test_dir+'%s_ps_fidu39.npy'%(BG))[:,:cut7000]
#ps_avg = np.load(test_dir+'%s_avg_ps.npy'%(BG))[:,:cut7000]

ps_CFHT0 = np.load(test_dir+'%s_ps_CFHT.npy'%(BG))
ps_fidu_mat0 = np.load(test_dir+'%s_ps_fidu39.npy'%(BG))
ps_avg0 = np.load(test_dir+'%s_avg_ps.npy'%(BG))

#### 1) 2 smoothings + powspec
idx_full = range(89)
#### 2) 1.0 smoothing + powspec
idx_pk10ps = range(25,89)
#### 3) 1.0 smoothing + powspec
idx_pk18ps = delete(range(89),range(25,50))
#### 4) 1.0 smoothing
idx_pk10 = range(25)
#### 5) 1.8 smoothing
idx_pk18 = range(25,50)
#### 6) ps
idx_ps = range(50,89)

## correlation matrix plotting
#x = array(del_N).squeeze()#sqrt(diag(cov_mat))
#X,Y=np.meshgrid(x,x)
#corr_mat = array(cov_inv)*X*Y
##corr_mat = cov_mat/(X*Y)
#imshow(corr_mat,origin='lower')#,vmin=-0.3,vmax=0.3)
#colorbar()
#show()

def return_interp_cosmo_for_idx (idx):
	ps_CFHT = ps_CFHT0[idx]
	ps_fidu_mat = ps_fidu_mat0[:,idx]
	ps_avg = ps_avg0[:,idx]
		
	cov_mat = cov(ps_fidu_mat,rowvar=0)
	cov_inv = mat(cov_mat).I
	obs = ps_CFHT
	
	

	spline_interps = list()
	for ibin in range(ps_avg.shape[-1]):
		ps_model = ps_avg[:,ibin]
		iinterp = interpolate.Rbf(m, w, s, ps_model)
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
	return interp_cosmo, cov_inv, ps_CFHT

def plot_heat_map_w (idx=idx_full,w=-1.02):
	print 'w=',w
	heatmap = zeros(shape=(l,ll))
	#obs = ps_CFHT
	#if w in w_arr[10:70]:#only calculate the region of interest
	interp_cosmo, cov_inv, ps_CFHT = return_interp_cosmo_for_idx (idx)	
	for i in range(l):
		for j in range(ll):
			best_fit = (om_arr[i], w, si8_arr[j])
			
			ps_interp = interp_cosmo(best_fit)	
			del_N = np.mat(ps_interp - ps_CFHT)
			chisq = float(del_N*cov_inv*del_N.T)
			heatmap[i,j] = chisq
	return heatmap

def chisq2P(chisq_mat):#(idx=idx_full,w=-1):#aixs 0-w, 1-om, 2-si8
	#chisq_mat = plot_heat_map_w (idx=idx_full,w=-1)
	P = exp(-(chisq_mat-amin(chisq_mat))/2)
	P /= sum(P)
	V = WLanalysis.findlevel(P)
	return P, V

idx_arr = [idx_full, idx_pk10ps, idx_pk18ps, idx_pk10, idx_pk18, idx_ps]
chisq_arr = map(plot_heat_map_w, idx_arr)
P_arr = map(chisq2P, chisq_arr)
labels = ['full', 'pk10ps', 'pk18ps', 'pk10', 'pk18', 'ps']
X, Y = np.meshgrid(om_arr, si8_arr)

from pylab import *
f = figure(figsize=(8,8))
ax=f.add_subplot(111)
lines=[]
colors = ('r','b','g','m','y','k')
for i in arange(len(P_arr)):
	P, V=P_arr[i]
	#V=findlevel(P)
	CS = ax.contour(X, Y, P.T, levels=[V[0],], origin='lower', extent=(om_arr[0], om_arr[-1], si8_arr[0], si8_arr[-1]), colors=colors[i])#, linewidths=lws2[i+1], linestyles=lss2[i+1])
	lines.append(CS.collections[0])
leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':20},loc=0)
ax.tick_params(labelsize=16)
ax.set_xlabel(r'$\rm{\Omega_m}$',fontsize=20)
ax.set_ylabel(r'$\rm{\sigma_8}$',fontsize=20)
leg.get_frame().set_visible(False)
#show()
savefig(test_dir+'plot/contour_test_w-1.02.jpg')
close()
######### chisq cube ###############
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
