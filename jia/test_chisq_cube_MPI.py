# Jia Liu 11/14/2014
# after found bug in power spectrum computation, this code is to 
# compute chisq cube
# run with:
# ibrun python test_chisq_cube_MPI.py ALL 100
# ibrun python test_chisq_cube_MPI.py BAD 27

import numpy as np
from scipy import *
from scipy import interpolate
import os
import WLanalysis
from emcee.utils import MPIPool
import sys

#ell_7000 = ell_arr[:27]
BG = str(sys.argv[1])
cut7000 = int(sys.argv[2])
test_dir = '/home1/02977/jialiu/ps_chisq_compute/'
#test_dir = '/Users/jia/Documents/weaklensing/CFHTLenS/emulator/test_ps_bug/'
cosmo_params = genfromtxt(test_dir+'cosmo_params.txt')
m, w, s = cosmo_params.T

l,ll = 100,102
om_arr = linspace(0,1.2,l)
si8_arr = linspace(0,1.6,ll)
#w_arr = linspace(0,-3,101)
w_arr = linspace(0,-3,3)
	
ps_CFHT = np.load(test_dir+'%s_ps_CFHT.npy'%(BG))[:cut7000]
ps_fidu_mat = np.load(test_dir+'%s_ps_fidu39.npy'%(BG))[:,:cut7000]
ps_avg = np.load(test_dir+'%s_avg_ps.npy'%(BG))
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

def plot_heat_map_w (w):
	print 'w=',w
	heatmap = zeros(shape=(l,ll))
	for i in range(l):
		for j in range(ll):
			best_fit = (om_arr[i], w, si8_arr[j])
			ps_interp = interp_cosmo(best_fit)	
			del_N = np.mat(ps_interp - obs)
			chisq = float(del_N*cov_inv*del_N.T)
			heatmap[i,j] = chisq
	return heatmap
pool = MPIPool()
chisq_cube = pool.map(plot_heat_map_w, w_arr)
if cut7000 <39:
	np.save(test_dir+'%s_chisqcube_ps_ell7000'%(BG), array(chisq_cube).reshape(-1))
else:
	np.save(test_dir+'%s_chisqcube_ps'%(BG), array(chisq_cube).reshape(-1))

print 'done'
