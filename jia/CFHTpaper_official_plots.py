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

contour_peaks_smoothing = 1
contour_ps_fieldselect = 1
contour_peaks_powspec = 1
contour_including_w = 0

interp_2D_plane = 0
good_bad_peaks = 0
m_correction = 0
sample_interpolation = 0#remember need to delete cosmo #48 to work
sample_points = 0#for final fits wiht 3 random points
good_bad_powspec = 0
include_w = 0
SIGMA_contour = 0

########### constants ####################
l=100
ll=102
om0,om1 = 0, 67
si80,si81 = 20,85
w0,w1 = 10,70
om_arr = linspace(0,1.2,l)[om0:om1]
si8_arr = linspace(0,1.6,ll)[si80:si81]
w_arr = linspace(0,-3,101)[w0:w1]
sigmaG_arr = [1.0, 1.8, 3.5, 5.3]
fn_arr = ['idx_psPass7000_pk2smoothing', 'pk2moothing', 'pk10', 'pk18', 'pk35', 'pk53', 'psPass', 'psAll', 'psPass7000', 'psAll7000']

cube_arr = [load(test_dir+'chisqcube_%s.npy'%(fn)) for fn in fn_arr]

colors=('r','b','m','c','k','g')#,'r','c','b','g','m','k')
lss =('solid', 'dashed', 'solid', 'dashed', 'dashdot', 'dotted')#,'dashdot', 'dotted')
lss3=('-','-.','--',':','.')
lws = (4,4,2,2,4,4,2,2)

############################################

def cube2P(chisq_cube, axis=0):#aixs 0-w, 1-om, 2-si8
	if axis==0:
		chisq_cube = chisq_cube[w0:w1,om0:om1,si80:si81]
	else:
		chisq_cube = chisq_cube[:,om0:om1,si80:si81]
	P = sum(exp(-chisq_cube/2),axis=axis)
	P /= sum(P)
	return P

extents = (om_arr[0], om_arr[-1], si8_arr[0], si8_arr[-1])
def official_contour (cube_arr, labels, nlev, fn):
	f = figure(figsize=(8,8))
	ax=f.add_subplot(111)
	lines=[]
	X, Y = np.meshgrid(om_arr, si8_arr)
	
	for i in arange(len(cube_arr)):
		P=cube2P(cube)
		V=findlevel(P)
		CS = ax.contour(X, Y, P.T, levels=[V[0],], origin='lower', extent=extents, colors=colors[i], linewidths=lws2[i+1], linestyles=lss2[i+1])
		lines.append(CS.collections[0])
		if nlev == 2:
			CS2 = ax.contour(X, Y, P.T, levels=[V[1],], alpha=0.7, origin='lower', extent=extents, colors=colors[i], linewidths=lws[i], linestyles=lss[i])

	leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':20},loc=0)
	ax.tick_params(labelsize=16)
	ax.set_xlabel(r'$\rm{\Omega_m}$',fontsize=20)
	ax.set_ylabel(r'$\rm{\sigma_8}$',fontsize=20)
	leg.get_frame().set_visible(False)
	savefig(plot_dir+fn+'.pdf')
	close()

if contour_peaks_smoothing:
	labels = [r'$\rm{%.1f, arcmin}$'%(sigmaG) for sigmaG in sigmaG_arr2]
	labels.append(r'$\rm{1.0+1.8\, arcmin}$')
	icube_arr = cube_arr[[2,3,4,5,1]]
	fn = 'contour_peaks_smoothing'
	nlev = 1
	official_contour (cube_arr, labels, nlev, fn)
	
if contour_ps_fieldselect:
	labels = [r'$\rm{pass\, fields}$', r'$\rm{all\, fields}$', r'$\rm{pass\, fields(\ell<7,000)}$', r'$\rm{all\, fields(\ell<7,000)}$']
	icube_arr = cube_arr[[6, 7, 8, 9]]
	fn = 'contour_ps_fieldselect'
	nlev = 1
	official_contour (cube_arr, labels, nlev, fn)
	
if contour_peaks_powspec:
	labels = [r'$\rm{power\, spectrum}$', r'$\rm{peaks\, (1.0 + 1.8\,arcmin)}$', r'$\rm{power\, spectrum + peaks}$']
	icube_arr = cube_arr[[8, 1, 0]]
	fn = 'contour_peaks_powspec'
	nlev = 2
	official_contour (cube_arr, labels, nlev, fn)