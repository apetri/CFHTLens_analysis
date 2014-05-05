import WLanalysis
import os
import numpy as np
from scipy import *
import sys
from scipy.spatial import cKDTree
from emcee.utils import MPIPool

# calculate 2 point correlation function from CFHT data directly
# 2 ways of calculation:
# 1) on shear: yxewm, using kilbinger eq. 4, and compare to fig. 6
# 2) on covergence

# ---------------- 2PCF on shear --------------------------

twoPCF_dir = '/direct/astro+astronfs03/workarea/jia/CFHT/CFHT/twoPCF/'
full_dir = '/direct/astro+astronfs03/workarea/jia/CFHT/CFHT/full_subfields/'
yxewm = lambda i: (WLanalysis.readFits(full_dir+'yxewm_subfield%i_zcut0213.fit'%(i))).T
CFHT2pcf = genfromtxt(twoPCF_dir+'xi_pm+var.txt').T

#yxewm = lambda i: (WLanalysis.readFits('/Users/jia/Documents/weaklensing/CFHTLenS/2PCF/yxewm/yxewm_subfield%i_zcut0213.fit'%(i))).T
#CFHT2pcf = genfromtxt('/Users/jia/Documents/weaklensing/CFHTLenS/2PCF/CFHT2pcf').T

phi = lambda x0, y0, x1, y1: np.angle((x1-x0)+(y1-y0)*1.0j)

def et(e1, e2, phi0):
	e = e1+1.0j*e2
	ee = -e*exp(-2j*phi0)
	return ee # et = real(ee), ec = imag(ee)


bincenter = CFHT2pcf[0]
bins = bincenter # later will write to compute actual same bins as CFHT 
bins *= pi/10800.0 # convert from arcmin to radians, 1 arcmin = pi/10800 radians

#xiplus_arr = zeros(shape=(13,len(bins)))
#ximinus_arr= zeros(shape=(13,len(bins)))
#demoni_arr = zeros(shape=(13,len(bins)))

#for isf in range(14):
step = 100000 # step to 
def twoPCF(isf):
	#print 'subfield',isf
	fn = twoPCF_dir+'twoPCF_subfield%i'%(isf)
	if os.path.isfile(fn):
		return genfromtxt(fn)	
	else:
		xiplus_arr = zeros(len(bins))
		ximinus_arr= zeros(len(bins))
		norm_arr = zeros(len(bins))
		
		y, x, e1, e2, w, m = yxewm(isf)
		y -= amin(y)
		x -= amin(x)
		kdt = cKDTree(array([x,y]).T)
		for ibin in arange(len(bins)):
			pairs = array(list(kdt.query_pairs(bins[ibin]))).T	
			ipair = 0
			while ipair < len(pairs[0]):
				print 'subfield, bin, total gal pairs, ipair: ',isf, ibin, len(pairs[0]), ipair
				i, j = pairs[:,ipair:ipair+step]
				phi0 = phi(x[i], y[i], x[j], y[j])
				# these are complex ellipticities where e=et+iex
				ei = et(e1[i], e2[i], phi0)
				ej = et(e1[j], e2[j], phi0)
				
				xiplus_arr [ibin] += sum(w[i]*w[j] * real(ei * conj(ej)))
				ximinus_arr[ibin] += sum(w[i]*w[j] * real(ei * ej))
				norm_arr [ibin] += sum(w[i]*w[j]*(1+m[i])*(1+m[j]))
				ipair += step
			
		K = array([xiplus_arr, ximinus_arr, norm_arr])
		savetxt(fn,K)
		return K
	
	#xiplus_arr[1:] -= xiplus_arr[:-1]
	#ximinus_arr[1:] -= ximinus_arr[:-1]
	#norm_arr[1:] -= norm_arr[:-1]

		
pool = MPIPool()
pool.map(twoPCF, range(1,14))

print 'done-done-done: calculating 2pcf, now sum up all subfields'
# -------------------- grab everything ---------------------
bigmatrix = zeros(shape=(13, 3, len(bins)))

for isf in range(1,14):
	#fn = twoPCF_dir+'twoPCF_subfield%i'%(isf)
	#bigmatrix [isf-1] = genfromtxt(fn)
	bigmatrix [isf-1] = twoPCF(isf)
	
# cumulative sum to sum in each bin
bigmatrix[:, :, 1:] -= bigmatrix[:, :, :-1]
sum13fields = sum(bigmatrix, axis=0)
xip = sum13fields[0]/sum13fields[-1]
xim = sum13fields[1]/sum13fields[-1]
print 'done done done done!'