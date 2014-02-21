import numpy as np
cimport numpy as np
#from libc.math cimport cos, sin

def KSI_calc(smap, D0, mode="reflect"):
	'''Convolve smap with D, using schneider review eq 41, 45, where kmap each value is a complex #.'''
	ishape = smap.shape
	cdef int d0 = D0.shape[0]
	cdef float d = d0
	cdef int r = d0/2#D's width should be an odd number
	cdef np.ndarray[np.complex128_t, ndim=2] smap_cython = smap
	cdef np.ndarray[np.complex128_t, ndim=2] kmap = np.zeros(shape=ishape)+np.zeros(shape=ishape)*1.0j
	cdef np.ndarray[np.complex128_t, ndim=2] D = D0
	cdef np.ndarray[np.complex128_t, ndim=2] ismap
	cdef int smapx, smapy, x, y
	cdef double n
	cdef double complex kappa
	
	cdef np.ndarray[np.complex128_t, ndim=2] bigsmap = np.pad(smap_cython,(r,r),mode=mode)
	smapx = ishape[0]
	smapy = ishape[1]
	
	for x from r <= x < smapx+r:
		for y from r <= y < smapy+r:
			ismap = bigsmap[x-r:x+r+1,y-r:y+r+1]
			kappa = (np.real(D*ismap)).sum()
			n=len(ismap.nonzero()[0])/d**2
			if n > 0:
				kappa/=n*3.141592653589793#np.math.pi
			kmap[x-r,y-r]=kappa
	return kmap


def apMass_calc(smap1, smap2, Q0, phi0, mode="constant"):
	'''Aperture mass using Bard 2013 eq 1.'''
	ishape = smap1.shape
	print ishape
	d = Q0.shape[0]
	r = d/2
	cdef np.ndarray[np.float64_t, ndim=2] smap1_cython = smap1
	cdef np.ndarray[np.float64_t, ndim=2] smap2_cython = smap2
	cdef np.ndarray[np.float64_t, ndim=2] Q = Q0
	cdef np.ndarray[np.float64_t, ndim=2] phi = phi0
	cdef np.ndarray[np.float64_t, ndim=2] bigsmap1 = np.pad(smap1_cython,(r,r),mode=mode)
	cdef np.ndarray[np.float64_t, ndim=2] bigsmap2 = np.pad(smap2_cython,(r,r),mode=mode)
	cdef np.ndarray[np.float64_t, ndim=2] ismap1, ismap2, apM
	cdef np.ndarray[np.float64_t, ndim=2] kmap = np.zeros(shape=ishape)
	cdef int smapx, smapy, x, y
	cdef double Ng, apM_sum
	
	smapx = ishape[0]
	smapy = ishape[1]
	for x from r <= x < smapx+r:
		for y from r <= y < smapy+r:
			ismap1 = bigsmap1[x-r:x+r+1,y-r:y+r+1]
			ismap2 = bigsmap2[x-r:x+r+1,y-r:y+r+1]
			apM = -Q*(ismap1*np.cos(2*phi)+ismap2*np.sin(2*phi))
			apM_sum = apM.sum()
			Ng = len(ismap1.nonzero()[0])# #gal in aperture
			if Ng == 0:
				apM_sum = 0
			else:
				apM_sum /= Ng
			kmap[x-r,y-r]=apM_sum
	return kmap

def apMass_norm(weight, Q0, mode="constant"):
	'''normalization denominator for aperture mass calculation.'''
	ishape = weight.shape
	print ishape
	d = Q0.shape[0]
	r = d/2
	cdef np.ndarray[np.float64_t, ndim=2] w_cython = weight
	cdef np.ndarray[np.float64_t, ndim=2] Q = Q0
	cdef np.ndarray[np.float64_t, ndim=2] bigsmap1 = np.pad(w_cython,(r,r),mode=mode)
	cdef np.ndarray[np.float64_t, ndim=2] apM
	cdef np.ndarray[np.float64_t, ndim=2] kmap = np.zeros(shape=ishape)
	cdef int smapx, smapy, x, y
	cdef double Ng, apM_sum
	
	smapx = ishape[0]
	smapy = ishape[1]
	for x from r <= x < smapx+r:
		for y from r <= y < smapy+r:
			ismap1 = bigsmap1[x-r:x+r+1,y-r:y+r+1]
			apM = Q*ismap1
			apM_sum = apM.sum()
			Ng = len(ismap1.nonzero()[0])# #gal in aperture
			if Ng == 0:
				apM_sum = 0
			else:
				apM_sum /= Ng
			kmap[x-r,y-r]=apM_sum
	return kmap

def findpeak_list (kmap):
	"""Find peaks, input a ndarray of 2 dimensions.
	Return a list of peaks.
	"""
	cdef np.ndarray[np.float64_t, ndim=2] kmap_cython = kmap
	cdef int x, y, i, j, kmax, kmapy
	cdef int pflag = 0
	cdef list peaks = []
#	cdef np.ndarray[np.float64_t, ndim=1] peaks = np.ndarray(shape=(1), dtype=np.float)
	kmapx = kmap.shape[0] - 1
	kmapy = kmap.shape[1] - 1
	for x from 1 <= x < kmapx:
		for y from 1 <= y < kmapy:
			pflag = 1
			for i from x-1 <= i < x+2:
				if pflag == 1:
					for j from y-1 <= j < y+2:
						if kmap_cython[i,j] > kmap_cython [x, y]:
							pflag = 0
							break
				else:
					break
			if pflag == 1:
				peaks.append(kmap_cython [x, y])
#				peaks = np.append(peaks, kmap_cython [x, y])
	return peaks

def findpeak_mat (kmap):
	"""Find peaks, input a ndarray of 2 dimensions
	Return:
	1) The map with peak only and 0 for rest of entry
	2) The map with peak location set to 1, rest 0
	3) The array of peaks only
	"""
	cdef np.ndarray[np.float64_t, ndim=2] kmap_cython = kmap
	cdef int x, y, i, j, kmax, kmapy
	cdef int pflag = 0
	cdef np.ndarray[np.float64_t, ndim=2] peakmap = np.nan*np.zeros(shape=kmap.shape)

	kmapx = kmap.shape[0] - 1
	kmapy = kmap.shape[1] - 1
	for x from 1 <= x < kmapx:
		for y from 1 <= y < kmapy:
			pflag = 1
			for i from x-1 <= i < x+2:
				if pflag == 1:
					for j from y-1 <= j < y+2:
						if kmap_cython[i,j] > kmap_cython [x, y]:
							pflag = 0
							break
				else:
					break
			if pflag == 1:
				peakmap [x,y] = kmap_cython [x, y]
	return peakmap