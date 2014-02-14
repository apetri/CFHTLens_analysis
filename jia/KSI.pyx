import numpy as np
cimport numpy as np
#from libc.math cimport cos, sin

def KSI_calc(smap, D0, mode="reflect"):
	"""Convolve smap with D, using schneider review eq 41, 45, where kmap each value is a complex #"""
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


def apMass_calc(smap1, smap2, Q0, phi0, mode="reflect"):
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