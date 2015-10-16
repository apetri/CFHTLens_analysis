#!python
# Jia Liu (07/01/2015)
# To Alvaro for the multiplicative bias project (Summer 2015)
# This code does:
# (1) coordinate transform from [RA, DEC] to [x, y]
# (2) take in coordinate of galaxies, put on a grid (number counts)
# (3) calculate cross correlation of 2 maps

import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack
from scipy.fftpack import fftfreq, fftshift
import os
import scipy.ndimage as snd
#import astropy.io.fits as pyfits # for fits file reading

##################### constants #################
PPR=8468.416479647716 #pixels per radians
PPA=2.4633625 #pixels per arcmin, PPR/degrees(1)/60
APP=0.40594918531072871 #arcmin per pix, degrees(1/PPR)*60
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
sizes = (1330, 800, 1120, 950)

#################### functions ##################
rad2pix=lambda x, size: around(size/2.0-0.5 + x*PPR).astype(int) # from radian to pixels

def gnom_fun(center):
	'''Create a function that calculate the location of 
	a sky position (ra, dec) on a grid, centered at (ra0, dec0), 
	using Gnomonic projection (flat sky approximation). 
	Input:
	center = (ra0,dec0) in degrees (note the format is a list of 2 elements, not 2 numbers).
	Output:
	A function that transform (ra, dec) to (x, y), centered at (ra0, dec0), in unite of radians.
	Example:
	>> center = (45,60)
	>> f = gnom_fun(center)
	>> f((47,60))
	(-0.017452406234833018, 0.00026381981635707768) -> this is the x, y position in radians for (ra, dec) = (47, 60), centered at (ra0, dec0) =  (45,60)
	Ref: http://mathworld.wolfram.com/GnomonicProjection.html
	'''
	ra0, dec0 = center
	ra0 -= 180 #convert from ra to longitude
	ra0 = ra0*pi/180
	dec0 = dec0*pi/180
	def gnom(radec):
		if len(radec)>2:
			ra, dec = (array(radec).copy()).T
		else:
			ra,dec = radec[0].copy(), radec[1].copy()
		ra -= 180
		ra = ra*pi/180
		dec = dec*pi/180
		### the angular separation between (ra,dec) and (ra0,dec0)
		cosc = sin(dec0)*sin(dec)+cos(dec0)*cos(dec)*cos(ra-ra0)
		x=cos(dec)*sin(ra0-ra)/cosc
		y=(cos(dec0)*sin(dec)-sin(dec0)*cos(dec)*cos(ra-ra0))/cosc
		return x, y
	return gnom

def coords2grid_counts(radeclist, Wx):
	'''returns a grid map of galaxy counts.	
	Input:
	radeclist = (n x 2) matrix, for n galaxies, each row is (ra, dec)
	Wx = 1, 2, 3, or 4
	Pixel resolution is ~2.5 pixels per arcmin
	
	Output:
	a grid of galaxy counts (within each pixel)
	'''
	####### first: translate (ra, dec) to (x, y) in unit of pixels, for Wx field
	size=sizes[Wx-1]
	center = centers[Wx-1]
	f_Wx = gnom_fun(center)
	xy = array(map(f_Wx, radeclist))
	xy_pix = rad2pix(xy, size)
	x, y = array(xy_pix).T
	
	####### second: put galaxies to grid, note some pixels may have multiple galaxies
	grid_counts= zeros(shape=(size,size),dtype=int)
	xy = x+y*1j #so to sort xy as one array
	sorted_idx = argsort(xy) #get the index that gives a sorted array for xy
	ar = xy[sorted_idx] #sorted xy
	left_idx = arange(len(ar)) #left over idx that are used to put data into grid
	ar0 = ar.copy()

	j=0
	while len(left_idx) > 0: # len(left_idx) = #gals to be put into the grid
		a, b=unique(ar0, return_index=True)
		# a is the unique pixels
		# b is the index of those unique pixels		
		if j>0:
			b=b[1:]# b[0]=-1 from j=1		
		#put unique values into matrix
		ix = sorted_idx[b]
		grid_counts[x[ix],y[ix]]+= 1
		left_idx=setdiff1d(left_idx, b)# Return the sorted, unique values in 'left_idx' that are not in 'b'.
		ar0[b] = -1 # a number that's smaller than other indices (0..)
		j += 1 #j is the repeating #gal count, inside one pix
	return grid_counts