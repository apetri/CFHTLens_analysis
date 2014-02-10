# Jia Liu 2014/2/7 
# functions and routines used in my weak lensing analysis

import numpy as np
from scipy import *
from scipy import interpolate, stats
from pylab import *
import os
import scipy.ndimage as snd
import matplotlib.pyplot as plt
######### uncomment one of the following 2 lines, depends on astropy or pyfits is installed ###
import astropy.io.fits as pyfits
#import pyfits

###################################
#### CFHT list of redshifts
CFHTz = arange(0.025,3.5,.05)
idx2CFHTz = lambda idx:CFHTz[idx]

#### redshifts in our simulation ###
SIMz = array([ 0.00962476,  0.02898505,  0.04849778,  0.0681688 ,  0.08800411,
        0.10800984,  0.12819235,  0.14855811,  0.16911381,  0.1898663 ,
        0.21082264,  0.23199007,  0.25337603,  0.27498818,  0.29683438,
        0.31892272,  0.34126152,  0.36385934,  0.38672498,  0.4098675 ,
        0.43329621,  0.4570207 ,  0.48105085,  0.50539682,  0.53006909,
        0.55507842,  0.58043594,  0.60615309,  0.63224168,  0.65871385,
        0.68558217,  0.71285956,  0.74055938,  0.76869539,  0.79728181,
        0.82633333,  0.85586509,  0.88589277,  0.91643252,  0.94750108,
        0.97911572,  1.01129433,  1.04405538,  1.07741802,  1.11140204,
        1.14602793,  1.18131693,  1.21729101,  1.25397297,  1.29138642,
        1.32955585,  1.36850665,  1.40826517,  1.44885876,  1.49031581,
        1.5326658 ,  1.57593937,  1.62016833,  1.6653858 ,  1.71162617,
        1.75892526,  1.80732031,  1.85685013,  1.9075551 ,  1.95947731,
        2.01266062,  2.06715078])
idx2SIMz = lambda idx:SIMz[idx]

######### functions ############
def gnom_fun(center):
	'''Create a function that calculate the location of a sky position (ra, dec) on a grid, centered at (ra0, dec0), using Gnomonic projection (flat sky approximation). 
	Input:
	center = (ra0,dec0) in degrees (note the format is a list of 2 elements, not 2 numbers).
	Output:
	A function that transform (ra, dec) to (x, y), centered at (ra0, dec0).
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
		ra,dec = radec
		ra -= 180
		ra = ra*pi/180
		dec = dec*pi/180
		### the angular separation between (ra,dec) and (ra0,dec0)
		cosc = sin(dec0)*sin(dec)+cos(dec0)*cos(dec)*cos(ra-ra0)
		x=cos(dec)*sin(ra0-ra)/cosc
		y=(cos(dec0)*sin(dec)-sin(dec0)*cos(dec)*cos(ra-ra0))/cosc
		return x, y
	return gnom

def gnom_inv(xy,center):
	'''Inverse of gnom_fun(center)(xy), transfrom (x, y) to (ra, dec) centered at (ra0, dec0)
	Input:
	xy = (x, y), the position on the grid, in unit of radians.
	center = (ra0, dec0), the coordinate at the center of the grid, in degrees.
	Output:
	ra, dec, in degrees.
	Example (test validity - pass):
	>> xy = (-0.017452406234833018, 0.00026381981635707768) # = gnom_fun(center)((47,60))
	>> center = (45,60)
	>> gnom_inv(xy,center)
	(47.0, 59.999999999999993)
	'''
	x, y = xy
	ra0, dec0 = center
	ra0 -= 180
	ra0 = ra0*pi/180
	dec0 = dec0*pi/180
	rho = sqrt(x**2+y**2)
	c = arctan(rho)
	dec = arcsin(cos(c)*sin(dec0)+y*sin(c)*cos(dec0)/rho)
	ra = ra0-arctan(x*sin(c)/(rho*cos(dec0)*cos(c)-y*sin(dec0)*sin(c)))
	return degrees(ra)+180, degrees(dec)

def readFits (fitsfile):
	'''Input: 
	fitsfile = file name of the fitsfile
	Output:
	data array for the fitsfile.
	'''
	hdulist = pyfits.open(fitsfile)
	data = np.array(hdulist[0].data)
	return data

def writeFits (data, filename):
	'''Input:
	data = to be written
	filename = the file name of the fitsfile, note this needs to be the full path, otherwise will write to current directory.
	'''
	hdu = pyfits.PrimaryHDU(data)
	hdu.writeto(filename)

def ShrinkMatrix (matrix, ratio, avg=False):
	'''Shrink the matrix by ratio, if shape/ratio is not integer, then pad some 0s at the end to make newshape/ratio devisible. So if accurate shrinking is needed, the last row and column should be discarded.
	Input:
	matrix: the original matrix which we want to shrink its size.
	ratio: by how much we want to shrink it.
	avg: if True, the resulting matrix will be the average of the merged cells. By default, it's the sum.
	'''
	ylen, xlen=matrix.shape
	ynew = ylen/ratio
	xnew = xlen/ratio
	if xlen%ratio !=0:
		xnew = xlen/ratio+1
		matrix = np.pad(matrix,((0,0),(0,xnew*ratio-xlen)),mode='constant')
	if ylen%ratio !=0:
		ynew = ylen/ratio+1
		matrix = np.pad(matrix,((0,ynew*ratio-ylen),(0,0)),mode='constant')		
	if avg:
		matrix_new = matrix.reshape(ynew,ratio,xnew,ratio).mean(-1).mean(1)
	else:
		matrix_new = matrix.reshape(ynew,ratio,xnew,ratio).sum(-1).sum(1)
	return matrix2

def RebinMatrix (matrix, shape, avg=False):
	'''bin a 2D matrix to a new shape, shape dimensions must be devisible by matrix dimensions.
	'''
	sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
	#new rows, # of rows to sum, new cols, # of cols to sum
	if avg:
		matrix_new = matrix.reshape(sh).mean(-1).mean(1)
	else:
		matrix_new = matrix.reshape(sh).sum(-1).sum(1)	
	return matrix_new

def InterpPDF (x1, P, x2, edges=None):
	'''Interpolate from discrete PDF with bin centers at x1, to a new PDF centered at x2[1:-1]. x1 and PDF must have same dimensions. x1 must be equally spaced, but x2 can be uneven. Note P.size = x2.size - 2!
	Test (resulting plot: http://goo.gl/Pw5QeW):
	>> x1 = arange(0.025,3.5,.05)
	>> P = array([3.32000000e-09, 8.81000000e-10, 7.06000000e-10, 4.09000000e-09,
	2.49000000e-08, 1.25000000e-07, 1.78000000e-07, 4.02000000e-07, 1.39000000e-06,
	3.57000000e-06, 1.28000000e-05, 1.04000000e-04, 8.66000000e-04, 6.97000000e-03,
	2.31000000e-02, 4.37000000e-02, 6.70000000e-02, 8.60000000e-02, 1.05000000e-01,
	1.12000000e-01, 1.00000000e-01, 8.15000000e-02, 6.64000000e-02, 5.65000000e-02,
	4.82000000e-02, 4.48000000e-02, 4.26000000e-02, 3.77000000e-02, 2.86000000e-02,
	1.90000000e-02, 1.16000000e-02, 6.92000000e-03, 4.15000000e-03, 2.56000000e-03, 
	1.66000000e-03, 1.10000000e-03, 7.00000000e-04, 4.14000000e-04, 2.21000000e-04, 
	1.04000000e-04, 4.17000000e-05, 1.47000000e-05, 4.55000000e-06, 1.25000000e-06, 
	3.30000000e-07, 8.57000000e-08, 2.18000000e-08, 5.73000000e-09, 2.00000000e-09, 
	9.00000000e-10, 4.41000000e-10, 2.18000000e-10, 1.10000000e-10, 5.34000000e-11, 
	1.99000000e-11, 5.37000000e-12, 1.31000000e-12, 2.52000000e-13, 4.08000000e-14, 
	6.92000000e-15, 9.91000000e-16, 1.86000000e-16, 3.01000000e-17, 2.69000000e-18, 
	1.01000000e-19, 1.74000000e-21, 3.71000000e-23, 4.94000000e-25, 6.25000000e-27, 
	5.51000000e-29])
	>> x2 = sort(rand(50000))*2
	>> newP = InterpPDF(x1,P,x2)
	>> custm = stats.rv_discrete(name='custm', values=(arange(len(x2)-2), newP))
	>> R = custm.rvs(size=100000) 
	>> redshift = lambda R: x2[1:-1][R]
	>> R2 = redshift(R)
	>> c1,c2=histogram(R2, bins=arange(0.025-0.025,3.5+.025,0.05))
	>> plot (CFHTz, P,label='PDF',linewidth=2,drawstyle='steps-mid')
	>> plot (CFHTz, c1.astype(float)/len(R), label='random gen',linewidth=2,drawstyle='steps-mid')
	>> legend()
	>> show()
         '''
	P /= sum(P) #normalize PDF in case its sum is not 1.
	f = interpolate.InterpolatedUnivariateSpline(x1, P) # interpolate the PDF with a 3rd order spline function
	fint = lambda edge: f.integral(edge[0],edge[1]) # integrate the region between bin left and right
	if not bool(edges.any()): # find edges for each bin, if not already provided
		step = (x2[1:]-x2[:-1])/2
		binwidth = (step[1:]+step[:-1])/2
		midedges = array([x2[1:-1]-binwidth,x2[1:-1]+binwidth]).T
		leftedge = [2*x2[0]-midedges[0,0],midedges[0,0]]
		rightedge= [midedges[-1,-1], 2*x2[-1]-midedges[-1,-1]]
		edges = concatenate(([leftedge],midedges,[rightedge]))
	newP = array(map(fint,edges)).T
	newP /= sum(newP)
	return newP

def DrawFromPDF (x, P, n):
	'''
	Given a discrete PDF (x, P), draw n random numbers out from x.
	Example:
	>> DrawFromPDF(CFHTz,P,3)
	array([ 1.025,  1.225,  0.725])
	'''
	P /= sum(P) # normalize P
	custm = stats.rv_discrete(name='custm', values=(arange(len(x)), P))
	R = custm.rvs(size=n) 
	rndx = lambda R: x[R]
	return rndx(R)

