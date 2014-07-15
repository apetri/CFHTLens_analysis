##########################################################
### This code is for Jia's thesis project B. It does the following:
### 1) organizes the CFHT catalogue to 4 Wx fields, with proper 
### format that's easy to use in the future
### 2) pick out a random redshift and 2 peak redshift
### 3) converts RA DEC to (x, y) radian, using Genomonic projection,
### centered at map center
### 4) final products: convergence maps and galcount maps split in 
### different redshift bins

import numpy as np
from scipy import *
from pylab import *
import os
#import astropy.io.fits as pyfits
import WLanalysis

split_dir = '/home1/02977/jialiu/CFHT_cat/split/'
W_dir = lambda Wx: '/home1/02977/jialiu/CFHT_cat/%s/'%(Wx) #dir for W1..W4 field
splitfiles = os.listdir(split_dir)
RA1 =(30.0, 39.0)#starting RA for W1
DEC1=(-11.5,-3.5)
RA2 =(132.0, 137.0)
DEC2=(-6.0,-0.5)
RA3 =(208.0, 221.0)
DEC3=(51.0, 58.0)
RA4 =(329.5, 336.0)
DEC4=(-1.2, 5.0)
RAs=(RA1,RA2,RA3,RA4)
DECs=(DEC1,DEC2,DEC3,DEC4)

dpp=0.0016914558667664816#degrees per pixel = sqrt(12)/2048
xnum = lambda RA: round((amax(RA)-amin(RA))/dpp+1)
ynum = lambda DEC:round((amax(DEC)-amin(DEC))/dpp+1)

z_arr = arange(0.025,3.5,.05)
idx2z = lambda idx:z_arr[idx]
field2int = lambda str: int(str[1])

#DrawRedshifts = lambda iPz: concatenate([[z_arr[argmax(iPz)],], WLanalysis.DrawFromPDF(z_arr, iPz, 2)])
DrawRedshifts = lambda iPz: WLanalysis.DrawFromPDF(z_arr, iPz, 1)

def list2coords(radeclist): 
	'''Input: radeclist = (Wfield, ra, dec), a 3xN matrix, (ra, dec) in degrees
	Return: (subfield, x, y), a 3xN matrix, (x, y) in radians
	'''
	xylist = zeros(shape = radeclist.shape)
	j = 0 #subfield count
	for i in range(4): #W field count
		idx = where(radeclist[:,0] == i+1)[0] #find W1 enries
		if len(idx) > 0:
			print 'Found entry for W',i+1
			sublist = radeclist[idx] #pick out W1 entries
			print 'idx',idx
			sort_subf = sort_subfs[i] #get W1 configurations
			center = centers[i] #prepare for x,y calc
			f_Wx = gnom_fun(center)
			xy = degrees(array(map(f_Wx,sublist[:,1:3])))
			print 'xy',xy
			for isort_subf in sort_subf:
				x0,x1,y0,y1 = isort_subf[5:9]
				# find entries for each subfield
				iidx = where((xy[:,0]<x1)&(xy[:,0]>x0)&(xy[:,1]<y1)&(xy[:,1]>y0))[0]
				
				if len(iidx) > 0:
					print 'iidx j=',j,iidx
					isublist = sublist[iidx]#subfield
					icenter = isort_subf[-2:]#center for subfield
					f_sub = gnom_fun(icenter)
					xy_sub = array(map(f_sub,isublist[:,1:3]))
					xylist[idx[iidx],0] = j2s(j)+1
					xylist[idx[iidx],1:] = xy_sub					
					if j in (4,9): # needs to turn 90 degrees, counterclock
						iy = degrees(xy_sub.T[0])
						ix =- degrees(xy_sub.T[1])
					else:
						ix = degrees(xy_sub.T[0])
						iy = degrees(xy_sub.T[1])
				j+=1
		else:
			j+=len(sort_subfs[i])
	return xylist

def OrganizeSplitFile(ifile):#file2coord(ifile):
	'''read in one of the split file, pick out the redshift, and sort by fields, e2 is c2 correted, with e2-=c2'''	
	
	field = genfromtxt(split_dir+ifile,usecols=0,dtype=str)
	field = array(map(field2int,field))
	print 'field',field.shape,field

	# generate 2 random redshift and 1 peak
	Pz = genfromtxt(split_dir+ifile,usecols=arange(14,84),dtype=str)
	Pz = (np.core.defchararray.replace(Pz,',','')).astype(float)
	
	seed(99)
	z_rand1 = array(map(DrawRedshifts,Pz))
	seed(88)
	z_rand2 = array(map(DrawRedshifts,Pz))
	z_peak = z_arr[argmax(Pz,axis=1)]
	#Prand = concatenate([z_peak.reshape(-1,1), z_rand],axis=1)
	#print 'Prand',Prand.shape,datas

	sheardata = genfromtxt(split_dir+ifile,usecols=[1,2,5,6,7,8,9,10,11,12,13,84])
	##skip_header=1
	ra, dec, e1, e2, w, fitclass, r, snr, mask, m, c2, mag= sheardata.T
	e2 -= c2
	
	radeclist = concatenate((field.reshape(-1,1),sheardata[[0,1]]),axis=1)
	xylist = list2coords(radeclist)
		
	i=0
	for Wx in range(1,5):
		idx=where((field==Wx)&(mask<=1.0)&(fitclass==0)&(z>=0.2)&(z<=1.3))
		print ifile, Wx, len(idx[0])/50000.0
		if len(idx[0]) > 0:
			#data = (np.array([ra,dec,e1,e2,w,r,snr,m,c2,mag]).T)[idx]
			data = (np.array([ra,dec,e1,e2,w,r,snr,m,c2,mag,z]).T)[idx]
			WLanalysis.writeFits(data, W_dir(Wx)+ifile+'.fit')#,fmt=['%i','%i','%s','%s','%s','%.3f'])
		i+=1