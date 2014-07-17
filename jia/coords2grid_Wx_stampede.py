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
import WLanalysis
from emcee.utils import MPIPool

cat_dir='/home1/02977/jialiu/CFHT_cat/'
#cat_dir = '/Users/jia/CFHTLenS/catalogue/'
split_dir = cat_dir+'split/'
W_dir = lambda Wx: cat_dir+'W%s/'%(Wx) #dir for W1..W4 field
splitfiles = os.listdir(split_dir)

centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])

############################################################
########## calculate map size ##############################
#RA1 =(30.0, 39.0)#starting RA for W1
#DEC1=(-11.5,-3.5)
#RA2 =(132.0, 137.0)
#DEC2=(-6.0,-0.5)
#RA3 =(208.0, 221.0)
#DEC3=(51.0, 58.0)
#RA4 =(329.5, 336.0)
#DEC4=(-1.2, 5.0)
#RAs=(RA1,RA2,RA3,RA4)
#DECs=(DEC1,DEC2,DEC3,DEC4)
###dpp=0.0016914558667664816#degrees per pixel = sqrt(12)/2048
#dpp = 0.0067658234670659265#degrees per pixel = sqrt(12)/512
#xnum = lambda RA: round((amax(RA)-amin(RA))/dpp+1)
#ynum = lambda DEC:round((amax(DEC)-amin(DEC))/dpp+1)
##sized calculated using above 3 lines:
## W1 1331, W2 814, W3 1922, W4 962
sizes = (1331, 814, 1922, 962)
############################################################

z_arr = arange(0.025,3.5,.05)
idx2z = lambda idx:z_arr[idx]
field2int = lambda str: int(str[1])

#DrawRedshifts = lambda iPz: concatenate([[z_arr[argmax(iPz)],], WLanalysis.DrawFromPDF(z_arr, iPz, 2)])
DrawRedshifts = lambda iPz: WLanalysis.DrawFromPDF(z_arr, iPz, 1)

def list2coords(radeclist, Wx): 
	'''Input: radeclist = (Wfield, ra, dec), a 3xN matrix, (ra, dec) in degrees
	Return: (subfield, x, y), a 3xN matrix, (x, y) in radians
	'''
	xy = zeros(shape = radeclist.shape) #create xy list
	center = centers[Wx-1] #find the center for Wx field
	f_Wx = WLanalysis.gnom_fun(center)
	xy = degrees(array(map(f_Wx,radeclist)))
	return xy

def OrganizeSplitFile(ifile):
	'''read in one of the split file, pick out the redshift, and sort by fields, e2 is c2 correted, with e2-=c2'''	
	
	field = genfromtxt(split_dir+ifile,usecols=0,dtype=str)
	field = array(map(field2int,field))
	print ifile

	# generate 2 random redshift and 1 peak
	Pz = genfromtxt(split_dir+ifile,usecols=arange(14,84),dtype=str)
	Pz = (np.core.defchararray.replace(Pz,',','')).astype(float)
	
	seed(99)
	z_rand1 = array(map(DrawRedshifts,Pz)).ravel()
	seed(88)
	z_rand2 = array(map(DrawRedshifts,Pz)).ravel()
	z_peak = z_arr[argmax(Pz,axis=1)]
	z_all = concatenate([[z_peak,], [z_rand1,], [z_rand2,]]).T

	sheardata = genfromtxt(split_dir+ifile,usecols=[1,2,5,6,7,8,9,10,11,12,13,84])
	ra, dec, e1, e2, w, fitclass, r, snr, mask, m, c2, mag = sheardata.T
	e2 -= c2
	
	
	i=0
	for Wx in range(1,5):
		idx=where((field==Wx)&(mask<=1.0)&(fitclass==0)&(amin(z_all,axis=-1)>=0.2)&(amax(z_all,axis=-1)<=1.3))[0]
		print ifile, Wx, len(idx)/50000.0
		if len(idx) > 0:
			#data = (np.array([ra,dec,e1,e2,w,r,snr,m,c2,mag]).T)[idx]
			data = (np.array([ra,dec,e1,e2,w,r,snr,m,c2,mag,z_peak, z_rand1, z_rand2]).T)[idx]
			radeclist = sheardata[idx][:,[0,1]]
			xylist = list2coords(radeclist, Wx)
			xy_data = concatenate([xylist,data],axis=1)
			WLanalysis.writeFits(xy_data, W_dir(Wx)+ifile+'.fit')#,fmt=['%i','%i','%s','%s','%s','%.3f'])
		i+=1

#############################################
########## split file organizing ############
########## uncomment next 1 line ############
#pool.map(OrganizeSplitFile,splitfiles)
#############################################
zbins = array([0.4, 0.5, 0.6, 0.7, 0.85, 1.3])#arange(0.3,1.35,0.1)
def SumSplitFile2Grid(Wx):
	'''For Wx field, read in each split file, 
	and create e1, e2 grid for mass construction.
	Input: Wx=1,2,3,4
	Output: (Me1, Me2, Mw, galn) split in each redshift bins'''
	isize = sizes[Wx-1]
	ishape = (len(zbins), isize, isize)
	ishape_hi = (len(zbins)-1, isize, isize)#no need to do hi for zcut=1.3 since it's everything
	Me1_hi = zeros(shape=ishape_hi)
	Me2_hi = zeros(shape=ishape_hi)#hi is for higher redshift bins, lo is lower redshift
	Mw_hi = zeros(shape=ishape_hi)
	#Mk_hi = zeros(shape=ishape_hi)
	galn_hi = zeros(shape=ishape_hi)

	Me1_lo = zeros(shape=ishape)
	Me2_lo = zeros(shape=ishape)
	Mw_lo = zeros(shape=ishape)
	#Mk_lo = zeros(shape=ishape)
	galn_lo = zeros(shape=ishape)
	
	Wfiles = os.listdir(W_dir(Wx))#get the list of split file for Wx
	for iW in Wfiles:
		datas = WLanalysis.readFits(W_dir(Wx)+iW)
		#cols: x, y, ra, dec, e1, e2, w, r, snr, m, c2, mag, z_peak, z_rand1, z_rand2
		z = datas.T[-3]#z_peak, -2 is z_rand1, -1 is z_rand2
		i = 0 #zbin count
		for zcut in zbins:
			idx0 = where(z<zcut)[0]
			idx1 = where(z>=zcut)[0]
			for idx in [idx0,idx1]:
				y, x, e1, e2, w, m = (datas[idx].T)[[0,1,4,5,6,9]]#note x, y is reversed in python
				k = array([e1*w, e2*w, (1+m)*w])
				x = radians(x)
				y = radians(y)
				print 'W'+str(Wx), iW, 'coords2grid, zbin =',zbins[i]
				A, galn = WLanalysis.coords2grid(x, y, k, size=isize)
				if idx[0] == idx0[0]:
					Me1_lo[i] += A[0]
					Me2_lo[i] += A[1]
					Mw_lo[i] += A[2]
					galn_lo[i] += galn
				elif len(idx1)==0:#no need to calculate hi bin for zcut=1.3
					break
				else:
					Me1_hi[i] += A[0]
					Me2_hi[i] += A[1]
					Mw_hi[i] += A[2]
					galn_hi[i] += galn
			i+=1
	print 'Done collecting small fields for W'+str(Wx)
	
	for i in range(len(zbins)):
		for hl in ('lo','hi'):
			Me1_fn = cat_dir+'W%i_Me1w_%s_%s.fit'%(Wx, zbins[i],hl)
			Me2_fn = cat_dir+'W%i_Me2w_%s_%s.fit'%(Wx, zbins[i],hl)
			Mw_fn = cat_dir+'W%i_Mwm_%s_%s.fit'%(Wx, zbins[i],hl)
			galn_fn = cat_dir+'W%i_Me2w_%s_%s.fit'%(Wx, zbins[i],hl)
			if hl=='hi' and i==len(zbins)-1:
				continue
			elif hl=='lo':
				WLanalysis.writeFits(Me1_lo[i],Me1_fn)
				WLanalysis.writeFits(Me2_lo[i],Me2_fn)
				WLanalysis.writeFits(Mw_lo[i],Mw_fn)
				WLanalysis.writeFits(galn_lo[i],galn_fn)
			else:
				WLanalysis.writeFits(Me1_hi[i],Me1_fn)
				WLanalysis.writeFits(Me2_hi[i],Me2_fn)
				WLanalysis.writeFits(Mw_hi[i],Mw_fn)
				WLanalysis.writeFits(galn_hi[i],galn_fn)

print 'DONE-DONE-DONE'
pool = MPIPool()

