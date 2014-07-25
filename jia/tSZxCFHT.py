##########################################################
### This code is for cross correlate CFHT with Planck tSZ. 
### It does the following:
### 1) put tSZ to a grid same as CFHT size
### 2) cross correlate CFHT with tSZ

import numpy as np
from scipy import *
from pylab import *
import os
import WLanalysis
from scipy import interpolate

kSZ_dir = '/Users/jia/CFHTLenS/kSZ/'
kSZCoordsGen = lambda i: WLanalysis.readFits(kSZ_dir+'kSZ2_W%i.fit'%(i))
noiseCoordsGen = lambda i: WLanalysis.readFits(kSZ_dir+'kSZ2_noise_W%i.fit'%(i))

#kSZCoordsGen = lambda i: genfromtxt(kSZ_dir+'kSZ2_W%i.txt'%(i))
#txt2fits_kSZ = lambda Wx: WLanalysis.writeFits(kSZCoordsGen(Wx),kSZ_dir+'kSZ2_W%i.fit'%(Wx))
#txt2fits_noise = lambda Wx: WLanalysis.writeFits(noiseCoordsGen(Wx),kSZ_dir+'kSZ2_noise_W%i.fit'%(Wx))
#map(txt2fits,range(1,5))

kmapGen = lambda i: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/W%i_KS_1.3_lo_sigmaG05.fit'%(i))
galnGen = lambda i: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/W%i_galn_1.3_lo_sigmaG05.fit'%(i))
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
sizes = (1330, 800, 1120, 950)
PPR512=8468.416479647716
PPA512=2.4633625
edgesGen = lambda Wx: linspace(5,100,11)*sizes[Wx-1]/1330.0
rad2pix=lambda x, size: around(size/2.0-0.5 + x*PPR512).astype(int)
sigmaG_arr = [0.0, 1.0, 2.0, 5.0]

def list2coords(radeclist, Wx):
	size=sizes[Wx-1]
	xy = zeros(shape = radeclist.shape)
	center = centers[Wx-1]
	f_Wx = WLanalysis.gnom_fun(center)
	#xy = degrees(array(map(f_Wx,radeclist)))
	xy = array(map(f_Wx,radeclist))
	xy_pix = rad2pix(xy, size)
	return xy_pix

def interpGridpoints (xy, values, newxy, method='nearest'):
	newvalues = interpolate.griddata(xy, values, newxy, method=method)
	return newvalues

plot_dir = '/Users/jia/CFHTLenS/plot/tSZxCFHT/'
def plotimshow(img,ititle,vmin=None,vmax=None):		 
	 #if vmin == None and vmax == None:
	imgnonzero=img[nonzero(img)]
	if vmin == None:
		std0 = std(imgnonzero)
		x0 = median(imgnonzero)
		vmin = x0-3*std0
		vmax = x0+3*std0
	im=imshow(img,interpolation='nearest',origin='lower',aspect=1,vmin=vmin,vmax=vmax)
	colorbar()
	title(ititle,fontsize=16)
	savefig(plot_dir+'%s.jpg'%(ititle))
	close()	

def kSZmapGen (Wx, noise=False, method='nearest'):
	#print Wx
	size=sizes[Wx-1]
	if noise:
		
		kSZmap_fn = kSZ_dir+'kSZmap_noise_W%i_%s.fit'%(Wx,method)
	else:	
		kSZmap_fn = kSZ_dir+'kSZmap_W%i_%s.fit'%(Wx,method)
	isfile_kmap, kSZmap = WLanalysis.TestFitsComplete(kSZmap_fn, return_file = True)
	if isfile_kmap == False:
		if noise:
			kSZCoord = noiseCoordsGen(Wx)
		else:
			kSZCoord = kSZCoordsGen(Wx)
		radeclist = kSZCoord[:,:-1]
		values = kSZCoord.T[-1]
		xy = list2coords(radeclist,Wx)
		X,Y=meshgrid(range(size),range(size))
		X=X.ravel()
		Y=Y.ravel()
		newxy=array([X,Y]).T
		print 'newxy'
		newvalues = interpGridpoints (xy, values, newxy,method=method)
		print 'new values'
		kSZmap = zeros(shape=(size,size))
		kSZmap[Y,X]=newvalues
		WLanalysis.writeFits(kSZmap, kSZmap_fn)
		plotimshow(kSZmap, 'kSZmap_W%i_%s'%(Wx,method))
	return kSZmap


def KSxkSZ (Wx, method='nearest', sigmaG = 1.0):
	
	KS = kmapGen(Wx)
	kSZ = kSZmapGen (Wx, method=method)
	noise = kSZmapGen (Wx, method=method, noise=True)

	## masking
	#galn = galnGen(Wx)
	#mask = ones(shape=galn.shape)
	#idx = where(galn<0.5)
	#mask[idx] = 0
	#mask_smooth = WLanalysis.smooth(mask, sigmaG*PPA512)
	#KS *= mask_smooth
	#noise *= mask_smooth
	#kSZ *= mask_smooth
	##KS[idx]=0 ## put a smooth mask later
	##noise[idx]=0
	##kSZ[idx]=0
	
	fmask=1.0
	#fsky=1.0
	sizedeg = (sizes[Wx-1]/512.0)**2*12.0
	#fmask = sum(mask_smooth)/galn.shape[0]**2
	fsky = fmask*sizedeg/41253.0# 41253.0 deg is the full sky in degrees
	
	edges = edgesGen(Wx)
	ell_arr, CCK = WLanalysis.CrossCorrelate (KS,kSZ,edges=edges)
	ell_arr, CCB = WLanalysis.CrossCorrelate (KS,noise,edges=edges)
	CCK /= fmask**2
	CCB /= fmask**2
	
	# error
	autoK = WLanalysis.PowerSpectrum(KS, sizedeg = sizedeg, edges=edges)[-1]/fmask**2
	autokSZ = WLanalysis.PowerSpectrum(kSZ, sizedeg = sizedeg, edges=edges)[-1]/fmask**2
	autoB = WLanalysis.PowerSpectrum(noise, sizedeg = sizedeg, edges=edges)[-1]/fmask**2

	d_ell = ell_arr[1]-ell_arr[0]
	errK = sqrt(autoK*autokSZ/fsky/(2*ell_arr+1)/d_ell)
	errB = sqrt(autoK*autoB/fsky/(2*ell_arr+1)/d_ell)	
	return ell_arr, CCK, CCB, errK, errB

def CrossPower(CCK, CCB, errK, errB, method='nearest', sigmaG=1.0):
	
	f=figure(figsize=(8,6))
	ax=f.add_subplot(111)
	#ax.plot(ell_arr, CCK, 'bo', label='$\kappa$ x kSZ')
	#ax.plot(ell_arr, CCB, 'ro',label='$\kappa$ x noise')

	ax.errorbar(ell_arr, CCK, errK, fmt='o',color='b', label='$\kappa$ x kSZ')
	ax.errorbar(ell_arr, CCB, errB, fmt='o',color='r',label='$\kappa$ x noise')

	legend(loc=0)
	#ax.set_xscale('log')
	ax.set_xlabel('ell')
	ax.set_ylabel(r'$\ell(\ell+1)P_{n\kappa}(\ell)/2\pi$')
	ax.set_title('%s, %s arcmin'%(method, sigmaG))
	#show()
	savefig(plot_dir+'kSZxCFHT_%s_sigmaG%s.jpg'%(method,sigmaG))
	close()


for method in ('nearest','linear','cubic'):
	for sigmaG in sigmaG_arr[:-3]:
		print 'method, sigmaG', method, sigmaG
		CC_arr = array([KSxkSZ(Wx, method=method, sigmaG=sigmaG) for Wx in range(1,5)])
		errK_arr = CC_arr[:,3]
		errB_arr = CC_arr[:,4]
		weightK = 1/errK_arr/sum(1/errK_arr, axis=0)
		weightB = 1/errB_arr/sum(1/errB_arr, axis=0)
		errK = 1/sum(1/errK_arr, axis=0)
		errB = 1/sum(1/errB_arr, axis=0)
		ell_arr = CC_arr[0,0]
		CCK = sum(CC_arr[:,1]*weightK,axis=0)
		CCB = sum(CC_arr[:,2]*weightB,axis=0)
		CrossPower(CCK, CCB, errK, errB, method=method, sigmaG=sigmaG)