import numpy as np
from scipy import *
from pylab import *
import os
import astropy.io.fits as pyfits
from astropy import wcs
import scipy.ndimage as snd
from WLanalysis import *

#mask_dir = '/direct/astro+astronfs01/workarea/jia/CFHT/mask/'
#mask_bin_dir = '/direct/astro+astronfs01/workarea/jia/CFHT/mask_binned/'
#mask_smooth_dir = '/direct/astro+astronfs01/workarea/jia/CFHT/mask_smooth/'
mask_dir ='/direct/astro+astronfs03/workarea/jia/CFHT/CFHT/CFHTdownload/mask/'
mask_bin_dir = '/direct/astro+astronfs03/workarea/jia/CFHT/CFHT/junk/mask_binned/'
masks = os.listdir(mask_dir)

RA1 =[30.0, 39.0]#starting RA for W1, at the center of the pixel
DEC1=[-11.5,-3.5]
RA2 =[132.0, 137.0]
DEC2=[-6.0,-0.5]
RA3 =[208.0, 221.0]
DEC3=[51.0, 58.0]
RA4 =[329.5, 336.0]
DEC4=[-1.2, 5.0]
RAs=array([RA1,RA2,RA3,RA4])
DECs=array([DEC1,DEC2,DEC3,DEC4])
dpp=0.0016914558667664816
xnum = lambda RA: round((amax(RA)-amin(RA))/dpp+500)
#genWx = lambda i: zeros(shape=(xnum(RAs[i]),ynum(DECs[i])))
genWx = lambda i: zeros(shape=(xnum(DECs[i]), xnum(RAs[i])))

mask_W1=genWx(0)
mask_W2=genWx(1)
mask_W3=genWx(2)
mask_W4=genWx(3)
mask_Wx=[mask_W1,mask_W2,mask_W3,mask_W4]
mask_Wx_repeat=[genWx(0),genWx(1),genWx(2),genWx(3)]#just put a number at where there's image, to make sure we cover the whole field
j=0

#def genfp(ifile):
	#hdulist = pyfits.open(mask_dir+ifile)
	#headers=hdulist[0].header
	#w=wcs.WCS(headers)
	#hdulist.close()
	#footprint = w.calcFootprint()
	#return footprint

#def genfpii(ifile):
	#return int(ifile[1])-1


for ifile in masks:
	#fn = mask_dir+
	
	# get x0, y0
	print j,ifile
	hdulist = pyfits.open(mask_dir+ifile)
	headers=hdulist[0].header
	w=wcs.WCS(headers)
	hdulist.close()
	footprint = w.calcFootprint()
	
	
	# need to change here?
	RAl = amin(footprint[:,0])
	#RAr = amax(footprint[:,0])
	DECl= amin(footprint[:,1])
	#DECr= amax(footprint[:,1])
	
	ii=int(ifile[1])-1
	#x0 = around((DECl+2.5/60-DECs[ii,0])/dpp)
	#y0 = around((RAl+2.5/60-RAs[ii,0])/dpp)#RA left + some spare pixels - iith field RA0
	x0 = around((DECl-DECs[ii,0])/dpp)
	y0 = around((RAl-RAs[ii,0])/dpp)
	
	fn_mask = mask_bin_dir+'binned_mask_%s'%(ifile)
	imask = readFits(fn_mask)
	# Jia 05/08/2014 edit, need to flip small image to match the masks
	# rot90(a)[:,::-1]: rotate left, flip horizontal
	# imask = rot90(imask)[:,::-1]
	
	x1 = x0+imask.shape[0]
	y1 = y0+imask.shape[1]
	mask_Wx[ii][x0:x1,y0:y1]+=imask
	mask_Wx_repeat[ii][x0:x1,y0:y1]+=1
	j+=1
	
for i in range(4):
	imask_Wx=(mask_Wx[i]>0).astype(int)
	savetxt(mask_bin_dir+'Mask_W%i_fix05082014.txt'%(i+1),ShrinkMatrix(imask_Wx,4),fmt='%i')
	savetxt(ask_bin_dir+'Mask_W%i_repeat_fix05082014.fits'%(i+1),ShrinkMatrix(mask_Wx_repeat[i],4),fmt='%i')
	#writeFits(ShrinkMatrix(imask_Wx,4),mask_bin_dir+'Mask_W%i_fix05082014.fits'%(i+1))
	#writeFits(ShrinkMatrix(mask_Wx_repeat[i],4),mask_bin_dir+'Mask_W%i_repeat_fix05082014.fits'%(i+1))