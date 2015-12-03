import numpy as np
from scipy import *
import astropy.io.fits as pyfits
from astropy import wcs
from scipy import interpolate, stats, fftpack
import os, sys
import scipy.ndimage as snd
import WLanalysis
from emcee.utils import MPIPool

##################### constants #################
PPR=8468.416479647716 #pixels per radians
PPA=2.4633625 #pixels per arcmin, PPR/degrees(1)/60
APP=0.40594918531072871 #arcmin per pix, degrees(1/PPR)*60
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
sizes = (1330, 800, 1120, 950)

#################### map making functions ##################
#rad2pix=lambda x, size: around(size/2.0-0.5 + x*PPR).astype(int) # from 
#mask_dir='/Users/jia/weaklensing/CFHTLenS_downloads/mask/'
#map_dir = '/Users/jia/weaklensing/CFHTLenS_downloads/mask_pix/'
mask_dir='/work/02977/jialiu/multiplicative/mask_ludo/'
#fn_arr = os.listdir(mask_dir)
mask_big=lambda Wx: '/work/02977/jialiu/multiplicative/mask_ludo/W%i.16bit.small.reg2.fits'%Wx

Wx = int(sys.argv[1])

print 'Wx'
hdulist = pyfits.open(mask_big(Wx))
size=sizes[Wx-1]
center=centers[Wx-1]
headers=hdulist[0].header
w=wcs.WCS(headers)
footprint = w.calc_footprint()
data = np.array(hdulist[0].data)
data[data<=1]=0.0
data[data>1]=1.0
#print 'got data',time.strftime("%Y-%m-%d %H:%M")
xy=np.indices(data.shape)
f_Wx = WLanalysis.gnom_fun(centers[Wx-1])
step = ceil(data.shape[1]/63.0)

import time
print 'begin',time.strftime("%Y-%m-%d %H:%M")

# 1 field needs to be split into 24 cores to do
def partialdata2grid (icount):
    '''for a small portion of the data, put into a grid'''
    print 'icount',icount
    ixy = xy[:,:,step*icount:step*(1+icount)]
    radeclist = (array(w.wcs_pix2world(ixy[0],ixy[1],0)).T).reshape(-1,2) 
    
    ####### recycle xy, to save space..
    y, x = f_Wx (radeclist)
    idata=data[:,step*icount:step*(1+icount)].flatten().reshape(1,-1)
    #y, x = xy.T
    ipix, ipix_mask = WLanalysis.coords2grid(x, y, idata, size=sizes[Wx-1])
    print icount,'done coords2grid',time.strftime("%Y-%m-%d %H:%M")
    
    save(mask_dir+'smaller/W%i_%i_numpix'%(Wx,icount), ipix)
    save(mask_dir+'smaller/W%i_%i_nummask'%(Wx,icount), ipix_mask)
    #ipix is the num. of pixels fall in that big pix, ipix_mask is the mask
    return ipix, ipix_mask

p = MPIPool()    
if not p.is_master():
    p.wait()
    sys.exit(0)
small_map = sum(array(p.map(iCC, range(63))),axis=0)
save(mask_dir+'W%i_smaller_mask.npy',small_map)

p.close()
