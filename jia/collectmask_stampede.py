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

print 'Wx', Wx
hdulist = pyfits.open(mask_big(Wx))
size=sizes[Wx-1]
center=centers[Wx-1]
headers=hdulist[0].header
w=wcs.WCS(headers)
footprint = w.calc_footprint()
step = [522, 294, 420, 351][Wx-1]# ceil(data.shape[1]/63.0)

############ ONCE TIME: saves part of the data into files ######
data = np.array(hdulist[0].data)
data[data<=1]=0.0
data[data>1]=1.0#mask out everything has mask<1

for icount in range(63):
    print icount
    idata=data[:,step*icount:step*(1+icount)]#.flatten().reshape(1,-1)
    save(mask_dir+'smaller/cat_W%i_step%i_start%i'%(Wx,step, icount), idata)
#################################################################

#import time
#print 'begin',time.strftime("%Y-%m-%d %H:%M")
f_Wx = WLanalysis.gnom_fun(centers[Wx-1])

# 1 field needs to be split into 63 cores to do
def partialdata2grid (icount):
    '''for a small portion of the data, put into a grid'''
    print 'icount',icount
    idata = load(mask_dir+'smaller/cat_W%i_step%i_start%i.npy'%(Wx,step, icount))
    print 'loaded',icount
    ix, iy=np.indices(idata.shape)
    iy+=step*icount
    #radeclist = (array(w.wcs_pix2world(ix, iy, 0)).reshape(2,-1)).T ////jia changed on 12/9, since the coordinates seems to be off..
    #radeclist = (array(w.wcs_pix2world(iy, ix, 0)).reshape(2,-1)).T 
    #y, x = f_Wx (radeclist)
    y, x = f_Wx ((array(w.wcs_pix2world(iy, ix, 0)).reshape(2,-1)).T )
    print icount,'done f_wx %s'%(icount)#,time.strftime("%Y-%m-%d %H:%M")
    
    istep=ceil(len(y)/10.0)
    
    ipix_mask = zeros(shape=(sizes[Wx-1], sizes[Wx-1]))
    ipix = ipx_max.copy()
    jj=0
    
    while jj< len(y):
        print 'icount, jj',icount,jj
        iipix_mask,iipix = WLanalysis.coords2grid(ix[jj:jj+istep], iy[jj:jj+istep], idata.flatten().reshape(1,-1)[jj:jj+istep], size=sizes[Wx-1])
        ipix_mask += iipix_mask
        ipix += iipix
        jj+=istep
    #print icount,'done coords2grid %s'%(icount),time.strftime("%Y-%m-%d %H:%M")    
    
    save(mask_dir+'smaller/W%i_%i_numpix'%(Wx,icount), ipix)
    save(mask_dir+'smaller/W%i_%i_nummask'%(Wx,icount), ipix_mask)
    #ipix is the num. of pixels fall in that big pix, ipix_mask is the mask
    return ipix, ipix_mask

p = MPIPool()    
if not p.is_master():
    p.wait()
    sys.exit(0)

ismall_map=p.map(partialdata2grid, range(127))
small_map = sum(array(ismall_map),axis=0)
save(mask_dir+'W%i_smaller_mask.npy'%(Wx),small_map)

p.close()
