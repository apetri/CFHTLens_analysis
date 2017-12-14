# Jia Liu 2014/2/7 
# functions and routines used in my weak lensing analysis

import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack
from scipy.fftpack import fftfreq, fftshift,ifftshift
#from pylab import *
import os
import scipy.ndimage as snd
#import matplotlib.pyplot as plt
######### uncomment one of the following 2 lines######
import astropy.io.fits as pyfits
#import pyfits
import KSI
from random import gauss

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
#import fitsio
def readFits (fitsfile):
    '''Input: 
    fitsfile = file name of the fitsfile
    Output:
    data array for the fitsfile.
    '''
    hdulist = pyfits.open(fitsfile)
    data = np.array(hdulist[0].data)
    #data = fitsio.read(fitsfile)#.astype(float)
    return data

def writeFits (data, filename, rewrite = False):
    '''Input:
    data = to be written
    filename = the file name of the fitsfile, note this needs to be the full path, otherwise will write to current directory.
    '''
    hdu = pyfits.PrimaryHDU(data)
    if rewrite:
        try:
            hdu.writeto(filename)
        except IOError:
            print filename,'exist, will over-write'
            os.remove(filename)
            hdu.writeto(filename)
    else:
        hdu.writeto(filename)
    
ell2arcmin = lambda ell: 360.0*60.0/ell
arcmin2ell = lambda arcmin: 360.0*60.0/arcmin

def TestFitsComplete (fn, return_file = False):
    '''Input: fn
    Return: True if fn exists and unbroken, false otherwise
    Option (impletement in the future): if return_file = True, 
    then return True/False, file/False
    '''
    if not os.path.isfile(fn): # if this file doesn't exist, return false
        isfile = False
        ifile = False
        try: 
            os.remove(fn)
        except Exception: # to correct for scratch failure in Oct/15/2014
            pass    
    elif os.stat(fn)[6]<10: #added 2014/11/27, check if empty file
        try: 
            os.remove(fn)
        except Exception: # to correct for scratch failure in Oct/15/2014
            pass
        isfile = False
        ifile = False
    else: # if file exist, test if can open (sometimes it's empty file), delete if empty
        try:
            ifile = readFits(fn)
            isfile = True
        except Exception:
            os.remove(fn)
            isfile = False
            ifile = False
    if return_file:
        return isfile, ifile
    else:
        return isfile
    
def TestComplete(file_ls,rm = False):
    '''Test if a list of file all exist.
    Input:
    file_ls: a list of files
    rm: if this list is imcomplete (some files don't exist), remove the other files if rm==True, else do nothing.
    Output:
    True if all files exist.
    Flase if not all files exist.
    '''
    allfiles = True
    for ifile in file_ls:
        if not os.path.isfile(ifile):
            allfiles = False
            break
    if allfiles == False and rm:
        for ifile in file_ls:
            if os.path.isfile(ifile):
                os.remove(ifile)
    return allfiles

def gnom_fun(center):
    '''Create a function that calculate the location of a sky position (ra, dec) on a grid, centered at (ra0, dec0), using Gnomonic projection (flat sky approximation). 
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
    return matrix_new

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
    x1 = concatenate(([-0.075, -0.025, 0],x1))
    P = concatenate(([0, 0, 0],P))#force PDF to be 0 at 0
    f = interpolate.InterpolatedUnivariateSpline(x1, P) # interpolate the PDF with a 3rd order spline function
    fint = lambda edge: f.integral(edge[0],edge[1]) # integrate the region between bin left and right
    if edges==None:#not bool(edges.any()): # find edges for each bin, if not already provided
        step = (x2[1:]-x2[:-1])/2
        binwidth = (step[1:]+step[:-1])/2
        midedges = array([x2[1:-1]-binwidth,x2[1:-1]+binwidth]).T
        leftedge = [2*x2[0]-midedges[0,0],midedges[0,0]]
        rightedge= [midedges[-1,-1], 2*x2[-1]-midedges[-1,-1]]
        edges = concatenate(([leftedge],midedges,[rightedge]))
    newP = array(map(fint,edges)).T
    newP /= sum(newP)
    return newP

def DrawFromPDF_junk (x, P, n):
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

def DrawFromPDF(P, n=2):
    '''
    Given a discrete PDF x (the probability), 1 number from x.
    '''
    P /= sum(P) # normalize P
    custm = stats.rv_discrete(name='custm', values=(arange(len(P)), P))
    R = custm.rvs(size=n) 
    return R
########## begin: CFHT catalogue to smoothed shear maps #########
PPR512=8468.416479647716#pixels per radians
PPA512=2.4633625#pixels per arcmin, PPR/degrees(1)/60
APP512=0.40594918531072871#arcmin per pix, degrees(1/PPR)*60

def coords2grid(x, y, k, size=512):
    '''returns a grid map, and galaxy counts.
    
    Input:
    x, y: x, y in radians, (0, 0) at map center
    k: the quantity in catalogue, to be put onto a grid.
    note: k can be 1-D arry of length N (=len(x)), but also can be multiple dimension (M, N), if so, also return a multiple dimension grid.
    Pixel resolution is ~2.5 pixels per arcmin
    
    Output:
    Mk, galn (galaxy counts per pixel)
    (written 2/14/2014)
    '''
    rad2pix=lambda x: around(size/2.0-0.5 + x*PPR512).astype(int)
    x = rad2pix(x)
    y = rad2pix(y)
    idx_range = where((x<size) & (x>=0) & (y<size) & (y>=0))[0]
    x = x[idx_range]
    y = y[idx_range]
    k = k[:,idx_range]
    ## first put galaxies to grid, note some pixels may have multiple galaxies
    if len(k.shape)>1:
        Mk = zeros(shape=(k.shape[0],size,size))
    else:
        Mk = zeros(shape=(1,size,size))
    galn= zeros(shape=(size,size),dtype=int)
    
    ## put e1,e2,w,galcount into grid, taken into account the 
    xy = x+y*1j #so to sort xy as one array
    sorted_idx = argsort(xy) #get the index that gives a sorted array for xy
    
    ## edit on 12/09, miminize memory usage
    #ar = xy[sorted_idx] #sorted xy
    #left_idx = arange(len(ar)) #left over idx that are used to put data into grid
    #ar0 = ar.copy()    
    ar0=xy[sorted_idx]
    left_idx = arange(len(ar0))
    xy = None 
    #############################
    
    j=0
    while len(left_idx) > 0: # len(left_idx) = #gals to be put into the grid
        #a, b=unique(ar0, return_index=True)
        b=unique(ar0, return_index=True)[1]
        # a is the unique pixels
        # b is the index of those unique pixels
        
        if j>0:
            b=b[1:]# b[0]=-1 from j=1
        
        #put unique values into matrix
        ix = sorted_idx[b]
        for l in range(Mk.shape[0]):
            Mk[l][x[ix],y[ix]] += k[l][ix]
        galn[x[ix],y[ix]]+= 1

        left_idx=setdiff1d(left_idx, b)# Return the sorted, unique values in 'left_idx' that are not in 'b'.
        ar0[b] = -1 # a number that's smaller than other indices (0..)
        j += 1 #j is the repeating #gal count, inside one pix
    return Mk.squeeze(), galn

def weighted_smooth(kmap, Mw, PPA=PPA512, sigmaG=1):
    sigma = sigmaG * PPA # argmin x pixels/arcmin
    
    smooth_w = snd.filters.gaussian_filter(Mw.astype(float),sigma,mode='constant')
    
    smooth_kmap = snd.filters.gaussian_filter(kmap.astype(float),sigma,mode='constant')
    
    smooth_kmap /= smooth_w
    smooth_kmap[isnan(smooth_kmap)] = 0
    return smooth_kmap

smooth = lambda kmap, sigma: snd.filters.gaussian_filter(kmap.astype(float),sigma,mode='constant')
########## end: CFHT catalogue to smoothed shear maps #########

########## begin: mass construcion (shear to convergence) #####

########## method 1, Kaiser-Square 93, convolution ##########
def D_kernel(size): 
    '''Create the kernel as shown in Schneider review eq 41.
    Size = width of the kernel.
    size = 51 seem sufficient for 2048x2048.
    Note: needs complex conjugate of D, see eq 44
    '''
    y, x = np.indices((size,size),dtype=float)
    center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
    x -= center[0]
    y -= center[1]
    D = -1/(x-y*1j)**2
    D[np.isnan(D)]=0+0j    
    return conjugate(D)

def KS93(shear1, shear2):
    '''KS-inversion from shear to convergence, using Schneider eq 44.
    Requires cython code KSI.pyx'''
    smap=shear1+shear2*1.0j    
    smap=smap
    smap=smap.astype(complex128)
    D = D_kernel(31)
    D = D.astype(complex128)
    kmap = KSI.KSI_calc(smap, D)
    return real(kmap)

###### method 2 (also KS93): in fourier space (Van Wearbeke eq. 7) #####
def KSvw(shear1, shear2):
    '''Kaiser-Squire93 inversion, follow Van Waerbeke 2013 eq 7'''
    shear1_fft = fftpack.fft2(shear1.astype(float))
    shear2_fft = fftpack.fft2(shear2.astype(float))
    n0, n1=shear1.shape
    freq0 = fftfreq(n0,d=1.0/n0)+0.5
    #freq0 = fftfreq(n0,d=1.0/n0)#correct one, but induce nan
    k1,k2 = meshgrid(freq0,freq0)
    kappa_fft = (k1**2-k2**2)/(k1**2+k2**2)*shear1_fft+2*k1*k2/(k1**2+k2**2)*shear2_fft
    #kappa_fft[isnan(kappa_fft)]=0
    kappa = fftpack.ifft2(kappa_fft)
    return real(kappa)

####### method 3: Aperture mass (Bard 2013 eq. 1)##############
Q = lambda x: 1.0/(1+exp(6.0-160*x)+exp(-47.0+50.0*x))*tanh(x/0.15)/(x/0.15) 
#x = ti/tmax, where tmax=theta_max, the radius where the filter is tuned.
tmax = 14#13.79483 #theta_max = 5.6 arcmin 

def Q_kernel(size,tmax = tmax):
    y, x = np.indices((size,size),dtype=float)
    center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
    x -= center[0]
    y -= center[1]
    #y = center[1]-y
    r = np.hypot(x, y)/tmax
    phi = np.angle(x+y*1.0j) # phi is the angle with respect to the horizontal axis between positions theta0 and theta in the map. 
    Q0 = Q(r)
    Q0[isnan(Q0)]=0
    return Q0, phi
    
def apMass(shear1, shear2, Mw = None, Mm = None, size = 2*tmax+7, tmax=tmax):
    if Mw.any():
        shear1*=Mw
        shear2*=Mw
    shear1 = snd.filters.gaussian_filter(shear1,0)#smooth(shear1,0) #to get rid of the endien problem
    shear2 = snd.filters.gaussian_filter(shear2,0)#smooth(shear2,0)
    shear1=shear1.astype(float64)
    shear2=shear2.astype(float64)
    Q0, phi = Q_kernel(size)
    apMmap = KSI.apMass_calc(shear1,shear2,Q0,phi)
    if Mm.any():
        weight = (Mw*Mm).astype(float64)
        apM_norm = KSI.apMass_norm(weight,Q0)
        apMmap /= apM_norm
        apMmap[isnan(apMmap)] = 0
    return apMmap

########## end: mass construcion (shear to convergence) #####

####### begin: randon orientation ###########################
def rndrot (e1, e2, iseed=None, deg=None):
    '''rotate galaxy with ellipticity (e1, e2), by a random angle. 
    generate random rotation while preserve galaxy size and shape info
    '''
    if iseed:
        np.random.seed(iseed)
    ells = e1+1j*e2
    if deg:
        if deg > pi:
            deg = radians(deg)
        ells_new = -ells*exp(-2j*deg)
    else:
        ells_new = -ells*exp(-4j*pi*rand(len(e1)))
    return real(ells_new), imag(ells_new)
####### end: randon orientation #############################

def eobs_fun (g1, g2, k, e1, e2):
    '''van wearbeke 2013 eq 5-6, get unbiased estimator for shear.
    Input:
    g1, g2: shear
    k: convergence
    e1, e2: galaxy intrinsic ellipticity
    Output:
    e_obs1, e_obs2
    '''
    g = (g1+1j*g2)/(1-k)
    eint = e1+1j*e2
    eobs = (g+eint)/(1-g*eint)
    return real(eobs), imag(eobs)

########## begin: power spectrum ############################
def bell_2D (size, sigma):
    '''return a b_ell matrix, smoothing scale simag in unit of pixels.
    b_ell = exp(-ell**2*sigmaG**2/2.0), where sigmaG in unit of radians.'''
    y, x = np.indices((size, size))
    center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
    if size%2 == 0:
        center+=0.5
    r = np.hypot(x - center[0], y - center[1])
    bell_mat = exp(-(r*sigma*2*pi/size)**2/2.0)
    return bell_mat
    
def azimuthalAverage(image, center = None, edges = None, logbins = True, bins = 50, return_num_modes=0):
    """
    Calculate the azimuthally averaged radial profile.
    Input:
    image = The 2D image
    center = The [x,y] pixel coordinates used as the center. The default is None, which then uses the center of the image (including fracitonal pixels).
    Output:
    ell_arr = the ell's, lower edge
    tbin = power spectrum
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
    
    ## added below on oct/31/2014, 
    ## since nyquist freqnecy is not centered for even # mapsize

    if image.shape[0]%2 == 0:
        center+=0.5
    
    r = np.hypot(x - center[0], y - center[1])#distance to center pixel, for each pixel

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind] # the index to sort by r
    i_sorted = image.flat[ind] # the index of the images sorted by r

    # find index that's corresponding to the lower edge of each bin
    kmin=1.0
    kmax=image.shape[0]/2.0
    if edges == None:
        if logbins:
            edges = logspace(log10(kmin),log10(kmax),bins+1)
        else:
            #edges = linspace(kmin,kmax+0.001,bins+1)    
            edges = linspace(kmin,kmax,bins+1)
    if edges[0] > 0:
        edges = append([0],edges)
        
    hist_ind = np.histogram(r_sorted,bins = edges)[0] # hist_ind: the number in each ell bins, sum them up is the index of lower edge of each bin, first bin spans from 0 to left of first bin edge.    
    hist_sum = np.cumsum(hist_ind)
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[hist_sum[1:]] - csim[hist_sum[:-1]]
    radial_prof = tbin/hist_ind[1:]
    if return_num_modes:
        #where N=hist_sum[1:]-hist_sum[:-1] is the number of modes in each bin, but note N/2 is the num of independent modes
        return edges[1:], radial_prof, (hist_sum[1:]-hist_sum[:-1])/2.0
    else:
        return edges[1:], radial_prof

edge2center = lambda x: x[:-1]+0.5*(x[1:]-x[:-1])

def azimuthalAverage3D(grid, edges = None, logbins = True, bins = 50, return_num_modes=0):
    z, y, x = np.indices(grid.shape)
    icenter=(x.max()-x.min())/2.0
    center = np.array([icenter, icenter, icenter])
    
    ## added below on oct/31/2014, 
    ## since nyquist freqnecy is not centered for even # mapsize

    if grid.shape[0]%2 == 0:
        center+=0.5
    
    r = sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)#distance to center pixel, for each pixel

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind] # the index to sort by r
    i_sorted = grid.flat[ind] # the index of the images sorted by r

    # find index that's corresponding to the lower edge of each bin
    kmin=1.0
    kmax=grid.shape[0]/2.0
    if edges == None:
        if logbins:
            edges = logspace(log10(kmin),log10(kmax),bins+1)
        else:
            #edges = linspace(kmin,kmax+0.001,bins+1)    
            edges = linspace(kmin,kmax,bins+1)
    if edges[0] > 0:
        edges = append([0],edges)
        
    hist_ind = np.histogram(r_sorted,bins = edges)[0] # hist_ind: the number in each ell bins, sum them up is the index of lower edge of each bin, first bin spans from 0 to left of first bin edge.    
    hist_sum = np.cumsum(hist_ind)
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[hist_sum[1:]] - csim[hist_sum[:-1]]
    radial_prof = tbin/hist_ind[1:]
    if return_num_modes:
        #where N=hist_sum[1:]-hist_sum[:-1] is the number of modes in each bin, but note N/2 is the num of independent modes
        return edges[1:], radial_prof, (hist_sum[1:]-hist_sum[:-1])/2.0
    else:
        return edges[1:], radial_prof
    
def PowerSpectrum3D(grid, logbins = True, bins=50):#edges should be pixels
    '''Calculate the power spectrum for a square image, with normalization.
    Input:
    img = input square image in numpy array.
    sizedeg = image real size in deg^2
    edges = ell bin edges, length = nbin + 1, if not provided, then do 1000 bins.
    sigmaG = smoothing scale in arcmin
    Output:
    powspec = the power at the bins
    ell_arr = lower bound of the binedges
    '''
    isize = grid.shape[0]
    #F = fftpack.fftshift(fftpack.fft2(img))
    F = fftshift(fftpack.fftn(grid))
    psd3D = np.abs(F)**2

    k_arr, psd1D = azimuthalAverage3D(psd3D, logbins = logbins, bins=bins)
    k_arr = edge2center(k_arr)
    #ell_arr *= 360./sqrt(sizedeg)# normalized to our current map size
    #norm = ((2*pi*sqrt(sizedeg)/360.0)**2)/(size**2)**2
    #powspec = ell_arr*(ell_arr+1)/(2*pi) * norm * psd1D
    return k_arr, psd1D

def PowerSpectrum(img, sizedeg = 12.25, edges = None, logbins = True, sigmaG=0, bins=50):#edges should be pixels
    '''Calculate the power spectrum for a square image, with normalization.
    Input:
    img = input square image in numpy array.
    sizedeg = image real size in deg^2
    edges = ell bin edges, length = nbin + 1, if not provided, then do 1000 bins.
    sigmaG = smoothing scale in arcmin
    Output:
    powspec = the power at the bins
    ell_arr = lower bound of the binedges
    '''
    img = img.astype(float)
    size = img.shape[0]
    #F = fftpack.fftshift(fftpack.fft2(img))
    F = fftshift(fftpack.fft2(img))
    psd2D = np.abs(F)**2
    ## correct for b_ell
    psd2D /= bell_2D(size, sigmaG*PPA512)**2
    ell_arr, psd1D = azimuthalAverage(psd2D, center=None, edges = edges,logbins = logbins, bins=bins)
    ell_arr = edge2center(ell_arr)
    ell_arr *= 360./sqrt(sizedeg)# normalized to our current map size
    norm = ((2*pi*sqrt(sizedeg)/360.0)**2)/(size**2)**2
    powspec = ell_arr*(ell_arr+1)/(2*pi) * norm * psd1D
    return ell_arr, powspec

def PowerSpectrum_Pell_binning(img, sizedeg = 12.25, edges = None, logbins = True):#edges should be pixels
    '''Calculate the power spectrum for a square image, with normalization.
    Input:
    img = input square image in numpy array.
    sizedeg = image real size in deg^2
    edges = ell bin edges, length = nbin + 1, if not provided, then do 1000 bins.
    Output:
    powspec = the power at the bins
    ell_arr = lower bound of the binedges
    '''
    y, x = np.indices(img.shape)
    center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
    r = np.hypot(x - center[0], y - center[1])
    r *= 360./sqrt(sizedeg)
    
    img = img.astype(float)
    size = img.shape[0]
    #F = fftpack.fftshift(fftpack.fft2(img))
    F = fftshift(fftpack.fft2(img))
    psd2D = np.abs(F)**2
    ell_arr, psd1D = azimuthalAverage(psd2D*r*(r+1), center=None, edges = edges,logbins = logbins)
    ell_arr = edge2center(ell_arr)
    ell_arr *= 360./sqrt(sizedeg)# normalized to our current map size
    norm = ((2*pi*sqrt(sizedeg)/360.0)**2)/(size**2)**2
    powspec = 1/(2*pi) * norm * psd1D
    return ell_arr, powspec
########## end: power spectrum ############################


def CrossCorrelate(img1, img2, edges = None, logbins = True, sigmaG1=0, sigmaG2=0,sizedeg=False,PPA=PPA512):#edges should be pixels
    '''Calculate the power spectrum for a square image, with normalization.
    Input:
    img1, img2 = input square image in numpy array.
    edges = ell bin edges, length = nbin + 1, if not provided, then do 50 bins.
    Output:
    powspec = the power at the bins
    ell_arr = lower bound of the binedges
    sigmaG1, sigmaG2 = smoothing scales for map1,2 in arcmin
    '''
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    size = img1.shape[0]
    if not sizedeg:
        sizedeg = (size/512.0)**2*12.0# assuming 512x512 for 12 deg^2 area
    #F = fftpack.fftshift(fftpack.fft2(img))
    F1 = fftshift(fftpack.fft2(img1))
    F2 = fftshift(fftpack.fft2(img2))
    psd2D = np.conj(F1)*F2#calculate cross correlation
    psd2D /= bell_2D(size, sigmaG1*PPA)*bell_2D(size, sigmaG2*PPA)
    ell_arr, psd1D = azimuthalAverage(psd2D, center=None, edges = edges,logbins = logbins)
    ell_arr = edge2center(ell_arr)
    ell_arr *= 360./sqrt(sizedeg)# normalized to our current map size
    norm = ((2*pi*sqrt(sizedeg)/360.0)**2)/(size**2)**2
    powspec = ell_arr*(ell_arr+1)/(2*pi) * norm * psd1D
    return ell_arr, powspec
########## begin: peak counts ############################
peaks_mat = lambda kmap: KSI.findpeak_mat(kmap.astype(float))
peaks_list = lambda kmap: array(KSI.findpeak_list(kmap.astype(float)))
def peaks_mask_hist (kmap, mask, bins, kmin = -0.04, kmax = 0.12):
    '''If kamp has a mask, return only peaks have no mask on them, histogramed to binedges.
    mask = 1 for good non-mask regions, 0 for mask.
    '''
    kmap_masked = kmap*mask
    kmap_masked[where(mask==0)] = kmax*10#give a high value to mask region, so even it's considered a peak, it will fall out of the histogram
    peaks = peaks_list(kmap_masked)
    peaks_hist = histogram(peaks,range=(kmin,kmax),bins=bins)[0]
    return peaks_hist
########## end: peak counts ############################    
    
def create_dir_if_nonexist(dirname):
    try:
        os.mkdir(dirname)
    except Exception:
        print 'error'
        pass

######### begin: build interpolator ###############
def buildInterpolator(obs_arr, cosmo_params):
    '''Build an interpolator:
    input:
    obs_arr = (points, Nbin), where # of points = # of models
    cosmo_params = (points, Nparams), currently Nparams is hard-coded
    to be 3 (om,w,si8)
    output:
    spline_interps
    Usage:
    spline_interps[ibin](im, wm, sm)
    '''
    m, w, s = cosmo_params.T
    spline_interps = list()
    for ibin in range(obs_arr.shape[-1]):
        model = obs_arr[:,ibin]
        iinterp = interpolate.Rbf(m, w, s, model)
        spline_interps.append(iinterp)
    #return spline_interps
    def interp_cosmo (params):
        '''Interpolate the powspec for certain param.
        Params: list of 3 parameters = (om, w, si8)
        Method: "multiquadric" for spline (default), and "GP" for Gaussian process.
        '''
        mm, wm, sm = params
        gen_ps = lambda ibin: spline_interps[ibin](mm, wm, sm)
        ps_interp = array(map(gen_ps, range(obs_arr.shape[-1])))
        ps_interp = ps_interp.reshape(-1,1).squeeze()
        return ps_interp
    return interp_cosmo

def buildInterpolator2D(obs_arr, cosmo_params, method='Rbf'):
    '''Build an interpolator:
    input:
    obs_arr = (points, Nbin), where # of points = # of models
    cosmo_params = (points, Nparams), currently Nparams is hard-coded
    to be 3 (om,w,si8)
    output:
    spline_interps
    Usage:
    spline_interps[ibin](im, wm, sm)
    '''
    m, s = cosmo_params.T
    spline_interps = list()
    for ibin in range(obs_arr.shape[-1]):
        model = obs_arr[:,ibin]
        if method == 'Rbf':
            iinterp = interpolate.Rbf(m, s, model)#
        elif method == 'linear':
            iinterp = interpolate.LinearNDInterpolator(cosmo_params,model)#
        elif method == 'clough':
            iinterp = interpolate.CloughTocher2DInterpolator(cosmo_params,model)#
        #iinterp = interpolate.Rbf(m, s, model)
        spline_interps.append(iinterp)
    #return spline_interps
    def interp_cosmo (params):
        '''Interpolate the powspec for certain param.
        Params: list of 3 parameters = (om, w, si8)
        Method: "multiquadric" for spline (default), and "GP" for Gaussian process.
        '''
        mm, sm = params
        gen_ps = lambda ibin: spline_interps[ibin](mm, sm)
        ps_interp = array(map(gen_ps, range(obs_arr.shape[-1])))
        ps_interp = ps_interp.reshape(-1,1).squeeze()
        return ps_interp
    return interp_cosmo

def findlevel (H):
    '''Find 68%, 95%, 99% confidence level for a probability 2D plane H.
    return V = [v68, v95, v99]
    '''
    H /= sum(H)
    H /= float(sum(H))
    H[isnan(H)]=0
    
    idx = np.argsort(H.flat)[::-1]
    H_sorted = H.flat[idx]
    H_cumsum = np.cumsum(H_sorted)
    idx68 = where(abs(H_cumsum-0.683)==amin(abs(H_cumsum-0.683)))[0]    
    idx95 = where(abs(H_cumsum-0.955)==amin(abs(H_cumsum-0.955)))[0]
    idx99 = where(abs(H_cumsum-0.997)==amin(abs(H_cumsum-0.997)))[0]
    v68 = float(H.flat[idx[idx68]])
    v95 = float(H.flat[idx[idx95]])
    v99 = float(H.flat[idx[idx99]])
    V = [v68, v95, v99]
    return V

def update_values_by_RaDec (new_ra, new_dec, master_ra, master_dec):
    '''For a list of [new_ra, new_dec], find index in [master_ra, master_dec]
    for elements that are in both lists.
    requires all values in master are unique.
    '''
    RADEC = master_ra+1.0j*master_dec
    iradec = new_ra+1.0j*new_dec
    idx_master_in_new = where(in1d(RADEC, iradec)==True)[0]## indices of RADEC elements that're in iradec
    idx_new_in_master = where(in1d(iradec, RADEC)==True)[0]
    #newidx_master = idx_new_in_master[argsort(iradec[idx_new_in_master])]
    #iRADEC = RADEC[idx]
    #unique_iradec, idx_inverse = unique(iradec, return_inverse=True)
    #newidx = idx[argsort(iRADEC)][idx_inverse]
    return newidx

############## CFHT specific operations ############
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
sizes = (1330, 800, 1120, 950)
rad2pix=lambda x, size: around(size/2.0-0.5 + x*PPR512).astype(int)
def list2coords(radeclist, Wx, offset=False, convert2rad=True):
    '''For a list of radec, return their pixelized position for Wx field.
    '''
    size=sizes[Wx-1]
    xy = zeros(shape = radeclist.shape)
    if offset:
        center = 0.5*(amin(radeclist,axis=0)+amax(radeclist, axis=0))
    else:
        center = centers[Wx-1]
    f_Wx = gnom_fun(center)
    #xy = array(map(f_Wx,radeclist))
    xy = array(f_Wx(radeclist)).T
    if convert2rad:
        xy = rad2pix(xy, size)
    return xy

def interpGridpoints (xy, values, newxy, method='nearest'):
    newvalues = interpolate.griddata(xy, values, newxy, method=method)
    return newvalues

def txt2map_fcn (fn, offset=False, method='nearest'):
    '''put values to grid, similar to cmblGen, except take in the file name.
    '''
    npy_fn = fn[:-3]+'npy'
    if os.path.isfile(npy_fn):
        return load(npy_fn)
    else:
        Wx = int(fn[fn.index('W')+1])
        print 'Wx, fn:', Wx, fn
        size=sizes[Wx-1]
        cmblCoord = genfromtxt(fn)
        radeclist = cmblCoord[:,:-1]
        values = cmblCoord.T[-1]
        xy = list2coords(radeclist, Wx, offset=offset)
        X,Y=meshgrid(range(size),range(size))
        X=X.ravel()
        Y=Y.ravel()
        newxy=array([X,Y]).T
        newvalues = interpGridpoints (xy, values, newxy,method=method)
        cmblmap = zeros(shape=(size,size))
        cmblmap[Y,X]=newvalues    
        cmblmap[isnan(cmblmap)]=0.0
        if offset:
            cmblmap = cmblmap.T
        np.save(fn[:-3]+'npy', cmblmap)
        return cmblmap

def extrap1d(interpolator):
    '''interpolate for values within the interpolator range, extrapolate for outside regions.
    '''
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(map(pointwise, array(xs)))

    return ufunclike

def GRF_Gen_1map (kmap):
    '''return a random gaussian field that has the same power spectrum as img.
    '''
    size = kmap.shape[0]
    F = fftshift(fftpack.fft2(kmap.astype(float)))
    psd2D = np.abs(F)**2 # = real**2 + imag**2

    ell_arr0, psd1D0 = azimuthalAverage(psd2D, center=None, edges = arange(sqrt(2)*size/2))
    ell_arr_center = edge2center(ell_arr0)

    randfft2 = zeros(shape=(size, size))
    y, x = np.indices((size,size))
    center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
    if size%2 == 0:
        center+=0.5
    r = np.hypot(x - center[0], y - center[1])

    extrap = psd1D0[-1]+(ceil(sqrt(2)*size/2+1.0)-ell_arr_center[-1])*(psd1D0[-1]-psd1D0[-2])/(ell_arr_center[-1]-ell_arr_center[-2])

    ell_arr = array([0,]+list(ell_arr_center)+ [ceil(sqrt(2)*size/2+1.0),])
    psd1D = array([psd1D0[0],]+list(psd1D0)+[extrap,])

    p1D_interp = interpolate.griddata(ell_arr, psd1D, r.flatten(), method='nearest')
    p1D_interp[isnan(p1D_interp)]=0

    p2D_mean = p1D_interp.reshape(size,size)

    ############
    #rand_angle = rand(size,size)*2.0*pi
    #psd2D_GRF_Fourier = sqrt(p2D_mean*2.0)*random.normal(0, 1, (size,size)) * (cos(rand_angle) + 1j * sin(rand_angle))
    
    ######## or ###########
    psd2D_GRF_Fourier = sqrt(p2D_mean)* (random.normal(0, 1, (size,size)) + 1j*random.normal(0, 1, (size,size)))
    
    GRF_image = fftpack.ifft2(ifftshift(psd2D_GRF_Fourier))
    GRF = real(GRF_image)
    
    return GRF

def ps1DGen(kmap):
    size = kmap.shape[0]
    F = fftshift(fftpack.fft2(kmap.astype(float)))
    psd2D = np.abs(F)**2 # = real**2 + imag**2

    ell_arr0, psd1D0 = azimuthalAverage(psd2D, center=None, edges = arange(sqrt(2)*size/2))
    ell_arr_center = edge2center(ell_arr0)

    randfft2 = zeros(shape=(size, size))
    y, x = np.indices((size,size))
    center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
    if size%2 == 0:
        center+=0.5
    r = np.hypot(x - center[0], y - center[1])

    extrap = psd1D0[-1]+(ceil(sqrt(2)*size/2+1.0)-ell_arr_center[-1])*(psd1D0[-1]-psd1D0[-2])/(ell_arr_center[-1]-ell_arr_center[-2])
    
    ell_arr = array([0,]+list(ell_arr_center)+ [ceil(sqrt(2)*size/2+1.0),])
    psd1D = array([psd1D0[0],]+list(psd1D0)+[extrap,])

    p1D_interp = interpolate.griddata(ell_arr, psd1D, r.flatten(), method='nearest')
    p1D_interp[isnan(p1D_interp)]=0

    p2D_mean = p1D_interp.reshape(size,size)
    return p2D_mean

class GRF_Gen:
    '''return a random gaussian field that has the same power spectrum as img.
    '''
    def __init__(self, kmap):
        self.size = kmap.shape[0]
        self.GRF = rand(self.size,self.size)
        self.p2D_mean = ps1DGen(kmap)
    
    def newGRF(self):
        #self.rand_angle = rand(self.size**2).reshape(self.size,self.size)*2.0*pi
        self.psd2D_GRF = sqrt(self.p2D_mean)* (random.normal(0, 1, (self.size,self.size)) + 1j*random.normal(0, 1, (self.size,self.size)))
        self.GRF_image = fftpack.ifft2(ifftshift(self.psd2D_GRF))
        self.GRF = real(self.GRF_image)
        return self.GRF

def prob_plane (chisq_fcn, param1_arr, param2_arr):
    heatmap = zeros(shape=(len(param1_arr),len(param2_arr)))
    for i in range(len(param1_arr)):
        for j in range(len(param2_arr)):
            heatmap[i,j] = chisq_fcn(param1_arr[i], param2_arr[j])
    prob = exp(-0.5*heatmap)
    prob /= sum(prob[~isnan(prob)])
    return heatmap, prob

def corr_mat (cov_mat):
    '''calculate the correlation mat
    '''
    diag_sqrt = sqrt(diag(cov_mat))
    X, Y = np.meshgrid(diag_sqrt, diag_sqrt)
    return cov_mat / (X*Y)
