##########################################################
### This code is for Jia's project B - try to find
### observational evidence of over density at peak location
### as discovered by Yang 2011.
### It does the following:
### 1) find PDF for # gal within 2 arcmin as fcn of peak
### hights
### 2) the same, as 1) but for random direction
### 3) L-M conversion: L_k -> halo mass, using Vale&JPO06 (2014/12)
### 4) kappa_proj assuming NFW (2014/12)

import numpy as np
from scipy import *
from pylab import *
import os
import WLanalysis
from scipy import interpolate,stats
from scipy.integrate import quad
import scipy.optimize as op
import sys

######## for stampede #####
#from emcee.utils import MPIPool
#obsPK_dir = '/home1/02977/jialiu/obsPK/'

######## for laptop #####
obsPK_dir = '/Users/jia/CFHTLenS/obsPK/'
plot_dir = obsPK_dir+'plot/'

cluster_counts = 0
halo_plots = 1
list_peaks_cat = 0 #! generate a list of galaxies for all peaks
project_mass = 0
#junk routines below
update_mag_i = 0
plot_galn_vs_kappa_hist = 0
do_hist_galn_magcut = 0
update_mag_all = 0 #! make a list of galaxie catagues with useful quantities
########### constants ######################
z_lo = 0.6
z_hi = '%s_hi'%(z_lo)

sizes = (1330, 800, 1120, 950)
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
PPR512=8468.416479647716
PPA512=2.4633625
c = 299792.458#km/s
Gnewton = 6.674e-8#cgs cm^3/g/s
H0 = 70.0
h = 0.7
OmegaM = 0.25#0.3
OmegaV = 1.0-OmegaM
rho_c0 = 9.9e-30#g/cm^3
M_sun = 1.989e33#gram

############################################
############ functions #####################
############################################

########### generate maps ##################
kmapGen = lambda i, z: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_KS_%s_sigmaG10.fit'%(i, z))
# This is smoothed galn
# galnGen = lambda i, z: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_galn_%s_hi_sigmaG10.fit'%(i, z))
galnGen_hi = lambda i, z: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_galn_%s_hi.fit'%(i, z))
galnGen_lo = lambda i, z: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_galn_%s_lo.fit'%(i, z))
bmodeGen = lambda i, z: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_Bmode_%s_sigmaG10.fit'%(i,z))

cat_gen = lambda Wx: np.load(obsPK_dir+'W%s_cat_z0213_ra_dec_weight_z_ugriz_SDSSr_SDSSz.npy'%(Wx)) #columns: ra, dec, z_peak, weight, MAG_u, MAG_g, MAG_r, MAG_iy, MAG_z, r_SDSS, z_SDSS


########## cosmologies #######################
# growth factor
H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV))
# luminosity distance Mpc
DL = lambda z: (1+z)*c*quad(H_inv, 0, z)[0]
# use interpolation instead of actual calculation, so we can operate on an array
z_arr = linspace(0.1, 1.4, 1000)
DL_arr = array([DL(z) for z in z_arr])
DL_interp = interpolate.interp1d(z_arr, DL_arr)
# find the rest magnitude at the galaxy, from observed magnitude cut
M_rest_fcn = lambda M_obs, z: M_obs - 5.0*log10(DL_interp(z)) - 25.0

def maskGen (Wx, sigma_pix=0, z=0.4):
	'''generate mask using galn (galaxy count) map
	sigma_pix is the smoothing scale of the mask in
	unit of pixels
	z should be the lower bound for the convergence map.
	'''
	galn = galnGen_hi(Wx, z=z)
	mask = ones(shape=galn.shape)
	#mask = zeros(shape=galn.shape)
	#mask[25:-25,25:-25] = 1
	idx = where(galn<0.5)
	mask[idx] = 0
	mask_smooth = WLanalysis.smooth(mask, sigma_pix)
	return mask_smooth

def Wcircle (arcmin=2.0, PPA=PPA512):
	'''create a circular mask, =1 for within 2 arcmin, =0 for outside
	'''
	isize = int(PPA*2*arcmin)+1
	if isize%2 == 0:
		isize += 1 #make an odd size, so the middle one can center at the peak
	mask_circle = zeros (shape=(isize, isize))
	y, x = np.indices((isize, isize))
	center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
	r = np.hypot(x - center[0], y - center[1])/PPA
	mask_circle[where(r<arcmin)]=1
	return mask_circle, isize/2

def quicktest(Wx):
	'''Check if peaks in kmap is also peaks in bmap, so somehow 
	peaks leak into bmode..
	'''
	bmap = bmodeGen(Wx, z=z_hi)
	kmap = kmapGen(Wx, z=z_hi)
	ipeak_mat = WLanalysis.peaks_mat(kmap)
	ipeak_matb = WLanalysis.peaks_mat(bmap)
	imask = maskGen (Wx, z=z_lo)
	ipeak_mat[where(imask==0)]=nan
	ipeak_matb[where(imask==0)]=nan
	print '[W%i], kmap peaks: %i, bmap peaks: %i, overlapping peaks: %i, bmap-kmap peaks: %i'%(Wx, sum(~isnan(ipeak_mat)), sum(~isnan(ipeak_matb)), sum(~isnan(ipeak_mat+ipeak_matb)), sum(~isnan(ipeak_matb))-sum(~isnan(ipeak_mat)))

def PeakGaln (Wx, z_lo=0.85, z_hi='1.3_lo', arcmin=2.0, noise=False, Bmode=False):
	'''For a map(kappa or bmode), find peaks, and # gal fall within
	arcmin of that peak.
	'''
	#print 'noise', noise, Wx
	mask_circle, o = Wcircle(arcmin=arcmin)
	if Bmode:
		kmap = bmodeGen(Wx, z=z_hi)
	else:
		kmap = kmapGen(Wx, z=z_hi)
	ipeak_mat = WLanalysis.peaks_mat(kmap)
	imask = maskGen (Wx, z=z_lo)
	ipeak_mat[where(imask==0)]=nan
	igaln = galnGen_lo(Wx, z=z_lo)
	if noise:
		idx_all = where((imask==1)&isnan(ipeak_mat))
		sample = randint(0,len(idx_all[0])-1,sum(~isnan(ipeak_mat)))
		idx = array([idx_all[0][sample],idx_all[1][sample]])
	else:
		idx = where(~isnan(ipeak_mat)==True)
	kappaGaln_arr = zeros(shape=(len(idx[0]),2))
	for i in range(len(idx[0])):
		x, y = idx[0][i], idx[1][i]
		kappaGaln_arr[i,0] = kmap[x, y]
		kappaGaln_arr[i,1] = sum(igaln[x-o:x+o+1, y-o:y+o+1]*mask_circle)
	return kappaGaln_arr.T

def PeakPos (Wx, z_lo=0.6, z_hi='0.6_lo',noise=False, Bmode=False):
	'''For a map(kappa or bmode), find peaks, and its(RA, DEC)
	return 3 columns: [kappa, RA, DEC]
	'''
	#print 'noise', noise, Wx
	if Bmode:
		kmap = bmodeGen(Wx, z=z_hi)
	else:
		kmap = kmapGen(Wx, z=z_hi)
	ipeak_mat = WLanalysis.peaks_mat(kmap)
	imask = maskGen (Wx, z=z_lo)
	ipeak_mat[where(imask==0)]=nan #get ipeak_mat, masked region = nan
	if noise: #find the index for peaks in noise map
		idx_all = where((imask==1)&isnan(ipeak_mat))
		sample = randint(0,len(idx_all[0])-1,sum(~isnan(ipeak_mat)))
		idx = array([idx_all[0][sample],idx_all[1][sample]])
	else:#find the index for peaks in kappa map
		idx = where(~isnan(ipeak_mat)==True)
	kappaPos_arr = zeros(shape=(len(idx[0]),3))#prepare array for output
	for i in range(len(idx[0])):
		x, y = idx[0][i], idx[1][i]#x, y
		kappaPos_arr[i,0] = kmap[x, y]
		x = int(x-sizes[Wx-1]/2)+1
		y = int(y-sizes[Wx-1]/2)+1
		x /= PPR512# convert from pixel to radians
		y /= PPR512
		kappaPos_arr[i,1:] = WLanalysis.gnom_inv((y, x), centers[Wx-1])
	return kappaPos_arr.T

def hist_galn (allfield, kmin=-0.04, kmax=0.12, bins=10):
	'''
	make a histogram, for each kappa bin, the average gal#, and std
	allfield = [kappa_arr, galn_arr]
	Output: [kappa, mean, std]
	'''
	kappa_arr, galn_arr = allfield
	edges = linspace(kmin, kmax, bins+1)
	hist_arr = zeros(shape=(bins,3)) # mean, std
	for i in range(bins):
		#print i
		igaln = galn_arr[where((kappa_arr>edges[i])&(kappa_arr<edges[i+1]))]
		hist_arr[i,0]=0.5*(edges[i]+edges[i+1])
		hist_arr[i,1]=mean(igaln)
		hist_arr[i,2]=std(igaln)
	return hist_arr.T

def collect_allfields (z_lo=0.85, z_hi='1.3_lo', arcmin=2.0, noise=False, kmin=-0.04, kmax=0.12, bins=10, Bmode=False):
	'''using grid (not catalgue),
	collect the kappa arr and galn arr for all 4 fields
	'''
	kappaGaln_arr=array([PeakGaln(Wx, z_lo=z_lo, z_hi=z_hi, arcmin=arcmin, noise=noise, Bmode=Bmode) for Wx in range(1,5)])
	kappa_arr = concatenate([kappaGaln_arr[i,0] for i in range(4)])
	galn_arr = concatenate([kappaGaln_arr[i,1] for i in range(4)])
	return kappa_arr, galn_arr

def neighbor_index(ra0, dec0, ra_arr, dec_arr, R=2.0):
	'''find the index of ra_arr, dec_arr for galaxies 
	within R arcmin of (ra0, dec0)
	note: ra0, dec0, ra_arr, dec_arr are all in degrees, white R in arcmin
	return list of index
	'''
	idx_square = where( (abs(dec_arr-dec0)<(1.2*R/60.0))& (abs(ra_arr-ra0)<(1.2*R/60.0)) )
	f = WLanalysis.gnom_fun((ra0,dec0))
	x_rad, y_rad = f((ra_arr[idx_square], dec_arr[idx_square]))
	dist = sqrt(x_rad**2+y_rad**2) #in unit of radians
	idx_dist = where(dist<radians(R/60.0))
	idx = idx_square[0][idx_dist]
	return idx, dist[idx_dist]
	
##################### MAG_z to M100 ##########
L_Lsun1 = lambda MAG_z, rminusz: 10**(-0.4*MAG_z+1.863+0.444*rminusz)#from mag
def L_Lsun_VO(M): 
	if M<1e19:
		out = 1.23e10*(M/3.7e9)**29.78*(1+(M/3.7e9)**(29.5*0.0255))**(-1.0/0.0255)/h**2
	else:
		out = 1.23e10*(M/3.7e9)**(29.78-29.5)/h**2
	return out
L_Lsun_CM = lambda M: 4.4e11*(M/1e11)**4.0*(0.9+(M/1e11)**(3.85*0.1))**(-1.0/0.1)

datagrid_VO = np.load(obsPK_dir+'Mhalo_interpolator_VO.npy')
Minterp = interpolate.CloughTocher2DInterpolator(datagrid_VO[:,:2],datagrid_VO[:,2])
Mminfun = lambda M, MAG_z, rminusz: L_Lsun_VO(M)-L_Lsun1(MAG_z, rminusz)
def findM(MAG_z, rminusz):
	try:
		x = op.brentq(Mminfun, 1e9, 1e40, args=(MAG_z, rminusz))
	except Exception:
		print 'have to use interpolation'
		x = Minterp(MAG_z, rminusz)
	fun = Mminfun(x, MAG_z, rminusz)
	if abs(fun) > 1:
		print 'abs(fun) > 1 with: MAG_z, r-z =', MAG_z, rminusz
	return x, fun
####### test Minterp, extent=(-25.1,-14.4,-5.6,6.3) - pass!#######
#randz = rand(10)*(25.1-14.4)-25.1
#randrminusz = rand(10)*(6.3-5.6)-5.6
#for i in range(10):
	#M_true, M_err = findM(randz[i], randrminusz[i])
	#M_interp = Minterp(randz[i], randrminusz[i])
	#print '%.2f\t%.2f\t%.2f'%(M_interp/M_true-1, M_err, Mminfun(M_interp,randz[i], randrminusz[i])/L_Lsun1(randz[i], randrminusz[i]))
##################################################################


def cat_galn_mag(Wx, z_lo=0.6, z_hi='0.6_lo', R=3.0, noise=False, Bmode=False):
	'''updated 2014/12/09
	First open a kappa map, get a list of peaks (with RA DEC), then open the catalogue for 
	all galaxies, and for each peak, find galaxies within R arcmin of the peak. Document the following values:
	1) identifier for a peak (which links to another file with identifier, kappa_peak, ra, dec)
	2) ra, dec
	3) redshift
	4) r_SDSS-z_SDSS, MAG_z, for finding L_k
	5) MAG_i in rest frame, for galaxie cut
	6) halo mass got from interpolator, using tablulated values (r-z, M_z) -> M_halo
	
	older version: return a list of peaks, with colums 0) identifier, 1) kappa, 2) mag_i, 3) z_peak
	'''
	print Wx
	kappa_arr, peak_ras, peak_decs= PeakPos(Wx, z_lo=z_lo, z_hi=z_hi, noise=noise, Bmode=Bmode)
	#icat = cat_gen(Wx)
	#idx = where(icat[:,-1]<z_lo)#older version, now no cut on z anymore
	#ra_arr, dec_arr, mag_arr, z_arr = icat[idx].T
	ra_arr, dec_arr, z_arr, weight, MAG_u, MAG_g, MAG_r, MAG_iy, MAG_z, r_SDSS, z_SDSS = cat_gen(Wx).T
	def loop_thru_peaks(i):
		'''for each peak, find the galaxies within R, then record their mag, z, Mhalo, etc.
		return colums [identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance from peak]
		'''
		print 'peak#',i
		ra0, dec0 = peak_ras[i], peak_decs[i]
		idx, dist = neighbor_index(ra0, dec0, ra_arr, dec_arr, R=R)
		# here shift magnitude to rest frame, calculate halo mass
		SDSSr_rest = M_rest_fcn(r_SDSS[idx], z_arr[idx])
		SDSSz_rest = M_rest_fcn(z_SDSS[idx], z_arr[idx])
		MAG_z_rest = M_rest_fcn(MAG_z[idx], z_arr[idx])
		MAG_i_rest = M_rest_fcn(MAG_iy[idx], z_arr[idx])
		rminusz = SDSSr_rest - SDSSz_rest		
		M_arr = array([findM(SDSSz_rest[j], rminusz[j]) for j in range(len(MAG_z_rest))])
		M_arr[where(abs(r_SDSS[idx])==99)]=0#set the bad data to 0
		M_arr[where(abs(z_SDSS[idx])==99)]=0#set the bad data to 0
		ipeak_halo_arr = concatenate([i*ones(len(idx))+0.1*Wx, ra_arr[idx], dec_arr[idx], z_arr[idx], SDSSr_rest, SDSSz_rest, MAG_i_rest, M_arr[:,0], dist, weight[idx]]).reshape(9,-1)
		return ipeak_halo_arr
	print 'total peaks:',len(kappa_arr)
	all_peaks_mag_z = map(loop_thru_peaks, range(len(kappa_arr)))
	return concatenate(all_peaks_mag_z,axis=1)


################## kappa projection 2014/12/14 ##############
rho_cz = lambda z: rho_c0*(OmegaM*(1+z)**3+(1-OmegaM))#critical density of universe at z
Rvir_fcn = lambda M, z: (M*M_sun/(4.0/3.0*pi*200*rho_cz(z)))**0.3333
rad2arcmin = lambda distance: degrees(distance)*60.0

def Gx_fcn (x, cNFW=5.0):
	if x < 1:
		out = 1.0/(x**2-1.0)*sqrt(cNFW**2-x**2)/(cNFW+1.0)+1.0/(1.0-x**2)**1.5*arccosh((x**2+cNFW)/x/(cNFW+1.0))
	elif x == 1:
		out = sqrt(cNFW**2-1.0)/(cNFW+1.0)**2*(cNFW+2.0)/3.0
	elif 1 < x <= cNFW:
		out = 1.0/(x**2-1.0)*sqrt(cNFW**2-x**2)/(cNFW+1.0)-1.0/(x**2-1.0)**1.5*arccos((x**2+cNFW)/x/(cNFW+1.0))
	elif x > cNFW:
		out = 0
	return out

def kappa_proj (z_fore, M100, ra_fore, dec_fore, cNFW=5.0):
	'''return a function, for certain foreground halo, 
	calculate the projected mass between a foreground halo and a background galaxy pair.
	'''
	f = 1.043#=1.0/(log(1+cNFW)-cNFW/(1+cNFW)) with cNFW=5.0
	Mvir = M100/1.227#cNFW = 5, M100/Mvir = 1.227
	Rvir = Rvir_fcn(Mvir, z)#cm
	two_rhos_rs = Mvir*M_sun*f*cNFW**2/(2*pi*Rvir**2)#cgs, see LK2014 footnote
	xy_fcn = WLanalysis.gnom_fun((ra_fore, dec_fore))
	Dl = DL(z_fore)/(1+z_fore)**2 # D_angular = D_luminosity/(1+z)**2
	Dl_cm = Dl*3.08567758e24
	theta_vir = Rvir/Dl_cm

	def kappa_proj_fcn (z_back, ra_back, dec_back):
		Ds = DL(z_back)/(1+z_back)**2
		Dls = Ds - Dl
		DDs = Ds/(Dl*Dls)/3.08567758e24# 3e24 = 1Mpc/1cm
		SIGMAc = (c*1e5)**2/4.0/pi/Gnewton*DDs
		x_rad, y_rad = xy_fcn((ra_back, dec_back))
		theta = sqrt(x_rad**2+y_rad**2)
		x = cNFW*theta/theta_vir
		Gx = Gx_fcn(x, cNFW)
		kappa_p = two_rhos_rs/SIGMAc*Gx
		return kappa_p
	return kappa_proj_fcn


def MassProj (gridofdata, zcut, R = 3.0, sigmaG=1.0):
	'''For one peak, I try to get a projected kappa from foreground halos. z>zcut are used for kappa map, z<zcut are foreground halos.
	steps:
	1) cut galaxies to background & foreground by zcut
	2) shoot light rays to each background galaxy, find kappa at that position
	3) smooth within R, find kappa_proj at peak location
	4) output: index for foreground galaxies with non zero contribution to kappa, zero contributions can be due to problematic foreground galaxie with no magnitude data.
	5) note, everything need to be weighted by CFHT weight
	'''
	idx_dist = where(degrees(gridofdata[-2])<R/60.0)[0]
	identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance, weight = gridofdata[:,idx_dist]
	idx_fore = where(redshift<zcut)[0]
	if len(idx_fore)==0: # no foreground galaxies
		return [], [], []
	else:
		idx_back = where(redshift>=zcut)[0]
		
		### weight using gaussian wintow
		weight_arr = exp(-rad2arcmin(distance[idx_back])**2.0/(2*pi*sigmaG**2.0))*weight[idx_back]
		weight_arr /= sum(weight_arr)
		
		###### kappa_arr.shape = [ngal_fore, ngal_back]
		kappa_arr = zeros(shape=(len(idx_fore),len(idx_back)))
		for j in range(len(idx_fore)):# foreground halo count
			jidx = idx_fore[j]
			z_fore, M, ra_fore, dec_fore = redshift[jidx], M_halo[jidx], ra[jidx], dec[jidx]
			ikappa_proj = kappa_proj (z_fore, M, ra_fore, dec_fore)	
			i = 0
			for iidx_back in idx_back:
				kappa_arr[j,i]=ikappa_proj(redshift[iidx_back], ra[iidx_back], dec[iidx_back])
				i+=1
		kappa_arr[isnan(kappa_arr)]=0
		icontribute = sum(kappa_arr*weight_arr, axis=1)#sum over back ground galaxies
		idx_nonzero=nonzero(icontribute)[0]
		ikappa = sum(icontribute)
		icontribute/=ikappa
		return idx_dist[idx_fore[idx_nonzero]], icontribute[idx_nonzero], ikappa*ones(len(idx_nonzero))

################################################
################ operations ####################
################################################
zcenters = arange(0.225, 1.3, 0.05)#center of z bins from CFHT
zbins = linspace(0.2,1.3,23)#edges
zPDF = array([ 0.45445094,  0.80881598,  0.93470199,  0.76456038,  1.10499822,
        0.8803627 ,  1.1195881 ,  1.50167501,  1.48711827,  1.60911659,
        1.47213078,  1.31645756,  1.2022788 ,  1.76004064,  1.00095837,
        1.08889524,  0.97712418,  0.91611398,  0.57378752,  0.31843705,
        0.53337544,  0.17501226])# = dP/dz, normed in between z=0.2-1.3

zPDF_normed = lambda zcut: zPDF[:where(zcenters<=zcut)[0][-1]+1]/sum(zPDF[:where(zcenters<=zcut)[0][-1]+1])

	
def Nhalo_vs_kappa (icontri_arr, iz_arr, izPDF):#iMhalo_arr
	'''(1) count from the most contribution halo, til get 50% of the contribution.
	(2) for a redshift PDF, normalized to N gals, for each redshift bins, assume sqrt(N) noise,
	find peaks that have SNR > 3, say that's the # of peaks.
	return: (Nhalo, Nzpeak)
	'''
	## use redshift to find clusters
	NPDF = len(iz_arr)*izPDF/sum(izPDF)
	ihist = histogram(iz_arr, bins=zbins[:len(izPDF)+1])[0]
	SNR = (ihist-NPDF) / sqrt(NPDF)
	iNclusters = sum(SNR>=3)
	
	## use galaxies, to find # of galaxies needed to contribute to largest mass
	icontri_arr /= sum(icontri_arr)
	iNgals = sum(cumsum(sort(icontri_arr)[::-1])<0.5)+1
	return iNclusters, iNgals

if cluster_counts:
	R, zcut, noise = 3.0, 0.7, True
	Rcut = 3.0
	
	halo_arr = np.load (obsPK_dir+'Halos_IDziM_DistContri_k4_kB_zcut%s_R%s_noise%s.npy'%(zcut, R, noise))

	def idxcuts(halo_arr):
		IDs, z_arr, MAGi_arr, Mhalo_arr, d_arr, contri_arr, \
			kappaP_arr, kappaConv_arr = halo_arr		
		idx_cut = where((MAGi_arr > -24) & (MAGi_arr < -18) & (Mhalo_arr < 5.3e15) &
			(rad2arcmin(d_arr) < Rcut) & (kappaP_arr < 1.0))[0]
		return idx_cut

	idx_cut = idxcuts(halo_arr)
	IDs, z_arr, MAGi_arr, Mhalo_arr, d_arr, contri_arr, \
		kappaP_arr, kappaConv_arr = halo_arr[:,idx_cut]

	uniqueID = unique(IDs)
	izPDF = zPDF_normed(zcut)
	def Nhalo_count(i):#for i in randint(0,11931,20):
		print i
		iidx = where(IDs==uniqueID[i])[0]
		icontri_arr, iz_arr = contri_arr[iidx], z_arr[iidx]	
		iNclusters, iNgals = Nhalo_vs_kappa(icontri_arr, iz_arr, izPDF)
		return uniqueID[i], kappaP_arr[iidx[0]], kappaConv_arr[iidx[0]], iNclusters, iNgals

	#all_Nhalos = map(Nhalo_count, range(len(uniqueID)))
	#save(obsPK_dir+'ClusterCounts_ID_k4_kB_Ncluster_Ngal_zcut%s_R%s_noise%s.npy'%(zcut, Rcut, noise), all_Nhalos)

if halo_plots:
	R = 3.0
	zcut = 0.6
	all_Nhalos = load(obsPK_dir+'ClusterCounts_ID_k4_kB_Ncluster_Ngal_zcut%s_R3.0_noiseFalse.npy'%(zcut))
	all_Nhalos_noise = load(obsPK_dir+'ClusterCounts_ID_k4_kB_Ncluster_Ngal_zcut%s_R3.0_noiseTrue.npy'%(zcut))
	
	ID, kappaP, kappaConv, Ncluster, Nhalo = array(all_Nhalos).T
	nID, nkappaP, nkappaConv, nNcluster, nNhalo = array(all_Nhalos_noise).T

	########### 2dhist = scatter plot, kappaP vs kappaConv##############
	#figure()
	#hist2d(kappaP, kappaConv,range=((0,0.0015),(-0.05,0.15)), bins=20)
	#xlabel('kappa_project (from foreground halos)')
	#ylabel('kappa_convergence (using background galaxies)')
	#coeff, P = stats.spearmanr(kappaP, kappaConv)
	#title('zcut=%s, coeff=%.5f, P=%.5f'%(zcut,coeff,P))
	#colorbar()
	#savefig(plot_dir+'conv_vs_proj_zcut%s.jpg'%(zcut))
	#close()
	####################################################################
	
	######### Yang 2011 Fig.5 ####################
	#Nbin_edges = linspace(0.5, 8.5, 9)
	#kappa_arr = [kappaP, kappaConv]
	#nkappa_arr = [nkappaP, nkappaConv]
	#sP, sC = std(kappaP), std(kappaConv)
	#cuts = ([[-inf, sP],[sP, 3*sP],[3*sP, inf]],[[-inf, sC],[sC, 3*sC],[3*sC, inf]])
	#f=figure(figsize=(12,8))
	#title_arr = [['low (proj)','med (proj)','hi (proj)'],['low (conv)','med (conv)','hi (conv)']]
	#for i in range(2):
		#for j in range(3):
			#x0, x1 = cuts[i][j]
			#idxS = where((kappa_arr[i]<x1)& (kappa_arr[i]>x0))[0]
			#idxN = where((nkappa_arr[i]<x1)& (nkappa_arr[i]>x0))[0]
			##Nhalo[idxS], Ncluster[idxS]
			#ax=f.add_subplot(2,3,j+1+i*3)
			
			#ax.hist(Ncluster[idxS], bins=Nbin_edges, histtype='step',label='peaks',normed=True)
			#ax.hist(Ncluster[idxN],bins=Nbin_edges, histtype='step',label='rnd. direction',normed=True)

			##hist(Nhalo[idxS], bins=Nbin_edges, histtype='step',label='peaks',normed=True)
			##hist(nNhalo[idxN],bins=Nbin_edges, histtype='step',label='rnd. direction',normed=True)
			#ax.set_title(title_arr[i][j])
			#if i==0 and j==0:
				#ax.legend(fontsize=10)
			#if i == 1:
				##xlabel('N_halo')
				#ax.set_xlabel('N_cluster (SNR>3 in redshift bins)')
			#if j == 0:
				#ax.set_ylabel('Num. peaks')
	#plt.subplots_adjust(wspace=0.25,hspace=0.25)
	##savefig(plot_dir+'Nhalo_Npeaks_zcut%s.jpg'%(zcut))
	#savefig(plot_dir+'Ncluster_Npeaks_zcut%s.jpg'%(zcut))
	#close()
	###############################################
	
	######## look at where I found clusters #########
	idx_cluster = nonzero(Ncluster)
	f=figure(figsize=(8,5))
	subplot(121)
	hist(kappaP[idx_cluster], range=(0,0.0015), histtype='step',label='clusters',normed=True)
	hist(kappaP, histtype='step', range=(0,0.0015), label='all peaks',normed=True)
	legend(fontsize=10)
	xlabel('kappa_project')
	title('peak counts')
	matplotlib.pyplot.locator_params(nbins=4)
	subplot(122)
	hist(kappaConv[idx_cluster],range=(-0.05, 0.2),histtype='step',label='clusters',normed=True)
	hist(kappaConv, range=(-0.05, 0.2), histtype='step',label='all peaks',normed=True)
	xlabel('kappa_convergence')
	title('peak counts')
	matplotlib.pyplot.locator_params(nbins=4)
	savefig(plot_dir+'PeakCounts_withCluster_zcut%s.jpg'%(zcut))
	close()

if project_mass:
	R=3.0
	zcut=0.6	
	noise=False
	#for znoise in [[z, noise] for z in (0.5, 0.6, 0.7) for noise in (True, False)]:
		#zcut, noise = znoise
	#zcut = float(sys.argv[1])
	#noise = bool(int(sys.argv[2]))
	
	kappa_list = np.load(obsPK_dir+'AllPeaks_kappa_raDec_zcut%.1f.npy'%(zcut))
	print 'got files'
	## columns: kappa, ra, dec
	alldata = np.load(obsPK_dir+'peaks_IDraDecZ_MAGrziMhalo_dist_weight_zcut%.1f_R%s_noise%s.npy'%(zcut, R, noise))
	## columns: identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance, weight
	
	ids = alldata[0, sort(np.unique(alldata[0], return_index=True)[1])]#all the identifiers

	print 'len(ids)',len(ids)
	def halo_contribution(i):#for i in randint(0,11931,20):
		print zcut, noise, i
		iidx = where(alldata[0]==ids[i])[0]
		oldgrid = alldata[:, iidx]
		idx_fore, icontribute, ikappa = MassProj (oldgrid, zcut)
		if len(idx_fore)==0:
			return nan*zeros(shape=(8,1))
		else:
			newgrid = oldgrid[:, idx_fore]
			identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance, weight = newgrid
			newarr = array([identifier, redshift, MAG_iy_rest, M_halo, distance, icontribute, ikappa, kappa_list[0,i]*ones(len(ikappa))])# things I need for final analysis
			return newarr
	halo_fn = obsPK_dir+'Halos_IDziM_DistContri_k4_kB_zcut%s_R%s_noise%s'%(zcut, R, noise)
	
	pool = MPIPool()
	all_halos = pool.map(halo_contribution, range(len(ids)))
	all_halos = concatenate(all_halos, axis=1)
	#np.save(halo_fn,all_halos)
	

#################################################################

if list_peaks_cat:
	'''create a list of peaks, for all peaks (in 4 fields), into -
	columns: identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance, weight
	'''
	R = 3.0
	for znoise in [[z_lo, noise] for z_lo in (0.5, 0.6, 0.7) for noise in (True, False)]:
		z_lo, noise = znoise
		z_hi = '%s_hi'%(z_lo)
		print 'z_lo, noise, R:',',', z_lo,',', noise,',', R
		fn = obsPK_dir+'peaks_IDraDecZ_MAGrziMhalo_dist_zcut%s_R%s_noise%s.npy'%(z_lo, R, noise)
		#columns: identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance, weight
		seed(int(z_lo*10+R*100))	
		temp_arr = [cat_galn_mag(Wx, z_lo=z_lo, z_hi=z_hi, R=R, noise=noise) for Wx in range(1,5)]
		np.save(fn, concatenate(temp_arr,axis=1))
		### the following block creates the catalogue for peaks [kappa, RA, DEC]
		# all_peaks = [PeakPos(Wx, z_lo=z_lo, z_hi=z_hi, noise=noise) for Wx in range(1,5)]
		# np.save(obsPK_dir+'AllPeaks_kappa_raDec_zcut%s.npy'%(z_lo), concatenate(all_peaks, axis=1))
		#############################################################


print 'done-done-done'