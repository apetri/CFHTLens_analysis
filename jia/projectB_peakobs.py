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
from scipy import interpolate
from scipy.integrate import quad
import scipy.optimize as op

list_peaks_cat = 0 #! generate a list of galaxies for all peaks
update_mag_all = 0 #! make a list of galaxie catagues with useful quantities
project_mass = 1
#junk routines below
update_mag_i = 0
plot_galn_vs_kappa_hist = 0
do_hist_galn_magcut = 0
########### constants ######################
z_lo = 0.6
z_hi = z_hi = '%s_hi'%(z_lo)
obsPK_dir = '/Users/jia/CFHTLenS/obsPK/'
plot_dir = '/Users/jia/weaklensing/CFHTLenS/obsPK/plot/'
kmapGen = lambda i, z: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_KS_%s_sigmaG10.fit'%(i, z))
# This is smoothed galn
# galnGen = lambda i, z: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_galn_%s_hi_sigmaG10.fit'%(i, z))
galnGen_hi = lambda i, z: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_galn_%s_hi.fit'%(i, z))
galnGen_lo = lambda i, z: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_galn_%s_lo.fit'%(i, z))
bmodeGen = lambda i, z: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_Bmode_%s_sigmaG10.fit'%(i,z))
sizes = (1330, 800, 1120, 950)
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
PPR512=8468.416479647716
PPA512=2.4633625
c = 299792.458#km/s
Gnewton = 6.674e-8#cgs cm^3/g/s
H0 = 70.0
OmegaM = 0.25#0.3
OmegaV = 1.0-OmegaM
rho_c0 = 9.9e-30#g/cm^3
M_sun = 1.989e33#gram

############ functions #####################
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

cat_gen = lambda Wx: np.load(obsPK_dir+'W%s_cat_z0213_ra_dec_weight_z_ugriz_SDSSr_SDSSz.npy'%(Wx)) #columns: ra, dec, z_peak, weight, MAG_u, MAG_g, MAG_r, MAG_iy, MAG_z, r_SDSS, z_SDSS

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
	
#w,mag=np.load('/Users/jia/CFHTLenS/obsPK/W3_cat_z0213.npy')[:,[6,11]]
##################### MAG_z to M_halo ##########
h = 0.7
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
#cNFW_fcn = lambda zM: 9.0/(1.0+zM[0])*(zM[1]/1.3e13)**(-0.13)#Bullock2001
#cNFW_fcn = lambda zM: 10.0/(1.0+zM[0])*(zM[1]/1e13)**(-0.2)#Takada&Jain2003
cNFW_fcn = lambda zM: 11.0/(1.0+zM[0])*(zM[1]/1e13)**(-0.13)#Lin&Kilbinger2014
rho_mz = lambda z: OmegaM*rho_c0*(1+z)**3#done, unit g/cm^3
Rvir_fcn = lambda M, z: (M*M_sun/(4.0/3.0*pi*178*rho_mz(z)))**0.3333#set delta_c=178, unit=cm
#Rvir = lambda M, z: (M*M_sun/(4.0/3.0*pi*delta_c*rho_mz(z)))**0.3333# free delta c 

########## convert from M100 (get from Lk) to Mvir (needed for NFW) ############
ratio_r100_rvir = lambda gamma, c: (1.0/(c*gamma+1.0)+log(c*gamma+1.0)-1.0)/(1.0/(c+1.0)+log(c+1.0)-1.0) - gamma**3.0*100.0/178.0
ratio_M100_Mvir = lambda c: op.brentq(ratio_r100_rvir, 1e-6, 50, args=(c))**3.0*100.0/178.0
############################################################################## 

def Gx_fcn (x, cNFW):
	if x < 1:
		out = 1.0/(x**2-1.0)*sqrt(cNFW**2-x**2)/(cNFW+1.0)+1.0/(1.0-x**2)**1.5*arccosh((x**2+cNFW)/x/(cNFW+1.0))
	elif x == 1:
		out = sqrt(cNFW**2-1.0)/(cNFW+1.0)**2*(cNFW+2.0)/3.0
	elif 1 < x <= cNFW:
		out = 1.0/(x**2-1.0)*sqrt(cNFW**2-x**2)/(cNFW+1.0)-1.0/(x**2-1.0)**1.5*arccos((x**2+cNFW)/x/(cNFW+1.0))
	elif x > cNFW:
		out = 0
	return out


def kappa_proj (cNFW, z_fore, M100, ra_fore, dec_fore):
	'''return a function, for certain foreground halo, 
	calculate the projected mass between a foreground halo and a background galaxy pair.
	'''
	f = 1.0/(log(1+cNFW)-cNFW/(1+cNFW))
	Mvir = M100/ratio_M100_Mvir(cNFW)
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
		print '%.3f\t%.3f\t%.3f\t%.3e\t%.3f\t%.3e\t'%(z_back, ra_back, dec_back, two_rhos_rs, SIGMAc, Gx)
		return kappa_p
	return kappa_proj_fcn

rad2arcmin = lambda distance: degrees(distance)*60.0
def MassProj (radec0, kappa0, zcut, gridofdata, R = 2.0, MAGcut = -18, sigmaG=1.0):
	'''For a peak at (ra0, dec0) = radec0, I try to get a projected kappa from foreground halos.
	steps:
	1) cut galaxies to background & foreground by zcut
	2) shoot light rays to each background galaxy, find kappa at that position
	3) smooth within R, find kappa_proj at peak location
	4) output: kappa_proj, kappa_proj_i, which is the contributed kappa from ith galaxy
	5) note, everything need to be weighted by CFHT weight
	'''
	ra0, dec0 = radec0
	idx_dist = where(degrees(gridofdata[-2])<R/60.0)[0]
	identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance, weight = gridofdata[:,idx_dist]
	idx_fore = where(redshift<zcut)[0]
	idx_back = where(redshift>=zcut)[0]
	cNFW_arr = cNFW_fcn(array([redshift[idx_fore], M_halo[idx_fore]]))

	kappa_arr = zeros(shape=(len(idx_fore),len(idx_back)))
	weight_arr = exp(-rad2arcmin(distance[idx_back])**2/(2*pi*sigmaG))*weight[idx_back]
	weight_arr /= sum(weight_arr)
	for j in range(len(idx_fore)):# foreground halo count
		jidx = idx_fore[j]
		cNFW, z_fore, M, ra_fore, dec_fore = cNFW_arr[j], redshift[jidx], M_halo[jidx], ra[jidx], dec[jidx]
		ikappa_proj = kappa_proj (cNFW, z_fore, M, ra_fore, dec_fore)	
	#iidx_back=idx_back[50]
	#z_back, ra_back, dec_back = redshift[iidx_back], ra[iidx_back], dec[iidx_back]

		i = 0
		for iidx_back in idx_back:
			kappa_arr[j,i]=ikappa_proj(redshift[iidx_back], ra[iidx_back], dec[iidx_back])
			i+=1
	kappa_arr[isnan(kappa_arr)]=0
	icontribute = sum(kappa_arr*weight_arr, axis=1)
	ikappa = sum(icontribute)
	return idx_dist[idx_fore], icontribute, ikappa

################################################
################ operations ####################
################################################


if project_mass:
	zcut=0.7
	R=3.0
	noise=False
	kappa_list = load(obsPK_dir+'AllPeaks_kappa_raDec_zcut%s.npy'%(z_lo))
	fn = obsPK_dir+'peaks_IDraDecZ_MAGrziMhalo_dist_weight_zcut%s_R%s_noise%s.npy'%(z_lo, R, noise)
	alldata = load(fn)
	ids = alldata[0, sort(np.unique(alldata[0], return_index=True)[1])]
	seed(20)
	for i in randint(0,11931,10):#range(kappa_list.shape[-1]):
		print i
		radec0 = kappa_list[1:,i]
		iidx = where(alldata[0]==ids[i])[0]
		gridofdata = alldata[:, iidx]
		idx_fore, icontribute, ikappa = MassProj (radec0, kappa_list[0,i], zcut, gridofdata)

		idx_nonzero=nonzero(icontribute)[0]
		identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance, weight = gridofdata[:,idx_fore][:,idx_nonzero]
		icontribute = icontribute[idx_nonzero]
		
		figure(figsize=(8,7))
		subplot(221)
		scatter(log10(M_halo),log10(icontribute/ikappa),marker='x',s=5)
		title ('peak #%i, kappa = %.4f'%(i, ikappa))
		xlabel('log10(M_halo/M_sun)')
		ylabel('log10(kappa/kappa_tot)')

		subplot(222)
		scatter(log10(rad2arcmin(distance)), log10(icontribute/ikappa),marker='x',s=5)
		xlabel('log10(r) arcmin')

		subplot(223)
		hist(redshift)
		xlabel('z')

		savefig(obsPK_dir+'plot/sample_contribute_vs_Mhalo_%s.jpg'%(i))
		close()
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
		fn = '/Users/jia/CFHTLenS/obsPK/peaks_IDraDecZ_MAGrziMhalo_dist_zcut%s_R%s_noise%s.npy'%(z_lo, R, noise)
		#columns: identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance, weight
		seed(int(z_lo*10+R*100))	
		temp_arr = [cat_galn_mag(Wx, z_lo=z_lo, z_hi=z_hi, R=R, noise=noise) for Wx in range(1,5)]
		np.save(fn, concatenate(temp_arr,axis=1))
		### the following block creates the catalogue for peaks [kappa, RA, DEC]
		# all_peaks = [PeakPos(Wx, z_lo=z_lo, z_hi=z_hi, noise=noise) for Wx in range(1,5)]
		# save(obsPK_dir+'AllPeaks_kappa_raDec_zcut%s.npy'%(z_lo), concatenate(all_peaks, axis=1))
		#############################################################


