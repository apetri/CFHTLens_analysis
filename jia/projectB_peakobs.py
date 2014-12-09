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

plot_galn_vs_kappa_hist = 0
list_peaks_cat = 0
update_mag_i = 0
update_mag_all = 0
do_hist_galn_magcut = 0
z_lo = 0.6
z_hi = z_hi = '%s_hi'%(z_lo)
########### constants ######################
obsPK_dir = '/Users/jia/CFHTLenS/obsPK/'
plot_dir = '/Users/jia/weaklensing/CFHTLenS/plot/obsPK/'
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
c = 299792.458
H0 = 70.0
OmegaM = 0.25#0.3
OmegaV = 1.0-OmegaM

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

cat_gen_junk = lambda Wx: np.load('/Users/jia/CFHTLenS/obsPK/W%s_cat_z0213_ra_dec_magy_zpeak.npy'%(Wx))
cat_gen = lambda Wx: np.load('/Users/jia/CFHTLenS/obsPK/W%s_cat_z0213_ra_dec_weight_z_ugriz_SDSSr_SDSSz.npy'%(Wx))

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

def PeakPos (Wx, z_lo=0.85, z_hi='1.3_lo', arcmin=2.0, noise=False, Bmode=False):
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
	idx = idx_square[0][where(dist<radians(R/60.0))]
	return idx
	
#w,mag=np.load('/Users/jia/CFHTLenS/obsPK/W3_cat_z0213.npy')[:,[6,11]]
def cat_galn_mag(Wx, z_lo=0.85, z_hi='1.3_lo', R=2.0, noise=False, Bmode=False):
	'''return a list of peaks, with colums 0) identifier, 1) kappa, 2) mag_i, 3) z_peak
	'''
	print Wx
	kappa_arr, peak_ras, peak_decs= PeakPos(Wx, z_lo=z_lo, z_hi=z_hi, arcmin=R, noise=noise, Bmode=Bmode)
	icat = cat_gen(Wx)
	idx = where(icat[:,-1]<z_lo)
	ra_arr, dec_arr, mag_arr, z_arr = icat[idx].T
	def loop_thru_peaks(i):
		'''for each peak, find the galaxies within R, then record their mag, z
		return colums [identifier, kappa, mag_i, z_peak]
		'''
		ra0, dec0 = peak_ras[i], peak_decs[i]
		idx = neighbor_index(ra0, dec0, ra_arr, dec_arr, R=R)
		return concatenate([i*ones(len(idx))+0.1*Wx, kappa_arr[i]*ones(len(idx)), mag_arr[idx], z_arr[idx]]).reshape(4,-1)
	all_peaks_mag_z = map(loop_thru_peaks, range(len(kappa_arr)))
	return concatenate(all_peaks_mag_z,axis=1)

def hist_galn_magcut(z_lo, z_hi, R=2.0, mag_cut=-19, noise=False):
	'''This requires that the icat files exist already.
	This function reads the file, then cut out galaxies by magnitude, then count #galn for each peak.
	'''
	icat0 = np.load('/Users/jia/CFHTLenS/obsPK/peaks_mag_%s_lo_%s_R%s_noise%s.npy'%(z_lo, z_hi, R, noise))#colums 0) identifier, 1) kappa, 2) mag_i, 3) z_peak
	# exclude or include the -99, 99 galaxies?, or get those from other bands?
	icat = icat0[:,where((icat0[2]>-99)&(icat0[2]<99))].squeeze()
	mag_i, z_peak = icat[2:]
	mag_rest = M_rest_fcn(mag_i, z_peak)
	icat_cut = icat[:,where(mag_rest<mag_cut)].squeeze()
	sort_idx = argsort(icat_cut[0])
	unique_idx = nonzero(icat_cut[0,sort_idx[1:]]-icat_cut[0,sort_idx[:-1]])
	unique_idx = concatenate([[0],unique_idx[0]+1])#include 0 into index
	galn_arr = concatenate([unique_idx[1:]-unique_idx[:-1],[len(icat_cut[0])-unique_idx[-1]]])
	kappa_arr = icat_cut[1,sort_idx[unique_idx]]
	return galn_arr, kappa_arr

##################### MAG_z to M_halo tabulated values 2014/12##########
h = 0.7
L_Lsun1 = lambda MAG_z, rminusz: 10**(-0.4*MAG_z+1.863+0.444*rminusz)#from mag
L_Lsun_VO = lambda M: 1.23e10*(M/3.7e9)**29.78*(1+(M/3.7e9)**(29.5*0.0255))**(-1.0/0.0255)
L_Lsun_CM = lambda M: 4.4e11*(M/1e11)**4.0*(0.9+(M/1e11)**(3.85*0.1))**(-1.0/0.1)

import scipy.optimize as op
Mminfun = lambda M, MAG_z, rminusz: L_Lsun_VO(M)/h**2-L_Lsun1(MAG_z, rminusz)
def findM(MAG_z_rminusz):
	print MAG_z_rminusz
	out = op.root(Mminfun, 1e12, args=(MAG_z_rminusz[0], MAG_z_rminusz[1]), method='lm')#lm:9414; anderson:slow..;hybr:5443;broyden1: slow...
	return float(out.x),float(out.fun)
#MAG_z_rminusz_arr = array([[MAG_z, rminusz] for MAG_z in linspace(-19, -25, 100) for rminusz in linspace(-5,5, 100)])
#M_arr = array(map(findM, MAG_z_rminusz_arr))
#print sum(abs(M_arr[:,-1])<1)
#################################################################
################################################
################ operations ####################
################################################

####### get a list of peaks, with colums 0) identifier, 1) kappa, 2) mag_i, 3) z_peak
if list_peaks_cat:
	for z_lo in (0.5, 0.6, 0.7):
		z_hi = '%s_hi'%(z_lo)
		for noise in (True, False):
			for R in (1.0, 2.0, 3.0):
				print 'z_lo, noise, R:',',', z_lo,',', noise,',', R
				fn = '/Users/jia/CFHTLenS/obsPK/peaks_mag_%s_lo_%s_R%s_noise%s.npy'%(z_lo, z_hi, R, noise)
				if os.path.isfile(fn):
					######################################
					## 10/11/2014, process M_obs -> M_rest
					#icat = np.load('/Users/jia/CFHTLenS/obsPK/peaks_mag_%s_lo_%s_R%s_noise%s.npy'%(z_lo, z_hi, R, noise))
					#mag_i, z_peak = icat[-2:]
					#mag_rest = M_rest_fcn(mag_i, z_peak)
					######################################
					print 'skip'
					continue
				seed(int(z_lo*10+R*100))	
				a=concatenate([cat_galn_mag(Wx, z_lo=z_lo, z_hi=z_hi, R=R, noise=noise) for Wx in range(1,5)],axis=1)
				np.save(fn,a)
if do_hist_galn_magcut:
	print 'hi'
	mag_cut = -21
	height_arr = ['high', 'med', 'low']
	def idx_height (galn_arr, kappa_arr, height='med'):
		if height == 'low':
			idx = where(kappa_arr<0.03)[0]
		if height == 'med':
			idx = where((kappa_arr>0.03)&(kappa_arr<0.06))[0]
		if height == 'high':
			#idx = where(kappa_arr>0.06)[0]
			idx = where(kappa_arr<100)[0]
		return galn_arr[idx].squeeze()
	
	for height in height_arr:
		f=figure(figsize=(12, 8))
		i = 1
		for z_lo in (0.5, 0.6, 0.7):
			for R in (1.0, 2.0, 3.0):
				z_hi = '%s_hi'%(z_lo)
				galn_arr, kappa_arr = hist_galn_magcut(z_lo, z_hi, mag_cut = mag_cut,noise=False, R=R)
				galn_noise_arr, kappa_noise_arr = hist_galn_magcut(z_lo, z_hi, mag_cut = mag_cut, noise=True, R=R)
				ax=f.add_subplot(3,3,i)
				
				galn_peaks = idx_height (galn_arr, kappa_arr, height=height)
				galn_noise = idx_height (galn_noise_arr, kappa_noise_arr, height=height)
				ax.hist(galn_peaks, histtype='step', bins=20, label='peaks, %s'%(height))
				ax.hist(galn_noise, histtype='step', ls='dashed', bins=20, label='noise, %s'%(height))
				if i >6:
					ax.set_xlabel('# gal')
				if i in (1, 4, 7):
					ax.set_ylabel('# peaks')
				
				if i == 1:
					ax.set_title('%s peaks, M<%s'%(height, mag_cut))
				leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':10},loc=0, title='z=%s, R=%sarcmin'%(z_lo, R))
				leg.get_frame().set_visible(False)
					
				i+=1
		savefig(plot_dir+'hist_galn_magcut%s_%s_rand.jpg'%(mag_cut, height))
		close()
			

if update_mag_i:
	## 10/06/2014, replace Mag_i = -99 items with Mag_y values
	# very messy, one time use only, because of CFHT failed i filter
	ra_arr, dec_arr, Mag_y_arr = np.load('/Users/jia/CFHTLenS/catalogue/Mag_y.npy').T
	radecy = np.load('/Users/jia/CFHTLenS/catalogue/Mag_y.npy')[:,:2]
	radecy0 = radecy[:,0]**radecy[:,1]#trick to make use of in1d for 2d
	
	for i in range(1,5):
		print i
		icat = cat_gen(i)
		icat_new = icat.copy()
		idx = where(icat[:,2]==-99)[0]
		print 'bad M_i', len(idx)
		
		radec99 = icat[idx,:2]
		radec990 = radec99[:,0]**radec99[:,1]#trick to make use of in1d for 2d
		# find the index for the intersect arrays, and tested both are unique arrays, no repeating items
		idx_99= nonzero(np.in1d(radec990, radecy0))
		idx_y = nonzero(np.in1d(radecy0, radec990))
		print 'len(idx_99), len(idx_y)',len(idx_99[0]), len(idx_y[0])
		# sort the ra, dec, to match the 2 list, and get index
		idx_sorted99 = argsort(radec990[idx_99])
		idx_sortedy = argsort(radecy0[idx_y])
		
		# check
		# radec990[idx_99[0][idx_sorted99]]-radecy0[idx_y[0][idx_sortedy]]
		# pass - returns 0
		
		icat_new[idx[idx_99[0][idx_sorted99]],-2] = Mag_y_arr[idx_y[0][idx_sortedy]]
		print 'mag_y=-99, mag_y==99',sum(icat_new[:,-2]==-99),sum(icat_new[:,-2]==99)
		np.save('/Users/jia/CFHTLenS/obsPK/W%s_cat_z0213_ra_dec_magy_zpeak'%(i), icat_new)
##
if plot_galn_vs_kappa_hist:
	Wx=4
	z_lo, z_hi, arcmin = 0.85, '0.4_hi', 3
	#for z_lo in (0.85,):# 0.6, 1.3):
		#for z_hi in ('0.4_hi',):#'0.6_hi'):#'1.3_lo', 
			#for arcmin in (3,):#1.5, 3.0):# 2.0, 
				#print z_lo, z_hi, arcmin
				#allfield_peaks = collect_allfields(z_lo=z_lo, z_hi=z_hi, arcmin=arcmin)
				#allfield_noise = collect_allfields(z_lo=z_lo, z_hi=z_hi, arcmin=arcmin, noise=1)
				#allfield_bmode = collect_allfields(z_lo=z_lo, z_hi=z_hi, arcmin=arcmin, Bmode=1)
				#allfield_bmode_noise = collect_allfields(z_lo=z_lo, z_hi=z_hi, arcmin=arcmin, Bmode=1,noise=1)
				
	allfield_peaks = PeakGaln(Wx, z_lo=z_lo, z_hi=z_hi, arcmin=arcmin)
	allfield_noise = PeakGaln(Wx, z_lo=z_lo, z_hi=z_hi, arcmin=arcmin, noise=1)
	allfield_bmode = PeakGaln(Wx, z_lo=z_lo, z_hi=z_hi, arcmin=arcmin, Bmode=1)
	allfield_bmode_noise = PeakGaln(Wx, z_lo=z_lo, z_hi=z_hi, arcmin=arcmin, Bmode=1,noise=1)
	
	# (Wx, z_lo=0.85, z_hi='1.3_lo', arcmin=2.0, noise=False, Bmode=False)	
	hist_peaks = hist_galn(allfield_peaks)
	hist_noise = hist_galn(allfield_noise)
	hist_Bmode = hist_galn(allfield_bmode)
	hist_Bmode_noise = hist_galn(allfield_bmode_noise)
	errorbar(hist_peaks[0],hist_peaks[1],hist_peaks[2],color='r',label='Kmap-peaks')	
	errorbar(hist_noise[0],hist_noise[1],hist_noise[2],color='r',linestyle='--',label='Kmap-noise')	
	errorbar(hist_Bmode[0],hist_Bmode[1],hist_Bmode[2],color='b',label='Bmode-peaks')	
	errorbar(hist_Bmode_noise[0],hist_Bmode_noise[1],hist_Bmode_noise[2],color='b',linestyle='--',label='Bmode-noise')
	legend(fontsize=12)
	izhi=float(z_hi[:3])
	if izhi == 1.3:
		izhi=0
	title(r'$W%i\,R=%s\, arcmin,\, z_{lo}=[0,\,%s],\, z_{hi}=[%s,\,1.3]$'%(Wx, arcmin, z_lo, izhi))
	xlabel('Kappa')
	ylabel('# of galaxies')
	savefig(plot_dir+'W%i_galn_peaks_%sarcmin_zlo%s_zhi%s.jpg'%(Wx,arcmin, z_lo, z_hi))
	#savefig(plot_dir+'galn_peaks_%sarcmin_zlo%s_zhi%s.jpg'%(arcmin, z_lo, z_hi))
	close()
	
if update_mag_all:
	## 12/08/2014, code to: 
	## (1) replace Mag_i = -99 items with Mag_y values
	## (2) add ugriz bands to the catalogue
	## (3) convert from MegaCam to SDSS AB system
	color_cat = load(obsPK_dir+'CFHTdata_RA_DEC_ugriyz_2014-12-08T21-58-57.npy')
	RA, DEC, star_flag, weight, MAG_u, MAG_g, MAG_r, MAG_i, MAG_y, MAG_z = color_cat.T
	RADEC = RA+1.0j*DEC
	# merge i and y band, rid of the 99 values
	idx_badi = where(abs(MAG_i)==99)[0]
	MAG_iy = MAG_i.copy()
	MAG_iy[idx_badi]=MAG_y[idx_badi]
	# test # of bad magnitude in i, y, and iy 
	#array([sum(abs(arr)==99) for arr in (MAG_i, MAG_y, MAG_iy)])/7522546.0
	#[963311, 6562757, 3523]
	#[0.128, 0.872, 0.000468]
	
	### convert to SDSS ##############
	### r_SDSS=r_Mega +0.011 (g_Mega - r_Mega)
	### z_SDSS=z_Mega -0.099 (i_Mega - z_Mega)
	r_SDSS=MAG_r + 0.011*(MAG_g - MAG_r)
	z_SDSS=MAG_z - 0.099*(MAG_iy - MAG_z)
	# rz = r_SDSS - z_SDSS # should do after redshift
	idx_badrz = where(amax(abs(array([MAG_g, MAG_r, MAG_iy, MAG_z])), axis=0)==99)[0]
	r_SDSS[idx_badrz] = MAG_r[idx_badrz]
	z_SDSS[idx_badrz] = MAG_z[idx_badrz] # replace bad r_SDSS with MAG_r, in case it's caused by MAG_g
	##################################
	color_cat_reorder = array([weight, MAG_u, MAG_g, MAG_r, MAG_iy, MAG_z, r_SDSS, z_SDSS]).T
	for i in range(1,5):
		print i
		icat = cat_gen_junk(i) #ra, dec, mag_i, z_peak
		iradec = icat.T[0]+1.0j*icat.T[1]
		
		idx = where(in1d(RADEC, iradec)==True)[0]
		if idx.shape[0] != icat.shape[0]:
			print 'Error in shape matching'
		
		iRADEC = RADEC[idx]
		id1 = argsort(iradec)
		id2 = argsort(iRADEC)
			
		### test - the 2 arrays should be identical - pass!
		### iRADEC[id2] - iradec[id1]
		
		icat_new = concatenate([icat[id1][:,[0,1,3]], color_cat_reorder[idx[id2]]], axis=1)
		np.save(obsPK_dir+'W%s_cat_z0213_ra_dec_weight_z_ugriz_SDSSr_SDSSz'%(i), icat_new)
		# columns: ra, dec, z_peak, weight, MAG_u, MAG_g, MAG_r, MAG_iy, MAG_z, r_SDSS, z_SDSS
		
		### test 
		### a=icat[id1][:,-2]
		### b=icat_new[:,-4]
		### sum((a-b)==0) - pass!