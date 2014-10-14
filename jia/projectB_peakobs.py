##########################################################
### This code is for Jia's project B - try to find
### observational evidence of over density at peak location
### as discovered by Yang 2011.
### It does the following:
### 1) find PDF for # gal within 2 arcmin as fcn of peak
### hights
### 2) the same, as 1) but for random direction
### 3) future modification needed to include L-M conversion

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

########### constants ######################
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
OmegaM = 0.3
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

cat_gen = lambda Wx: np.load('/Users/jia/CFHTLenS/obsPK/W%s_cat_z0213_ra_dec_magy_zpeak.npy'%(Wx))

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

def hist_cat(z_lo, z_hi, mag_cut, R, noise=False):
	'''This requires that the icat files exist already.
	This function reads the file, then cut out galaxies by magnitude, then count #galn for each peak.
	'''
	icat = np.load('/Users/jia/CFHTLenS/obsPK/peaks_mag_%s_lo_%s_R%s_noise%s.npy'%(z_lo, z_hi, R, noise))#colums 0) identifier, 1) kappa, 2) mag_i, 3) z_peak
	# exclude or include the -99, 99 galaxies?, or get those from other bands?
	mag_i, z_peak = icat[2:]
	mag_rest = M_rest_fcn(mag_i, z_peak)
	icat_cut = icat[:,where(mag_rest<mag_cut)].squeeze()
	sort_idx = argsort(icat_cut[0])
	unique_idx = nonzero(icat_cut[0,sort_idx[1:]]-icat_cut[0,sort_idx[:-1]])
	unique_idx = concatenate([[0],unique_idx[0]+1])#include 0 into index
	galn_arr = concatenate([unique_idx[1:]-unique_idx[:-1],[len(icat_cut[0])-unique_idx[-1]]])
	kappa_arr = icat_cut[1,sort_idx[unique_idx]]
	return galn_arr, kappa_arr

################################################
################ operations ####################

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
#if hist_galn_magcut:
	

## 10/06/2014, replace Mag_i = -99 items with Mag_y values
if update_mag_i:
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