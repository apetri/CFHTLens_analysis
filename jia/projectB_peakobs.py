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
import sys, os

make_kappa_predict = 0
if make_kappa_predict:
	######## for stampede #####
	from emcee.utils import MPIPool
	obsPK_dir = '/home1/02977/jialiu/obsPK/'
else:
	######## for laptop #####
	obsPK_dir = '/Users/jia/CFHTLenS/obsPK/'
	plot_dir = obsPK_dir+'plot/'

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
OmegaM = 0.3#0.25#
OmegaV = 1.0-OmegaM
#rho_c0 = 9.9e-30#g/cm^3
M_sun = 1.989e33#gram
sigmaG_arr = (0.5, 1.0, 1.8, 3.5, 5.3, 8.9)
############################################
############ functions #####################
############################################

########### generate maps ##################
maskGen = lambda Wx, zcut, sigmaG: load(obsPK_dir+'maps/Mask_W%s_%s_sigmaG%02d.npy'%(Wx,zcut,sigmaG*10))

kmapGen = lambda Wx, zcut, sigmaG: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_KS_%s_hi_sigmaG%02d.fit'%(Wx,zcut,sigmaG*10))*maskGen(Wx, zcut, sigmaG)

bmodeGen = lambda Wx, zcut, sigmaG: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_Bmode_%s_sigmaG%02d.fit'%(Wx, zcut, sigmaG))*maskGen(Wx, zcut, sigmaG)

cat_gen = lambda Wx: np.load(obsPK_dir+'W%s_cat_z0213_ra_dec_redshift_weight_MAGi_Mvir_Rvir_DL.npy'%(Wx))
cat_gen_old = lambda Wx: np.load(obsPK_dir+'VO06mass/W%s_cat_z0213_ra_dec_redshift_weight_MAGi_Mvir_Rvir_DL.npy'%(Wx))
#columns: ra, dec, redshift, weight, i, Mhalo, Rvir, DL

##############################################
########## cosmologies #######################
##############################################

# growth factor
Hcgs = lambda z: H0*sqrt(OmegaM*(1+z)**3+OmegaV)*3.24e-20
H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV))
# luminosity distance Mpc
DC_integral = lambda z: c*quad(H_inv, 0, z)[0]
z_arr = linspace(0.1, 1.4, 1000)
DC_arr = array([DC_integral(z) for z in z_arr])
DC = interpolate.interp1d(z_arr, DC_arr)
DA = lambda z: DC(z)/(1.0+z)
DL = lambda z: DC(z)*(1.0+z)
# find the rest magnitude at the galaxy, from observed magnitude cut
#M_rest_fcn = lambda M_obs, z: M_obs - 5.0*log10(DL_interp(z)) - 25.0
##rho_cz = lambda z: rho_c0*(OmegaM*(1+z)**3+(1-OmegaM))#critical density
rho_cz = lambda z: 0.375*Hcgs(z)**2/pi/Gnewton

##############################################
##################### MAG_z to M100 ##########
##############################################

datagrid_VO = np.load(obsPK_dir+'Mhalo_interpolator_VO.npy')#Mag_z, r-z, M100, residual
Minterp = interpolate.CloughTocher2DInterpolator(datagrid_VO[:,:2],datagrid_VO[:,2])
#usage: Minterp(MAGz_arr, r-z_arr)

Rvir_fcn = lambda M, z: (M*M_sun/(4.0/3.0*pi*200*rho_cz(z)))**0.3333#in cm
rad2arcmin = lambda distance: degrees(distance)*60.0

##############################################
######### find RA DEC for peaks ##############
##############################################

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

	
#############################################################
################## kappa projection 2014/12/14 ##############
#############################################################

def Gx_fcn (x, cNFW):#=5.0):
	if x < 1:
		out = 1.0/(x**2-1.0)*sqrt(cNFW**2-x**2)/(cNFW+1.0)+1.0/(1.0-x**2)**1.5*arccosh((x**2+cNFW)/x/(cNFW+1.0))
	elif x == 1:
		out = sqrt(cNFW**2-1.0)/(cNFW+1.0)**2*(cNFW+2.0)/3.0
	elif 1 < x <= cNFW:
		out = 1.0/(x**2-1.0)*sqrt(cNFW**2-x**2)/(cNFW+1.0)-1.0/(x**2-1.0)**1.5*arccos((x**2+cNFW)/x/(cNFW+1.0))
	elif x > cNFW:
		out = 0
	return out

f = 1.043
c0, beta = 11, 0.13 # lin & kilbinger2014
def kappa_proj (Mvir, z_fore, x_fore, y_fore, z_back, x_back, y_back, cNFW=5.0):
	'''return a function, for certain foreground halo, 
	calculate the projected mass between a foreground halo and a background galaxy pair.
	'''
	######## updated next 2 lines to have a variable cNFW
	#cNFW = c0/(1+z_back)*(Mvir/1e13)**(-beta)
	f=1.0/(log(1.0+cNFW)-cNFW/(1.0+cNFW))# = 1.043 with cNFW=5.0
	Rvir = Rvir_fcn(Mvir, z_fore)
	two_rhos_rs = Mvir*M_sun*f*cNFW**2/(2*pi*Rvir**2)#cgs, see LK2014 footnote
	
	Dl_cm = 3.08567758e24*DA(z_fore)
	##3.08567758e24cm/Mpc### 	
	#SIGMAc = 1.07163e+27/DlDlsDs#(c*1e5)**2/4.0/pi/Gnewton=1.0716311756473212e+27
	##347.2916311625792=1.07163e+27/3.08567758e24
	SIGMAc = 347.29163*DC(z_back)*(1+z_fore)/(DC(z_fore)*(DC(z_back)-DC(z_fore)))
	theta = sqrt((x_fore-x_back)**2+(y_fore-y_back)**2)
	x = cNFW*theta/Rvir*Dl_cm
	Gx = Gx_fcn(x, cNFW)
	kappa_p = two_rhos_rs/SIGMAc*Gx
	return kappa_p

########## update halo mass using L12
#Mhalo_params_arr = [[12.520, 10.916, 0.457, 0.566, 1.53],
		    #[12.725, 11.038, 0.466, 0.610, 1.95],
		    #[12.722, 11.100, 0.470, 0.393, 2.51]]
##log10M1, log10M0, beta, sig, gamma

#redshift_edges=[[0, 0.48], [0.48,0.74], [0.74, 1.30]]

#master_ra, master_dec, w, M_star = genfromtxt('/Users/jia/CFHTLenS/catalogue/CFHTLens_2015-02-05T05-08-44.tsv', skip_header=1).T

#def Mstar2Mhalo (Mstar_arr, redshift_arr):
	#Mhalo_arr = zeros(len(Mstar_arr))
	#for i in range(3):
		#z0,z1 = redshift_edges[i]
		#log10M1, log10M0, beta, sig, gamma = Mhalo_params_arr[i]
		#print log10M1, log10M0, beta, sig, gamma 
		#Mhalo_fcn = lambda log10Mstar: log10M1+beta*(log10Mstar-log10M0)+10.0**(sig*(log10Mstar-log10M0))/(1+10.0**(-gamma*(log10Mstar-log10M0)))-0.5
		#idx = where((redshift_arr>z0)&(redshift_arr<=z1))[0]
		#Mhalo_arr[idx] = Mhalo_fcn(Mstar_arr[idx])
	#return Mhalo_arr
	
#for Wx in (4,):#range(1,5):
	#print Wx
	#icat = cat_gen_old(Wx).T
	#ra, dec, redshift, weight, MAGi, Mhalo, Rvir, DL = icat.copy()
	#idx = WLanalysis.update_values_by_RaDec(ra, dec, master_ra, master_dec)
	#iM_star = M_star[idx]
	#Mhalo_L12 = 10**Mstar2Mhalo(iM_star, redshift)
	#icat[5] = Mhalo_L12
	#icat[6] = Rvir_fcn(Mhalo_L12, redshift)
	#icat[7] = DL_interp(redshift)
	#save(obsPK_dir+'W%s_cat_z0213_ra_dec_redshift_weight_MAGi_Mvir_Rvir_DL.npy'%(Wx), icat[:,iM_star>-99].T)
	
############### halo mass using Guo 2010 ###############
#M0 = 10**11.4
#def minfcn (Mstar):
	#def root (Mhalo):
		#Mstar_Guo = Mhalo*0.129*((Mhalo/M0)**(-0.926)+(Mhalo/M0)**(0.261))**(-2.44)
		#return Mstar - Mstar_Guo
	#return root
	
#Mstar2Mhalo = lambda Mstar: op.bisect(minfcn(Mstar), 1e6, 1e19)
#Mstar_arr = logspace(7,13,10000)
#Mhalo_arr = array([Mstar2Mhalo(Mstar) for Mstar in Mstar_arr])
#Mstar_interp = interpolate.interp1d(concatenate([(-100,),Mstar_arr]), concatenate([(-100,),Mhalo_arr]))
##residuals = array([minfcn(Mstar_arr[i])(Mhalo_arr[i]) for i in range(10000)])
##master_ra, master_dec, w, M_star = genfromtxt('/Users/jia/CFHTLenS/catalogue/CFHTLens_2015-02-05T05-08-44.tsv', skip_header=1).T
#for Wx in range(1,5):
	#print Wx
	#icat = cat_gen_old(Wx).T
	#ra, dec, redshift, weight, MAGi, Mhalo, Rvir, DL = icat.copy()
	#idx = WLanalysis.update_values_by_RaDec(ra, dec, master_ra, master_dec)
	#iM_star = M_star[idx]
	#Mhalo_G10 = Mstar_interp(10**iM_star)
	#Mhalo_G10[Mhalo_G10>1e15]=1e15
	#icat[5] = Mhalo_G10
	#icat[6] = Rvir_fcn(Mhalo_G10, redshift)
	#icat[7] = DL_interp(redshift)
	#save(obsPK_dir+'W%s_cat_z0213_ra_dec_redshift_weight_MAGi_Mvir_Rvir_DL.npy'%(Wx), icat[:,iM_star>-99].T)
#############################################################
############ operations #####################################
#############################################################

if make_kappa_predict:
	from scipy.spatial import cKDTree
	zcut = 0.2	#this is the lowest redshift for backgorund galaxies. use 0.2 to count for all galaxies.
	r = 0.006	# 0.002 rad = 7arcmin, 
			#within which I search for contributing halos

	Wx = int(sys.argv[1])
	center = centers[Wx-1]
	icat = cat_gen(Wx).T

	ra, dec, redshift, weight, MAGi, Mhalo, Rvir, DL = icat
	## varying DL
	DL = DL_interp(redshift)
	#Mhalo[Mhalo>2e15] = 2e15#prevent halos to get crazy mass
	f_Wx = WLanalysis.gnom_fun(center)#turns to radians
	xy = array(f_Wx(icat[:2])).T

	idx_back = where(redshift>zcut)[0]
	xy_back = xy[idx_back]

	kdt = cKDTree(xy)
#nearestneighbors = kdt.query_ball_point(xy_back[:100], 0.002)
	def kappa_individual_gal (i):
		'''for individual background galaxies, find foreground galaxies within 20 arcmin and sum up the kappa contribution
		'''
		print i
		iidx_fore = array(kdt.query_ball_point(xy_back[i], r))	
		x_back, y_back = xy_back[i]
		z_back, DL_back = redshift[idx_back][i], DL[idx_back][i]
		ikappa = 0
		for jj in iidx_fore:
			x_fore, y_fore = xy[jj]
			jMvir, jRvir, z_fore, DL_fore = Mhalo[jj], Rvir[jj], redshift[jj], DL[jj]
			if z_fore >= z_back:
				kappa_temp = 0
			else:
				kappa_temp = kappa_proj (jMvir, jRvir, z_fore, x_fore, y_fore, DL_fore, z_back, x_back, y_back, DL_back, cNFW=5.0)
				if isnan(kappa_temp):
					kappa_temp = 0
			ikappa += kappa_temp
			
			if kappa_temp>0:
				theta = sqrt((x_fore-x_back)**2+(y_fore-y_back)**2)
				print '%i\t%s\t%.2f\t%.3f\t%.3f\t%.4f\t%.6f'%(i, jj,log10(jMvir), z_fore, z_back, rad2arcmin(theta), kappa_temp)	
		return ikappa

	#a=map(kappa_individual_gal, randint(0,len(idx_back)-1,5))
	step=2e3
	
	def temp (ix):
		print ix
		temp_fn = obsPK_dir+'temp/kappa_proj%i_%07d.npy'%(Wx, ix)
		if not os.path.isfile(temp_fn):
			kappa_all = map(kappa_individual_gal, arange(ix, amin([len(idx_back), ix+step])))
			np.save(temp_fn,kappa_all)
	pool = MPIPool()
	ix_arr = arange(0, len(idx_back), step)
	pool.map(temp, ix_arr)
	
	all_kappa_proj = concatenate([np.load(obsPK_dir+'temp/kappa_proj%i_%07d.npy'%(Wx, ix)) for ix in ix_arr])
	np.save(obsPK_dir+'kappa_predict_W%i.npy'%(Wx), all_kappa_proj)
	
#########################################################
####################### plotting correlation ############
#########################################################
make_predict_maps = 0
plot_predict_maps = 0
peak_proj_vs_lensing = 0
cross_correlate = 0

#kmap_predict_Gen = lambda Wx, sigmaG: np.load(obsPK_dir+'maps/r20arcmin_varyingcNFW_VO06/kmap_W%i_predict_sigmaG%02d.npy'%(Wx, sigmaG*10))
kmap_predict_Gen = lambda Wx, sigmaG: np.load(obsPK_dir+'maps/kmap_W%i_predict_sigmaG%02d.npy'%(Wx, sigmaG*10))

kmap_lensing_Gen = lambda Wx, sigmaG: WLanalysis.readFits(obsPK_dir+'maps/W%i_KS_1.3_lo_sigmaG%02d.fit'%(Wx, sigmaG*10))

bmode_lensing_Gen = lambda Wx, sigmaG: WLanalysis.readFits(obsPK_dir+'maps/W%i_Bmode_1.3_lo_sigmaG%02d.fit'%(Wx, sigmaG*10))

if make_predict_maps:
	for Wx in (2,):#range(1,5):#
	
		############### get catalogue
		sizes = (1330, 800, 1120, 950)
		print Wx
		isize = sizes[Wx-1]
		center = centers[Wx-1]

		
		icat = cat_gen(Wx).T #
		ra, dec, redshift, weight, MAGi, Mhalo, Rvir, DL = icat
		##########next 3 lines, cut at z=0.4 for source galaxies
		k = np.load(obsPK_dir+'kappa_predict_W%i.npy'%(Wx))
		idx_k = where(k>0)[0]
		k = k[idx_k]
		icat = icat[:,idx_k]
		
		#####################################
		f_Wx = WLanalysis.gnom_fun(center)	
		y, x = array(f_Wx(icat[:2]))
		weight = icat[3]
		#k = np.load(obsPK_dir+'kappa_predict_W%i.npy'%(Wx))#kappa_predict_Mmax2e15_W%i.npy
		A, galn = WLanalysis.coords2grid(x, y, array([k*weight, weight, k]), size=isize)
		Mkw, Mw, Mk = A
		###########################################
		
		for sigmaG in  (0.5, 1.0, 1.8, 3.5, 5.3, 8.9):
			print Wx, sigmaG
			
			mask0 = maskGen(Wx, 0.5, sigmaG)
			mask = WLanalysis.smooth(mask0, 5.0)
			################ make maps ######################
			kmap_predict = WLanalysis.weighted_smooth(Mkw, Mw, PPA=PPA512, sigmaG=sigmaG)
			kmap_predict*=mask
			np.save(obsPK_dir+'maps/kmap_W%i_predict_sigmaG%02d.npy'%(Wx, sigmaG*10), kmap_predict)
			###########################################
if plot_predict_maps:
	def plot_predict_maps_fcn(WxsigmaG):
		Wx, sigmaG = WxsigmaG
		mask0 = maskGen(Wx, 0.5, sigmaG)
		mask = WLanalysis.smooth(mask0, 5.0)
		
		kmap_lensing = kmap_lensing_Gen(Wx, sigmaG)
		kmap_predict = kmap_predict_Gen(Wx, sigmaG)
		bmode_lensing= bmode_lensing_Gen(Wx, sigmaG)
		
		mask_nan = mask0.copy()
		mask_nan[mask0==0]=nan
		
		#imshow(kmap_lensing*mask_nan, vmax=3*std(kmap_lensing), vmin=-2*std(kmap_lensing), origin = 'lower')
		#title('W%i kmap_lensing'%(Wx))
		#colorbar()
		#savefig(plot_dir+'kmap_W%i_sigmaG%s_lensing.jpg'%(Wx,sigmaG))
		#close()

		##imshow(kmap_predict, vmax=3*std(kmap_predict), vmin=-2*std(kmap_predict), origin = 'lower')
		#imshow(kmap_predict*mask_nan, vmax=4*std(kmap_predict), vmin=0, origin = 'lower')
		imshow(kmap_predict, origin = 'lower')
		title('W%i kmap_predict'%(Wx))
		colorbar()
		savefig(plot_dir+'kmap_G10_W%i_sigmaG%s_predict.jpg'%(Wx,sigmaG))
		close()
	map(plot_predict_maps_fcn, [[Wx, sigmaG] for Wx in (2,) for sigmaG in sigmaG_arr])#range(1,5)

	################ plot the correlation ###########
	
	##P2D_signal = ifft2(fft2(kmap_predict)*conj(fft2(kmap_lensing)))
	##P2D_bmode = ifft2(fft2(kmap_predict)*conj(fft2(bmode_lensing)))
	
	###P2D_galn = ifft2(fft2(kmap_predict)*conj(fft2(pgaln)))
	###P2D_lensinggaln = ifft2(fft2(kmap_lensing)*conj(fft2(lgaln)))
	
	##labels=['pred x lens','pred x bmode','lens x galn']#'pred x galn'
	##f=figure()
	##ax=f.add_subplot(111)
	##i=0
	##for iP2D in (P2D_signal, P2D_bmode):#, P2D_lensinggaln):
	###for imap in (kmap_lensing, bmode_lensing):
		###edges, power = WLanalysis.CrossCorrelate(kmap_predict, imap, edges = logspace(0,log10(500),10))
		##edges, power = WLanalysis.azimuthalAverage(real(fftshift(iP2D)))
		##ax.plot(edges[1:]/PPA512, power, label=labels[i])
		###ax.plot(edges/PPA512, power, label=labels[i])
		##i+=1
	##legend(fontsize=10)
	###ax.set_xlabel('arcmin')
	##ax.set_ylabel('Power')
	##ax.set_xscale('log')
	###savefig(plot_dir+'junk2/Noise2pcf_W%i_sigmaG%02d.jpg'%(Wx, sigmaG*10))
	##savefig(plot_dir+'Test2pcf_W%i_sigmaG%02d.jpg'%(Wx, sigmaG*10))
	##close()
	
if peak_proj_vs_lensing:
	'''for certain kappa_peak in project map, find the lensing kappa at that location
	'''
	sigmaG = 1.0
	k=1
	f=figure(figsize=(8,10))
	for sigmaG in sigmaG_arr:
		ax=f.add_subplot(3,2,k)
		#edge_arr = linspace(-.04,0.1,10)
		edge_arr = logspace(-3,-1,6)
		def return_kappa_arr (Wx, sigmaG=sigmaG):
			mask = maskGen(Wx, 0.5, sigmaG)
			kmap_predict = kmap_predict_Gen(Wx, sigmaG)
			#kmap_predict -= mean(kmap_predict)
			kmap_lensing = kmap_lensing_Gen(Wx, sigmaG)
			bmode = bmode_lensing_Gen(Wx, sigmaG)
			kproj_peak_mat = WLanalysis.peaks_mat(kmap_predict)
			#kproj_peak_mat = WLanalysis.peaks_mat(kmap_lensing)
			idx_pos = (kproj_peak_mat!=0)&(~isnan(kproj_peak_mat))&(mask>0)
			kappa_proj = kmap_predict[idx_pos]
			kappa_lensing = kmap_lensing[idx_pos]
			kappa_bmode = bmode[idx_pos]
			
			######## do an overlay of peaks on top of convergence #######
			if sigmaG == 8.9:
				kmap_predict2 = kmap_predict_Gen(Wx, 5.3)
				mask2 = maskGen(Wx, 0.5, 5.3)
				kproj_peak_mat = WLanalysis.peaks_mat(kmap_predict2)
				kproj_peak_mat[mask2==0] = nan
				kproj_peak_mat[isnan(kproj_peak_mat)]=0
				peaksmooth = WLanalysis.smooth(kproj_peak_mat,10)
				kstd=std(kmap_lensing)
				#pstd=std(kmap_predict2)
				
				kmap_lensing[peaksmooth>2*std(peaksmooth)]=nan
				kmap_lensing[mask2==0]=-99
				kmap_predict2[peaksmooth>2*std(peaksmooth)]=nan
				f2=figure(figsize=(20,12))
				axx=f2.add_subplot(121)
				axy=f2.add_subplot(122)
				axx.imshow(kmap_lensing,origin='lower',vmin=-2*kstd,vmax=3*kstd,interpolation='nearest')
				#f2.colorbar()
				axx.set_title('lensing')
				axy.imshow(kmap_predict2,origin='lower',interpolation='nearest')
				#plt.colorbar(cax=axy)
				axy.set_title('predict')
				savefig(plot_dir+'peaks_location_W%s.jpg'%(Wx))
				close()
				
			return kappa_proj, kappa_lensing, kappa_bmode
		kappa_proj, kappa_lensing, kappa_bmode = return_kappa_arr(2, sigmaG)
		#out = map(return_kappa_arr, range(1,5))#4x3
		#kappa_proj, kappa_lensing, kappa_bmode = [concatenate([out[i][j] for i in range(4)]) for j in range(3)]

		ymean = zeros(len(edge_arr)-1)
		ymeanB =zeros(len(edge_arr)-1)
		for i in arange(len(edge_arr)-1):
			idx_pos2=(kappa_proj<edge_arr[i+1])&(kappa_proj>edge_arr[i])
			ymean[i]=mean(kappa_lensing[idx_pos2])
			ymeanB[i]=mean(kappa_bmode[idx_pos2])
			#idx_pos2=(kappa_lensing<edge_arr[i+1])&(kappa_lensing>edge_arr[i])
			#idx_pos3=(kappa_bmode<edge_arr[i+1])&(kappa_bmode>edge_arr[i])
			#ymean[i]=mean(kappa_proj[idx_pos2])
			#ymeanB[i]=mean(kappa_proj[idx_pos3])
		ax.scatter(kappa_proj, kappa_lensing, s=1)
		ax.plot(edge_arr[1:], ymean, 'ro',label='convergence')
		ax.plot(edge_arr[1:], ymeanB, 'bx',label='B mode')
		ax.set_xlim(0, 3*std(kappa_proj))
		if k==1:
			ax.legend(loc=0,fontsize=10)
		if k>4:
			ax.set_xlabel('kappa_predict')
		ax.set_title('%s arcmin'%(sigmaG))
		k+=1
		#ax.set_xscale('log')
	savefig(plot_dir+'kappa_pred_lensing_G10.jpg')
	close()
	
if cross_correlate:
	sigmaG = 1.0
	edgesGen = lambda Wx: logspace(log10(5),log10(300),7)*sizes[Wx-1]/1330.0
	def returnCC (Wx):
		edges = edgesGen(Wx)
		#print Wx
		mask = maskGen(Wx, 0.5, sigmaG)
		galn = WLanalysis.readFits(obsPK_dir+'maps/W%i_galn_1.3_lo_sigmaG%02d.fit'%(Wx, sigmaG*10))
		kmap = kmap_lensing_Gen(Wx, sigmaG)
		bmode = bmode_lensing_Gen(Wx, sigmaG)
		kproj = kmap_predict_Gen(Wx, sigmaG)
		
		ell_arr, pk = WLanalysis.CrossCorrelate(kmap*mask, galn*mask,edges=edges)
		ell_arr, pb = WLanalysis.CrossCorrelate(kmap*mask, bmode*mask,edges=edges)
		ell_arr, pp = WLanalysis.CrossCorrelate(kproj*mask, galn*mask,edges=edges)

		ell_arr, ppk = WLanalysis.CrossCorrelate(kproj*mask, kmap*mask,edges=edges)
		ell_arr, ppb = WLanalysis.CrossCorrelate(kproj*mask, bmode*mask,edges=edges)
		return ell_arr, pk, pb, pp, ppk, ppb
	
	out = array(map(returnCC,range(1,5)))
	ell_arr, pk, pb, pp, ppk, ppb = mean(out, axis=0)
	ell_arr_err, pk_err, pb_err, pp_err, ppk_err, ppb_err = std(out, axis=0)
	
	f=figure(figsize=(8,12))
	ax=f.add_subplot(211)
	ax2=f.add_subplot(212)	
	colors=['r','b','g','m']
	ax2.errorbar(ell_arr, pk, pk_err, fmt='o',label='kappa x galn')
	ax2.errorbar(ell_arr, pb, pb_err,fmt='*', label='bmode x galn')
	ax2.errorbar(ell_arr, pp, pp_err,fmt='d', label='kproj x galn')

	ax.errorbar(ell_arr, ppk,ppk_err,fmt='o', label='kproj x kappa')
	ax.errorbar(ell_arr, ppb,ppb_err,fmt='d', label='kproj x bmode')

	ax.set_xscale('log')
	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':14},loc=0)
	leg.get_frame().set_visible(False)
	ax.set_xlabel('ell')
	ax.set_ylabel('Power')
	ax.set_xlim(1e3,2e4)
	ax.set_title('sigmaG = %s arcmin'%(sigmaG))
	ax.set_xlim(ell_arr[0]/2,ell_arr[-1]*2)
	ax2.set_xscale('log')
	ax2.legend(fontsize=10,loc=2)
	ax2.set_xlabel('ell')
	ax2.set_ylabel('Power')
	leg2=ax2.legend(ncol=1, labelspacing=0.3, prop={'size':14},loc=0)
	leg2.get_frame().set_visible(False)
	ax2.set_xlim(ell_arr[0]/2,ell_arr[-1]*2)
	#show()
	savefig(plot_dir+'CCmean_L12_kproj_kappa_sigmaG%02d.jpg'%(sigmaG*10))
	close()

	######
	#kproj_peak_mat = WLanalysis.peaks_mat(kproj)
	#mask = maskGen(Wx, 0.5, sigmaG)
	#kproj_peak_mat[where(mask==0)]=nan
	#kproj_peak_mat[isnan(kproj_peak_mat)]=0
	#kproj_peak_smooth = WLanalysis.smooth(kproj_peak_mat,10)
	#mask2 = maskGen(Wx, 0.5, sigmaG)
	#kmap[where(mask2==0)]=nan
	#imshow(kmap, origin='lower',vmin=-2*std(kmap[~isnan(kmap)]),vmax=3*std(kmap[~isnan(kmap)]))
	#contour(kproj_peak_smooth, origin='lower',levels=(5e-6,))
	#show()


			

print 'done-done-done'