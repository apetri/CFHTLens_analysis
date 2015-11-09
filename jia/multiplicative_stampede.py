#!python
# Jia Liu (10/15/2015)
# multiplicative bias project with Alvaro and Colin
# This code does:
# (1) calculate 2 cross-correlations x 3 cuts = 6
#     (galn x CFHT, galn x Planck CMB lensing);
# (2) calculate their theoretical error;
# (3) generate 100 GRF from galn maps, to get error estimation (600 in tot)
# (4) compute the model
# (5) calculate SNR
# (6) estimate m

import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack
from scipy.fftpack import fftfreq, fftshift
import os
import scipy.ndimage as snd
import WLanalysis
import sys

########## knobs #############
compute_sim_err = 1
compute_model = 0
plot_cc = 0
#main_dir = '/Users/jia/weaklensing/multiplicative/'
main_dir = '/work/02977/jialiu/multiplicative/'

#################### constants and small functions ##################
sizes = (1330, 800, 1120, 950)

galnGen = lambda Wx, cut: load (main_dir+'cfht_galn/W%i_cut%i.npy'%(Wx, cut))
CkappaGen = lambda Wx: WLanalysis.readFits (main_dir+'cfht_kappa/W%s_KS_1.3_lo_sigmaG10.fit'%(Wx))
PkappaGen = lambda Wx: load (main_dir+'planck2015_kappa/dat_kmap_flipper2048_CFHTLS_W%s_map.npy'%(Wx))
CmaskGen = lambda Wx: load (main_dir+'cfht_mask/Mask_W%s_0.7_sigmaG10.npy'%(Wx))
PmaskGen = lambda Wx: load (main_dir+'planck2015_mask/kappamask_flipper2048_CFHTLS_W%s_map.npy'%(Wx))
maskGen = lambda Wx: CmaskGen(Wx)*PmaskGen(Wx)
PlanckSim15Gen = lambda Wx, r: load('/work/02977/jialiu/cmblensing/planck/sim15/sim_%04d_kmap_CFHTLS_W%s.npy'%(r, Wx))

edgesGen = lambda Wx: linspace(1,60,7)*sizes[Wx-1]/1330.0
### omori & holder bin edges #####
#edgesGen = lambda Wx: linspace(1.25, 47.49232195,21)*sizes[Wx-1]/1330.0

edges_arr = map(edgesGen, range(1,5))
sizedeg_arr = array([(sizes[Wx-1]/512.0)**2*12.0 for Wx in range(1,5)])
####### test: ell_arr = WLanalysis.PowerSpectrum(CmaskGen(1), sizedeg = sizedeg_arr[0],edges=edges_arr[0])[0]
ell_arr = WLanalysis.edge2center(edgesGen(1))*360.0/sqrt(sizedeg_arr[0])

factor = (ell_arr+1)*ell_arr/(2.0*pi)
mask_arr = map(maskGen, range(1,5))
fmask_arr = array([sum(mask_arr[Wx-1])/sizes[Wx-1]**2 for Wx in range(1,5)])
fmask2_arr = array([sum(mask_arr[Wx-1]**2)/sizes[Wx-1]**2 for Wx in range(1,5)])
fsky_arr = fmask_arr*sizedeg_arr/41253.0
d_ell = ell_arr[1]-ell_arr[0]
#################################################

def theory_CC_err(map1, map2, Wx):
	map1*=mask_arr[Wx-1]
	map2*=mask_arr[Wx-1]
	#map1-=mean(map1)
	#map2-=mean(map2)
	auto1 = WLanalysis.PowerSpectrum(map1, sizedeg = sizedeg_arr[Wx-1], edges=edges_arr[Wx-1],sigmaG=1.0)[-1]/fmask2_arr[Wx-1]/factor
	auto2 = WLanalysis.PowerSpectrum(map2, sizedeg = sizedeg_arr[Wx-1], edges=edges_arr[Wx-1],sigmaG=1.0)[-1]/fmask2_arr[Wx-1]/factor	
	err = sqrt(auto1*auto2/fsky_arr[Wx-1]/(2*ell_arr+1)/d_ell)
	CC = WLanalysis.CrossCorrelate(map1, map2, edges = edges_arr[Wx-1], sigmaG1=1.0, sigmaG2=1.0)[1]/fmask2_arr[Wx-1]/factor
	return CC, err

def find_SNR (CC_arr, errK_arr):
	'''Find the mean of 4 fields, and the signal to noise ratio (SNR).
	Input:
	CC_arr: array of (4 x Nbin) in size, for cross correlations;
	errK_arr: error bar array, same dimension as CC_arr;
	Output:
	SNR = signal to noise ratio
	CC_mean = the mean power spectrum of the 4 fields, an Nbin array.
	err_mean = the mean error bar of the 4 fields, an Nbin array.
	'''
	weightK = 1/errK_arr**2/sum(1/errK_arr**2, axis=0)
	CC_mean = sum(CC_arr*weightK,axis=0)
	err_mean = sqrt(1.0/sum(1/errK_arr**2, axis=0))
	SNR = sqrt( sum(CC_mean**2/err_mean**2) )
	SNR2 = sqrt(sum (CC_arr**2/errK_arr**2))
	return SNR, SNR2, CC_mean, err_mean

#def ps1DGen(kmap):
	#size = kmap.shape[0]
	#F = fftshift(fftpack.fft2(kmap.astype(float)))
	#psd2D = np.abs(F)**2 # = real**2 + imag**2

	#ell_arr0, psd1D0 = WLanalysis.azimuthalAverage(psd2D, center=None, edges = arange(sqrt(2)*size/2))
	#ell_arr_center = WLanalysis.edge2center(ell_arr0)

	#randfft2 = zeros(shape=(size, size))
	#y, x = np.indices((size,size))
	#center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
	#if size%2 == 0:
		#center+=0.5
	#r = np.hypot(x - center[0], y - center[1])
	
	#extrap = psd1D0[-1]+(ceil(sqrt(2)*size/2+1.0)-ell_arr_center[-1])*(psd1D0[-1]-psd1D0[-2])/(ell_arr_center[-1]-ell_arr_center[-2])
	
	#ell_arr = array([0,]+list(ell_arr_center)+ [ceil(sqrt(2)*size/2+1.0),])
	#psd1D = array([psd1D0[0],]+list(psd1D0)+[extrap,])
	
	#p1D_interp = interpolate.griddata(ell_arr, psd1D, r.flatten(), method='nearest')
	#p1D_interp[isnan(p1D_interp)]=0

	#p2D_mean = p1D_interp.reshape(size,size)
	#p2D_std = sqrt(p2D_mean/2.0)
	#return p2D_mean, p2D_std

########### compute sim error ######################
if compute_sim_err:
	from scipy.fftpack import fftfreq, fftshift,ifftshift
	from random import gauss
	from emcee.utils import MPIPool
	p = MPIPool()
	#class GRF_Gen:
		#'''return a random gaussian field that has the same power spectrum as img.
		#'''
		#def __init__(self, kmap):
			#self.size = kmap.shape[0]
			#self.GRF = rand(self.size,self.size)
			#self.p2D_mean, self.p2D_std = ps1DGen(kmap)
		
		#def newGRF(self):
			#self.psd2D_GRF = gauss(self.p2D_mean, self.p2D_std)
			#self.rand_angle = rand(self.size**2).reshape(self.size,self.size)*2.0*pi
			#self.psd2D_GRF_Fourier = sqrt(self.psd2D_GRF) * [cos(self.rand_angle) + 1j * sin(self.rand_angle)]
			#self.GRF_image = fftpack.ifft2(ifftshift(self.psd2D_GRF_Fourier))[0]
			#self.GRF = sqrt(2)*real(self.GRF_image)
			#return self.GRF
		
	seednum=0
	Wx, cut = int(sys.argv[1]), int(sys.argv[2])
	Pkmap = PkappaGen(Wx)*mask_arr[Wx-1]

	print 'Wx, cut', Wx, cut
	galn = galnGen(Wx, cut)*mask_arr[Wx-1]
	igaln = galn.copy()
	
	random.seed(seednum)
	#x = WLanalysis.GRF_Gen(galn)
	
	#Ckmap0 = CkappaGen(Wx)*mask_arr[Wx-1]
	#CFHTx = WLanalysis.GRF_Gen (Ckmap0)
	
	def iCC (i):
		
		#igaln = x.newGRF()*mask_arr[Wx-1]
		######## # use Planck sim map, and CFHT GRF map
		Pkmap = PlanckSim15Gen(Wx, i)*mask_arr[Wx-1]
		#Ckmap = CFHTx.newGRF()
		Ckmap = load('/work/02977/jialiu/kSZ/CFHT/Noise/W%i_Noise_sigmaG10_%04d.npy'%(Wx, i))
		#############
		
		CCP = WLanalysis.CrossCorrelate(Pkmap, igaln, edges = edges_arr[Wx-1], sigmaG1=1.0, sigmaG2=1.0)[1]/fmask2_arr[Wx-1]/factor
		CCC = WLanalysis.CrossCorrelate(Ckmap, igaln, edges = edges_arr[Wx-1], sigmaG1=1.0, sigmaG2=1.0)[1]/fmask2_arr[Wx-1]/factor
		return CCP, CCC

	if not p.is_master():
		p.wait()
		sys.exit(0)
	CCsim_err_arr = array(p.map(iCC, range(100)))
	save(main_dir+'powspec/CC_Plancksim_CFHTrot_cut%i_W%i.npy'%(cut, Wx), CCsim_err_arr)

	p.close()
############# finish compute sim error #####################


############ calculate theory #################
if compute_model:
	for cut in (22,23,24):#cut=22
		print cut
		from scipy.integrate import quad
		z_center= arange(0.025, 3.5, 0.05)
		dndzgal = load(main_dir+'dndz/dndz_0213_cut%s_noweight.npy'%(cut))[:,1]
		dndzkappa = load(main_dir+'dndz/dndz_0213_weighted.npy')[:,1]
		Ptable = genfromtxt('/Users/jia/weaklensing/cmblensing/P_delta_Planck15')
		z_center = concatenate([[0,], z_center, [4.0,]])
		dndzgal = concatenate([[0,], dndzgal, [0,]])
		dndzkappa = concatenate([[0,], dndzkappa, [0,]])
		dndzgal /= 0.05*sum(dndzgal)
		dndzkappa /= 0.05*sum(dndzkappa)
		dndzgal_interp = interpolate.interp1d(z_center,dndzgal ,kind='cubic')
		dndzkappa_interp = interpolate.interp1d(z_center,dndzkappa ,kind='cubic')

		OmegaM = 0.3156#,0.29982##0.33138#
		H0 = 67.27
		OmegaV = 1.0-OmegaM
		h = H0/100.0
		c = 299792.458#km/s
		H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV))
		DC_fcn = lambda z: c*quad(H_inv, 0, z)[0] # comoving distance Mpc
		z_ls = 1100 #last scattering
		z_arr = linspace(0, 4.0, 200)
		DC = interpolate.interp1d(linspace(0,1100,300), [DC_fcn(iz) for iz in linspace(0,1100,300)])
		
		integrand = lambda zs, z: dndzkappa_interp(zs)*(1-DC(z)/DC(zs))
		W_wl_fcn = lambda z: quad(integrand, z, 4.0, args=(z,))[0]
		W_wl0 = array(map(W_wl_fcn, z_arr))
		W_wl = interpolate.interp1d(z_arr, W_wl0)
		W_cmb = lambda z: (1-DC(z)/DC(z_ls))

		aa = array([1/1.05**i for i in arange(33)])
		zz = 1.0/aa-1 # redshifts
		kk = Ptable.T[0]
		iZ, iK = meshgrid(zz,kk)
		Z, K = iZ.flatten(), iK.flatten()
		P_deltas = Ptable[:,1:34].flatten()

		Pmatter_interp = interpolate.CloughTocher2DInterpolator(array([K*h, Z]).T, 2.0*pi**2*P_deltas/(K*h)**3)
		Pmatter = lambda k, z: Pmatter_interp (k, z)
		
		ell_arr2 = linspace(30, 2000, 20)

		Cplan_integrand = lambda z, ell: (1.0+z)/DC(z)*dndzgal_interp(z)*W_cmb(z)*Pmatter(ell/DC(z), z)

		Ccfht_integrand = lambda z, ell: (1.0+z)/DC(z)*dndzgal_interp(z)*W_wl(z)*Pmatter(ell/DC(z), z)
		
		print 'Cplan_arr'
		Cplan_arr = 1.5*OmegaM*(H0/c)**2*array([quad(Cplan_integrand, 0.002, 3.5 , args=(iell))[0] for iell in ell_arr2])
		
		print 'Ccfht_arr'
		Ccfht_arr = 1.5*OmegaM*(H0/c)**2*array([quad(Ccfht_integrand, 0.002, 3.5 , args=(iell))[0] for iell in ell_arr2])
		
		save(main_dir+'powspec/Cplanck_cut%s_arr.npy'%(cut), array([ell_arr2, Cplan_arr]).T)
		save(main_dir+'powspec/Ccfht_cut%s_arr.npy'%(cut), array([ell_arr2, Ccfht_arr]).T)
		save(main_dir+'powspec/Ccfht_over_Cplanck_cut%s.npy'%(cut), array([ell_arr2, Ccfht_arr/Cplan_arr]).T)

############ done theory ######################

################ test against omori and holder ##########

#for cut in (22,23, 24):
	#planck_CC_err = array([theory_CC_err(PkappaGen(Wx), galnGen(Wx,cut), Wx) for Wx in range(1,5)])

	#cfht_CC_err = array([theory_CC_err(CkappaGen(Wx), galnGen(Wx,cut), Wx) for Wx in range(1,5)])
	
	
	#planck_SNR = find_SNR (planck_CC_err[:,0,:], planck_CC_err[:,1,:])	
	#cfht_SNR = find_SNR (cfht_CC_err[:,0,:], cfht_CC_err[:,1,:])
	
	#print 'i<%i\tSNR(planck)=%.2f\tSNR(cfht)=%.2f (using mean, or 20 bins between ell=[50,1900])'%(cut,planck_SNR[0],cfht_SNR[0])
	#print 'i<%i\tSNR(planck)=%.2f\tSNR(cfht)=%.2f (all 20x4=80bins)'%(cut,planck_SNR[1],cfht_SNR[1])
	
	
############# plotting: 2 cross-correlation and theory #################
if plot_cc:
	for cut in (22, 23, 24): ######## 3 field cross power spectrum
		### compute C_ell, only needed once
		#planck_CC_err = array([theory_CC_err(PkappaGen(Wx), galnGen(Wx,cut), Wx) for Wx in range(1,5)])
		#cfht_CC_err = array([theory_CC_err(CkappaGen(Wx), galnGen(Wx,cut), Wx) for Wx in range(1,5)])
		#save(main_dir+'powspec/planck_CC_err_%s.npy'%(cut), planck_CC_err)
		#save(main_dir+'powspec/cfht_CC_err_%s.npy'%(cut), cfht_CC_err)
		
		planck_CC_err = load(main_dir+'powspec/planck_CC_err_%s.npy'%(cut))
		cfht_CC_err = load(main_dir+'powspec/cfht_CC_err_%s.npy'%(cut))
		
		planck_SNR = find_SNR (planck_CC_err[:,0,:], planck_CC_err[:,1,:])	
		cfht_SNR = find_SNR (cfht_CC_err[:,0,:], cfht_CC_err[:,1,:])
		
		#print 'i<%i\tSNR(planck)=%.2f\tSNR(cfht)=%.2f (using mean, or 6 bins)'%(cut,planck_SNR[0],cfht_SNR[0])
		#print 'i<%i\tSNR(planck)=%.2f\tSNR(cfht)=%.2f (using all 24 bins)'%(cut,planck_SNR[1],cfht_SNR[1])
		
		from pylab import *
		f=figure(figsize=(8,8))
		SNR_arr = ((planck_CC_err, planck_SNR), (cfht_CC_err, cfht_SNR))
		
		##########simerr
		errK_arr2 = array([std(load(main_dir+'powspec/CC_Plancksim_cut%i_W%i.npy'%(cut, Wx)), axis=0) for Wx in range(1,5)])*1e5*ell_arr
		##################
		for i in (1,2):
			if i == 1:
				proj='planck'
				
			if i == 2:
				proj='cfht'
				
			ell_theo, CC_theo = load(main_dir+'powspec/C%s_cut%s_arr.npy'%(proj,cut)).T
			ax=f.add_subplot(2,1,i)
			iCC, iSNR = SNR_arr[i-1]
			CC_arr = iCC[:,0,:]*ell_arr*1e5
			ax.plot(ell_theo, CC_theo*ell_theo*1e5, '--',label='Planck')

			errK_arr = iCC[:,1,:]*ell_arr*1e5
			SNR, SNR2, CC_mean, err_mean =iSNR
			
			##########uncomment to use simerr
			errK_arr = errK_arr2[:,i-1,:]
			SNR, SNR2, CC_mean, err_mean = find_SNR(CC_arr/(1e5*ell_arr), errK_arr/(1e5*ell_arr))
			##################
			
			print 'i<%i\tSNR(%s)=%.2f (6bins),\t%.2f (24bins)'%(cut,proj,SNR, SNR2)
			#print 'i<%i\tSNR(planck)=%.2f\tSNR(cfht)=%.2f (using all 24 bins)'%(cut,planck_SNR[1],cfht_SNR[1])
		
		
			CC_mean *= 1e5*ell_arr
			err_mean *= 1e5*ell_arr
			
			ax.bar(ell_arr, 2*err_mean, bottom=(CC_mean-err_mean), width=ones(len(ell_arr))*80, align='center',ec='brown',fc='none',linewidth=1.5, alpha=1.0)#
			
			ax.plot([0,2000], [0,0], 'k-', linewidth=1)
			seed(16)#good seeds: 6, 16, 25, 41, 53, 128, 502, 584
			for Wx in range(1,5):
				cc=rand(3)#colors[Wx-1]
				ax.errorbar(ell_arr+(Wx-2.5)*15, CC_arr[Wx-1], errK_arr[Wx-1], fmt='o',ecolor=cc,mfc=cc, mec=cc, label=r'$\rm W%i$'%(Wx), linewidth=1.2, capsize=0)
			leg=ax.legend(loc=3,fontsize=14,ncol=2)
			leg.get_frame().set_visible(False)
			ax.set_xlabel(r'$\ell$',fontsize=14)
			ax.set_xlim(0,2000)
			
			ax.set_ylabel(r'$\ell C_{\ell}^{\kappa_{%s}\Sigma}(\times10^{5})$'%(proj),fontsize=14)
			if i==1:
				ax.set_title('i<%s'%(cut), fontsize=14)# SNR(planck)=%.2f, SNR(cfht)=%.2f'%(cut, planck_SNR[0], cfht_SNR[0]),fontsize=14)
				#ax.set_ylim(-4,5)
			ax.tick_params(labelsize=14)

		#show()
		savefig(main_dir+'plot/CC_planck15_simErr_cut%s.jpg'%(cut))
		close()

############## ratio plot
#for cut in range(22,25):
	
	#planck_CC_arr = load(main_dir+'powspec/planck_CC_err_%s.npy'%(cut))
	#cfht_CC_arr = load(main_dir+'powspec/cfht_CC_err_%s.npy'%(cut))
	#CC_arr = cfht_CC_arr[:,0,:]/planck_CC_arr[:,0,:]
	#errK_arr = abs(CC_arr)*sqrt((cfht_CC_arr[:,1,:]/cfht_CC_arr[:,0,:])**2+ (planck_CC_arr[:,1,:]/planck_CC_arr[:,0,:])**2)

	#SNR, SNR2, CC_mean, err_mean = find_SNR(CC_arr, errK_arr)
	#ell_arr2, Rtheory = load(main_dir+'powspec/Ccfht_over_Cplanck_cut%s.npy'%(cut)).T

	#from pylab import *
	#f=figure(figsize=(8,6))
	#ax=f.add_subplot(111)
		
	#ax.bar(ell_arr, 2*err_mean, bottom=(CC_mean-err_mean), width=ones(len(ell_arr))*80, align='center',ec='brown',fc='none',linewidth=1.5, alpha=1.0)#
	
	#ax.plot(ell_arr2, Rtheory, '--',label='Planck')
	#ax.plot([0,2000], [0,0], 'k-', linewidth=1)
	#seed(16)#good seeds: 6, 16, 25, 41, 53, 128, 502, 584
	#for Wx in range(1,5):
		#cc=rand(3)#colors[Wx-1]
		#ax.errorbar(ell_arr+(Wx-2.5)*15, CC_arr[Wx-1], errK_arr[Wx-1], fmt='o',ecolor=cc,mfc=cc, mec=cc, label=r'$\rm W%i$'%(Wx), linewidth=1.2, capsize=0)
	#leg=ax.legend(loc=3,fontsize=14,ncol=2)
	#leg.get_frame().set_visible(False)
	#ax.set_xlabel(r'$\ell$',fontsize=14)
	#ax.set_xlim(0,2000)
	
	#ax.set_ylabel(r'$C_{\ell}^{\kappa_{gal}\Sigma}/C_{\ell}^{\kappa_{cmb}\Sigma}$',fontsize=14)

	#ax.set_title('i<%s, SNR=%.2f (6bins), %.2f (24bins)'%(cut, SNR, SNR2), fontsize=14)
	#ax.set_ylim(-1,1)
	##show()
	#savefig(main_dir+'plot/CC_ratio_cut%s.jpg'%(cut))
	#close()
