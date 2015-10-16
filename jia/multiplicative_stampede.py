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

#################### constants and small functions ##################
sizes = (1330, 800, 1120, 950)
#main_dir = '/Users/jia/weaklensing/multiplicative/'
main_dir = '/work/02977/jialiu/multiplicative/'

galnGen = lambda Wx, cut: load (main_dir+'cfht_galn/W%i_cut%i.npy'%(Wx, cut))
CkappaGen = lambda Wx: WLanalysis.readFits (main_dir+'cfht_kappa/W%s_KS_1.3_lo_sigmaG10.fit'%(Wx))
PkappaGen = lambda Wx: load (main_dir+'planck2015_kappa/dat_kmap_flipper2048_CFHTLS_W%s_map.npy'%(Wx))
CmaskGen = lambda Wx: load (main_dir+'cfht_mask/Mask_W%s_0.7_sigmaG10.npy'%(Wx))
PmaskGen = lambda Wx: load (main_dir+'planck2015_mask/kappamask_flipper2048_CFHTLS_W%s_map.npy'%(Wx))
maskGen = lambda Wx: CmaskGen(Wx)*PmaskGen(Wx)

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

def ps1DGen(kmap):
	size = kmap.shape[0]
	F = fftshift(fftpack.fft2(kmap.astype(float)))
	psd2D = np.abs(F)**2 # = real**2 + imag**2

	ell_arr0, psd1D0 = WLanalysis.azimuthalAverage(psd2D, center=None, edges = arange(sqrt(2)*size/2))
	ell_arr_center = WLanalysis.edge2center(ell_arr0)

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
	p2D_std = sqrt(p2D_mean/2.0)
	return p2D_mean, p2D_std

from scipy.fftpack import fftfreq, fftshift,ifftshift
from random import gauss
from emcee.utils import MPIPool
p = MPIPool()
class GRF_Gen:
	'''return a random gaussian field that has the same power spectrum as img.
	'''
	def __init__(self, kmap):
		self.size = kmap.shape[0]
		self.GRF = rand(self.size,self.size)
		self.p2D_mean, self.p2D_std = ps1DGen(kmap)
	
	def newGRF(self):
		self.psd2D_GRF = gauss(self.p2D_mean, self.p2D_std)
		self.rand_angle = rand(self.size**2).reshape(self.size,self.size)*2.0*pi
		self.psd2D_GRF_Fourier = sqrt(self.psd2D_GRF) * [cos(self.rand_angle) + 1j * sin(self.rand_angle)]
		self.GRF_image = fftpack.ifft2(ifftshift(self.psd2D_GRF_Fourier))[0]
		self.GRF = sqrt(2)*real(self.GRF_image)
		return self.GRF


#def sim_err (kmap, galn, Wx, seednum=0):
	#'''generate 100 GRF from galn, then compute the 100 cross correlation with kmap'''
	#random.seed(seednum)
	#x = GRF_Gen(galn)
	##CC_arr = array([WLanalysis.CrossCorrelate(kmap*mask_arr[Wx-1], x.newGRF()*mask_arr[Wx-1], edges = edges_arr[Wx-1], sigmaG1=1.0, sigmaG2=1.0)[1]/fmask2_arr[Wx-1]/factor for i in range(100)])
	#iCC = lambda i: WLanalysis.CrossCorrelate(kmap*mask_arr[Wx-1], x.newGRF()*mask_arr[Wx-1], edges = edges_arr[Wx-1], sigmaG1=1.0, sigmaG2=1.0)[1]/fmask2_arr[Wx-1]/factor
	#CC_arr = array(p.map(iCC, range(100)))
	#return CC_arr

########### compute sim error ######################
seednum=0
for Wx in range(1,5):
	Ckmap = CkappaGen(Wx)
	Pkmap = PkappaGen(Wx)
	for cut in (22, 23, 24):
		print 'Wx, cut', Wx, cut
		galn = galnGen(Wx, cut)
		
		random.seed(seednum)
		x = GRF_Gen(galn)
		
		#iCCP = lambda i: 
		def iCCP (i):
			return WLanalysis.CrossCorrelate(Pkmap*mask_arr[Wx-1], x.newGRF()*mask_arr[Wx-1], edges = edges_arr[Wx-1], sigmaG1=1.0, sigmaG2=1.0)[1]/fmask2_arr[Wx-1]/factor
		Psim_err_arr = array(p.map(iCCP, range(100)))
		
		random.seed(seednum)
		#iCCC = lambda i: 
		def iCCC(i):
			return WLanalysis.CrossCorrelate(Ckmap*mask_arr[Wx-1], x.newGRF()*mask_arr[Wx-1], edges = edges_arr[Wx-1], sigmaG1=1.0, sigmaG2=1.0)[1]/fmask2_arr[Wx-1]/factor
		Csim_err_arr = array(p.map(iCCC, range(100)))
		
		#Csim_err_arr = sim_err(Ckmap, galn, Wx)
		#Psim_err_arr = sim_err(Pkmap, galn, Wx)
		save(main_dir+'powspec/CCsim_cfht_cut%i_W%i.npy'%(cut, Wx), Csim_err_arr)
		save(main_dir+'powspec/CCsim_planck_cut%i_W%i.npy'%(cut, Wx), Psim_err_arr)
		

	
#for cut in (22,23, 24):
	##planck_CC_err = array([theory_CC_err(PkappaGen(Wx), galnGen(Wx,cut), Wx) for Wx in range(1,5)])

	##cfht_CC_err = array([theory_CC_err(CkappaGen(Wx), galnGen(Wx,cut), Wx) for Wx in range(1,5)])
	
	##save(main_dir+'powspec/planck_CC_err_%s.npy'%(cut), planck_CC_err)
	##save(main_dir+'powspec/cfht_CC_err_%s.npy'%(cut), cfht_CC_err)
	
	#planck_CC_err = load(main_dir+'powspec/planck_CC_err_%s.npy'%(cut))
	#cfht_CC_err = load(main_dir+'powspec/cfht_CC_err_%s.npy'%(cut))
	
	#planck_SNR = find_SNR (planck_CC_err[:,0,:], planck_CC_err[:,1,:])	
	#cfht_SNR = find_SNR (cfht_CC_err[:,0,:], cfht_CC_err[:,1,:])
	
	##print 'i<%i\tSNR(planck)=%.2f\tSNR(cfht)=%.2f (using mean, or 6 bins)'%(cut,planck_SNR[0],cfht_SNR[0])
	##print 'i<%i\tSNR(planck)=%.2f\tSNR(cfht)=%.2f (using all 24 bins)'%(cut,planck_SNR[1],cfht_SNR[1])
	
	#from pylab import *
	#f=figure(figsize=(8,8))
	#SNR_arr = ((planck_CC_err, planck_SNR), (cfht_CC_err, cfht_SNR))
	#for i in (1,2):
		#if i == 1:
			#proj='planck'
		#if i == 2:
			#proj='cfht'
		#ax=f.add_subplot(2,1,i)
		#iCC, iSNR = SNR_arr[i-1]
		#CC_arr = iCC[:,0,:]*ell_arr*1e5
		
		#errK_arr = array([std(load(main_dir+'powspec/CCsim_cfht_cut%i_W%i.npy'%(cut, Wx)), axis=0) for Wx in range(1,5)])*1e5*ell_arr
		#SNR, SNR2, CC_mean, err_mean = find_SNR(CC_arr/(1e5*ell_arr), errK_arr/(1e5*ell_arr))
		##errK_arr = iCC[:,1,:]*ell_arr*1e5
		##SNR, SNR2, CC_mean, err_mean =iSNR
		
		#print 'i<%i\tSNR(%s)=%.2f (6bins), %.2f (24bins)'%(cut,proj,SNR, SNR2)
		##print 'i<%i\tSNR(planck)=%.2f\tSNR(cfht)=%.2f (using all 24 bins)'%(cut,planck_SNR[1],cfht_SNR[1])
	
	
		#CC_mean *= 1e5*ell_arr
		#err_mean *= 1e5*ell_arr
		
		#ax.bar(ell_arr, 2*err_mean, bottom=(CC_mean-err_mean), width=ones(len(ell_arr))*80, align='center',ec='brown',fc='none',linewidth=1.5, alpha=1.0)#
		
		#ax.plot([0,2000], [0,0], 'k-', linewidth=1)
		#seed(16)#good seeds: 6, 16, 25, 41, 53, 128, 502, 584
		#for Wx in range(1,5):
			#cc=rand(3)#colors[Wx-1]
			#ax.errorbar(ell_arr+(Wx-2.5)*15, CC_arr[Wx-1], errK_arr[Wx-1], fmt='o',ecolor=cc,mfc=cc, mec=cc, label=r'$\rm W%i$'%(Wx), linewidth=1.2, capsize=0)
		##handles, labels = ax.get_legend_handles_labels()
		##handles = [handles[i] for i in (3, 4,5,6,0,1, 2)]
		##labels = [labels[i] for i in (3,4,5,6,0,1, 2)]
		#leg=ax.legend(loc=3,fontsize=14,ncol=2)
		#leg.get_frame().set_visible(False)
		#ax.set_xlabel(r'$\ell$',fontsize=14)
		#ax.set_xlim(0,2000)
		
		#ax.set_ylabel(r'$\ell C_{\ell}^{\kappa_{%s}\Sigma}(\times10^{5})$'%(proj),fontsize=14)
		#if i==1:
			#ax.set_title('i<%s, SNR(planck)=%.2f, SNR(cfht)=%.2f'%(cut, planck_SNR[0], cfht_SNR[0]),fontsize=14)
			##ax.set_ylim(-4,5)
		#ax.tick_params(labelsize=14)
	#savefig(main_dir+'plot/CC_simErr_cut%s.jpg'%(cut))
	#close()
	#show()

