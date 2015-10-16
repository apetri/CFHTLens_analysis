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
main_dir = '/Users/jia/weaklensing/multiplicative/'
#'/work/02977/jialiu/multiplicative/'

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

#for cut in (22,23, 24):
	##planck_CC_err = array([theory_CC_err(PkappaGen(Wx), galnGen(Wx,cut), Wx) for Wx in range(1,5)])

	##cfht_CC_err = array([theory_CC_err(CkappaGen(Wx), galnGen(Wx,cut), Wx) for Wx in range(1,5)])
	
	##save(main_dir+'powspec/planck_CC_err_%s.npy'%(cut), planck_CC_err)
	##save(main_dir+'powspec/cfht_CC_err_%s.npy'%(cut), cfht_CC_err)
	
	#planck_CC_err = load(main_dir+'powspec/planck_CC_err_%s.npy'%(cut))
	#cfht_CC_err = load(main_dir+'powspec/cfht_CC_err_%s.npy'%(cut))
	
	#planck_SNR = find_SNR (planck_CC_err[:,0,:], planck_CC_err[:,1,:])	
	#cfht_SNR = find_SNR (cfht_CC_err[:,0,:], cfht_CC_err[:,1,:])
	
	#print 'i<%i\tSNR(planck)=%.2f\tSNR(cfht)=%.2f (using mean, or 6 bins)'%(cut,planck_SNR[0],cfht_SNR[0])
	#print 'i<%i\tSNR(planck)=%.2f\tSNR(cfht)=%.2f (using all 24 bins)'%(cut,planck_SNR[1],cfht_SNR[1])
	
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
		#errK_arr = iCC[:,1,:]*ell_arr*1e5
		#SNR, SNR2, CC_mean, err_mean =iSNR
		#CC_mean *=1e5*ell_arr
		#err_mean *=1e5*ell_arr
		
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
	#savefig(main_dir+'plot/CC_cut%s.jpg'%(cut))
	#close()
	##show()

