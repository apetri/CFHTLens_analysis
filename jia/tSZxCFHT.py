#!python
# Jia Liu 2015/03/23
# This code calculates the model for tSZ x weak lensing

import WLanalysis
import os
import numpy as np
from scipy import *
import sys
from scipy.integrate import quad
import scipy.optimize as op
from scipy import interpolate
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.gridspec as gridspec
from scipy import ndimage as snd

create_maps = 0
init_mask = 0
HM857 = 0 # step 1
yx857 = 0 # step 2
kx857 = 0 # step 3
cc_yxk = 1 # step 4

tSZ_dir = '/Users/jia/weaklensing/tSZxCFHT/'
plot_dir = tSZ_dir+'plot/'
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
PPA512 = 2.4633625
sizes = (1330, 800, 1120, 950)
sizedeg_arr = array([(sizes[Wx-1]/512.0)**2*12.0 for Wx in range(1,5)])
prefix_arr = ('nilc_ymap', 'milca_ymap', 'GARY_ymap', 'JCH_ymap50')
sigmaG_arr = array([4.246, 4.246, 0, 4.246])*PPA512 #smoothing for the 4 maps

####### create maps, and plot them out ######
if create_maps:
	for fn in os.listdir(tSZ_dir+'planck/'):
		if fn[-3:]=='txt':
			print fn
			print fn[:-3]+'npy'
			full_fn = tSZ_dir+'planck/'+fn
			imap=WLanalysis.txt2map_fcn(full_fn, offset = False)
		if fn[-3:]=='npy':
			imap = load(tSZ_dir+'planck/'+fn)
		if 'mask' in fn:
			imshow(imap, origin='lower', vmin=0, vmax=1)
		elif '857' in fn:
			imshow(imap, origin='lower')
		elif 'JCH_ymap' in fn:
			imshow(imap, origin='lower', vmin=-2e-5, vmax=2e-5)
		elif 'GARY' in fn:
			imap = WLanalysis.smooth(imap, 2.5*4)
			imshow(imap, origin='lower', vmin=-2e-5, vmax=2e-5)
		else:
			#imshow(imap, origin='lower', vmin=-3*std(imap), vmax=3*std(imap))
			imshow(imap, origin='lower', vmin=-3e-6, vmax=3e-6)
		title(fn[:-4])
		colorbar()
		savefig(plot_dir+fn[:-3]+'jpg')
		close()

########### cross correlation	
cat_dir = '/Users/jia/weaklensing/CFHTLenS/catalogue/'

kmapGen = lambda i: load(cat_dir+'kmap_W%i_sigma10_zcut13.npy'%(i))

PSmaskGen = lambda i: np.load(tSZ_dir+'planck/PSmaskHFIall_flipper2048_CFHTLS_W%i.npy'%(i))

JCHPSmaskGen = lambda i: load(tSZ_dir+'planck/yJCHmask50_flipper2048_CFHTLS_W%i.npy'%(i))

def maskGen_init (Wx, sigma_pix=10, JCH=0):
	galn = WLanalysis.smooth(load(cat_dir+'Me_Mw_galn/W%i_galn_zcut13.npy'%(Wx)),PPA512)
	if JCH:
		galn *= JCHPSmaskGen(Wx)
	else:
		galn *= PSmaskGen(Wx)## add point source mask for cmbl
	mask = zeros(shape=galn.shape)
	mask[10:-10,10:-10] = 1 ## remove edge 10 pixels
	idx = where(galn<0.5)
	mask[idx] = 0
	mask_smooth = WLanalysis.smooth(mask, sigma_pix)	
	######## print out fksy and fsky 2 ##########
	sizedeg = (sizes[Wx-1]/512.0)**2*12.0
	fsky = sum(mask_smooth)/sizes[Wx-1]**2*sizedeg/41253.0
	fsky2 = sum(mask_smooth**2)/sizes[Wx-1]**2*sizedeg/41253.0
	fmask = sum(mask_smooth)/sizes[Wx-1]**2
	fmask2 = sum(mask_smooth**2)/sizes[Wx-1]**2
	print 'W%i, fsky=%.8f, fsky2=%.8f, fmask=%.8f, fmask2=%.8f'%(Wx, fsky, fsky2, fmask,fmask2) 
	#############################################
	return mask_smooth#fsky, fsky2#

if init_mask:
	JCH = 1
	for Wx in range(1,5):
		imask = maskGen_init(Wx, JCH=JCH)
		fn = 'W%i_JCHmask'%(Wx)#W%i_mask
		save(tSZ_dir+'mask/%s.npy'%(fn), imask) 
		imshow(imask, origin='lower', vmin=0, vmax=1)
		title(fn)
		colorbar()
		savefig(plot_dir+'%s.jpg'%(fn))
		close()
		
maskGen = lambda i: load(tSZ_dir+'mask/W%i_mask.npy'%(i))

JCHmaskGen = lambda i: load(tSZ_dir+'mask/W%i_JCHmask.npy'%(i))

tmapGen = lambda prefix, i: load(tSZ_dir+'planck/%s_CFHTLS_W%i.npy'%(prefix, i))

edgesGen = lambda Wx: linspace(1, 50, 8)*sizes[Wx-1]/1330.0

ell_arr = 40.0*WLanalysis.edge2center(linspace(1,50,8))
d_ell = ell_arr[1]-ell_arr[0]
b_ell = exp(-ell_arr**2*radians(1.0/60)**2/2.0)

def theory_err(map1, map2, Wx, fmask2, fsky):
	'''compute theoretical err, assume map1 map2 are already set 0 mean.
	'''	
	auto1 = WLanalysis.PowerSpectrum(map1, sizedeg = sizedeg_arr[Wx-1], edges=edgesGen(Wx))[-1]/fmask2
	auto2 = WLanalysis.PowerSpectrum(map2, sizedeg = sizedeg_arr[Wx-1], edges=edgesGen(Wx))[-1]/fmask2
	err = sqrt(auto1*auto2/fsky/(2*ell_arr+1)/d_ell)
	return err
#fmask2_arr = array([sum(maskGen(Wx)**2)/sizes[Wx-1]**2 for Wx in range(1,5)])
fmask2_arr = array([ 0.69965629,  0.57894277,  0.62390033,  0.44137115])

if HM857:
	cc_857 = array([WLanalysis.CrossCorrelate(load(tSZ_dir+'planck/Planck857_HM1_CFHTLS_W%i.npy'%(Wx))*maskGen(Wx), load(tSZ_dir+'planck/Planck857_HM2_CFHTLS_W%i.npy'%(Wx))*maskGen(Wx), edges=edgesGen(Wx))[1]/fmask2_arr[Wx-1] for Wx in range(1,5)])
	save(tSZ_dir+'cc_857.npy', cc_857)

if yx857:
	for ip in range(4):
		def yx857_ip(Wx):
			CIBmap = load(tSZ_dir+'planck/Planck857_full_CFHTLS_W%i.npy'%(Wx))
			tmap = tmapGen(prefix_arr[ip], Wx)
			if ip==3:
				tmap[abs(tmap)>1.0]=0
				mask = JCHmaskGen(Wx)
			else:
				mask = maskGen(Wx)
			tmap*=mask
			CIBmap*=mask
			tmap-=mean(tmap)
			CIBmap-=mean(CIBmap)
			fmask2 = sum(mask**2)/sizes[Wx-1]**2
			icc_yx857 = WLanalysis.CrossCorrelate(CIBmap, tmap, edges=edgesGen(Wx))[1]/fmask2
			return icc_yx857
		cc_yx857 = map(yx857_ip, range(1,5))
		save(tSZ_dir+'cc_yx857_%s.npy'%(prefix_arr[ip]), cc_yx857)

if kx857:
	cc_kx857 = array([WLanalysis.CrossCorrelate(load(tSZ_dir+'planck/Planck857_full_CFHTLS_W%i.npy'%(Wx))*maskGen(Wx), kmapGen(Wx)*maskGen(Wx), edges=edgesGen(Wx))[1]/fmask2_arr[Wx-1] for Wx in range(1,5)])
	save(tSZ_dir+'cc_kx857.npy', cc_kx857)
	
if cc_yxk:
	def txk (ipWx):
		'''ip = 0..3 counts the prefix
		Wx = 1..4 
		return the cross correlation
		'''
		ip, Wx = ipWx
		print prefix_arr[ip], Wx
		tmap = tmapGen(prefix_arr[ip], Wx)
		if ip==3:
			tmap[abs(tmap)>1.0]=0
			mask = JCHmaskGen(Wx)
		else:
			mask = maskGen(Wx)
		kmap = kmapGen(Wx)*mask
		tmap *=mask		
		## set mean to 0
		kmap -= mean(kmap)
		tmap -= mean(tmap)
		
		fmask2 = sum(mask**2)/sizes[Wx-1]**2
		fsky = sum(mask)/sizes[Wx-1]**2*sizedeg_arr[Wx-1]/41253.0
		edges = edgesGen(Wx)
		CC_signal = WLanalysis.CrossCorrelate(kmap, tmap, edges=edges)[1]/fmask2
		CC_err = theory_err (kmap, tmap, Wx, fmask2, fsky)
		return CC_signal, CC_err
	ipWx_arr = [[ip, Wx] for ip in range(4) for Wx in arange(1,5)]
	#all_cc = array(map(txk, ipWx_arr))
	#save(tSZ_dir+'all_cc.npy',all_cc)
	all_cc = load(tSZ_dir+'all_cc.npy')
	#for i in range(4):		
		#iarr=all_cc[i*4:(i+1)*4]
		#print i*4, (i+1)*4, iarr.shape
		#save(tSZ_dir+'cc_yxk_%s.npy'%(prefix_arr[i]), iarr)
	i = 0
	
	ell_JCH, halo1, halo2, C_tot = genfromtxt(tSZ_dir+'CellykappaCFHTLS_WMAP9_Jiadndz_zcut13.txt').T#CellykappaCFHTLS_Planck15_Jiadndz_zcut13.txt
	for ip in range(4):
		b_ell_tSZ = exp(-ell_arr**2*radians(sigmaG_arr[ip]/60)**2/2.0)
		b_ell_kappa = exp(-ell_arr**2*radians(1.0/60)**2/2.0)
		factor=2.0*pi/(1+ell_arr)/b_ell_tSZ/b_ell_kappa*1e11
		chisq=0
		f=figure(figsize=(8,5))
		ax=f.add_subplot(111)
		ax.plot(ell_JCH, C_tot*1e11)
		for Wx in arange(1,5):
			cc, err = all_cc[i]
			i+=1
			errorbar(ell_arr+20.0*(Wx-3), cc*factor, err*factor, label='W%i'%(Wx))
			chisq+=sum((cc/err)**2)
		SNR = sqrt(chisq)
		print prefix_arr[ip], SNR
		ax.plot((0,2000),(0,0),'k--')
		ax.set_xlabel(r'$\ell$')
		#ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$')
		ax.set_ylabel(r'$\ell\rm{C(\ell)} \times 10^{11}$')#/\pi
		ax.set_title(prefix_arr[ip])
		ax.set_xlim(0,2000)
		
		savefig(plot_dir+'phi_model_crosscorr_%s.jpg'%(prefix_arr[ip]))
		close()