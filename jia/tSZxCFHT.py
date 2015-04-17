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
HM857 = 0 # step 1 CIB^2
yx857 = 0 # step 2 Y x CIB
kx857 = 0 # step 3 CFHT x CIB
cc_yxk = 0 # step 4 Y x CFHT
SNR_calc = 1

tSZ_dir = '/Users/jia/weaklensing/tSZxCFHT/'
plot_dir = tSZ_dir+'plot/'
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
PPA512 = 2.4633625
sizes = (1330, 800, 1120, 950)
sizedeg_arr = array([(sizes[Wx-1]/512.0)**2*12.0 for Wx in range(1,5)])
prefix_arr = ('nilc_ymap', 'milca_ymap', 'GARY_ymap', 'JCH_ymap50')
sigmaG_arr = array([4.246, 4.246, 0, 4.246, 1.97]) #smoothing for the 4 maps, last number is for CIB maps

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

CIBGen = lambda i: load(tSZ_dir+'planck/Planck857_full_CFHTLS_W%i.npy'%(i))

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

#edgesGen = lambda Wx: linspace(1, 50, 8)*sizes[Wx-1]/1330.0
ell_arr=array([53,114,187,320,502,684,890,1158,1505,1956,2649])
ell_edges=array([23,84,145,229,411,592,774,1006,1308,1701,2211,3085])
edgesGen = lambda Wx: ell_edges*sqrt(sizedeg_arr[Wx-1])/360

#ell_arr = 40.0*WLanalysis.edge2center(linspace(1,50,8))
d_ell = ell_edges[1:]-ell_edges[:-1]
#b_ell = exp(-ell_edges**2*radians(amax(sigmaG_arr)/60)**2/2.0)

def theory_err(map1, map2, Wx, fmask2, fsky, sigmaG1, sigmaG2):
	'''compute theoretical err, assume map1 map2 are already set 0 mean.
	'''	
	auto1 = WLanalysis.PowerSpectrum(map1, sizedeg = sizedeg_arr[Wx-1], edges=edgesGen(Wx), sigmaG=sigmaG1)[-1]/fmask2
	auto2 = WLanalysis.PowerSpectrum(map2, sizedeg = sizedeg_arr[Wx-1], edges=edgesGen(Wx), sigmaG=sigmaG2)[-1]/fmask2
	err = sqrt(auto1*auto2/fsky/(2*ell_arr+1)/d_ell)
	return err
#fmask2_arr = array([sum(maskGen(Wx)**2)/sizes[Wx-1]**2 for Wx in range(1,5)])
fmask2_arr = array([ 0.69965629,  0.57894277,  0.62390033,  0.44137115])
#fsky_arr = array([sum(maskGen(Wx))/sizes[Wx-1]**2*sizedeg_arr[Wx-1]/41253.0 for Wx in range(1,5)])
fsky_arr = array([ 0.001473  ,  0.00047671,  0.00094933,  0.00049676])

def plot_cc_err (cc_arr, title, theorycurve = 0):	
	'''cc_arr has shape(4, 2, 11)'''
	cc_arr = cc_arr[:,:,:-1]
	factor=2.0*pi/(1+ell_arr[:-1])
	chisq=0
	f=figure(figsize=(8,5))
	ax=f.add_subplot(111)
	for Wx in arange(1,5):
		cc, err = cc_arr[Wx-1]
		ax.errorbar(ell_arr[:-1]+5.0*(Wx-3), cc*factor, err*factor, label='W%i'%(Wx), fmt='o')
		chisq+=sum((cc/err)**2)
	if type(theorycurve) != int:
		plot(theorycurve[0], theorycurve[1], label='theory')
	SNR = sqrt(chisq)
	print SNR
	ax.plot((0,2000),(0,0),'k--')
	ax.set_xlabel(r'$\ell$')
	ax.set_ylabel(r'$\ell\rm{C(\ell)}$')
	ax.set_xlim(0,2000)
	legend(fontsize=10)
	#ax.set_ylim(-1e-5,1e-5)
	ax.set_title('%s (SNR=%.2f)'%(title, SNR))
	savefig(plot_dir+'crosscorr_%s.jpg'%(title))
	close()
	
if HM857:
	print 'HM857'
	fn = tSZ_dir+'cc_857.npy'
	def HM857_fcn (Wx):
		mask=maskGen(Wx)
		sigmaG1=sigmaG_arr[-1]
		sigmaG2=sigmaG_arr[-1]
		fmask2=fmask2_arr[Wx-1]
		fsky=fsky_arr[Wx-1]
		map1=load(tSZ_dir+'planck/Planck857_HM1_CFHTLS_W%i.npy'%(Wx))*mask
		map2=load(tSZ_dir+'planck/Planck857_HM2_CFHTLS_W%i.npy'%(Wx))*mask
		cc_857 = WLanalysis.CrossCorrelate(map1, map2, edges=edgesGen(Wx), sigmaG1=sigmaG1, sigmaG2=sigmaG2)[1]/fmask2
		err_857 = sqrt(2)*theory_err(map1, map2, Wx, fmask2, fsky, sigmaG1, sigmaG2)
		return cc_857, err_857
	#cc_857 = array(map(HM857_fcn, range(1,5)))#cc_857.shape=(4, 2, 11)
	#save(fn, cc_857)
	cc_arr = load(fn)
	plot_cc_err (cc_arr, 'Cell_857')
	
if yx857:
	print 'yx857'
	for ip in range(4):
		def yx857_ip(Wx):
			sigmaG1=sigmaG_arr[-1]
			sigmaG2=sigmaG_arr[ip]
			CIBmap = CIBGen(Wx)
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
			fsky=sum(mask)/sizes[Wx-1]**2*sizedeg_arr[Wx-1]/41253.0
			icc_yx857 = WLanalysis.CrossCorrelate(CIBmap, tmap, edges=edgesGen(Wx), sigmaG1=sigmaG1, sigmaG2=sigmaG2)[1]/fmask2
			ierr_yx857 = theory_err(CIBmap, tmap, Wx, fmask2, fsky, sigmaG1, sigmaG2)
			return icc_yx857, ierr_yx857
		
		fn = tSZ_dir+'cc_yx857_%s.npy'%(prefix_arr[ip])
		#cc_yx857 = array(map(yx857_ip, range(1,5)))
		#save(fn, cc_yx857)
		cc_arr = load(fn)
		plot_cc_err (cc_arr, 'Cell_y_857_%s'%(prefix_arr[ip]))
	
		
if kx857:
	print 'kx857'
	def kx857_fcn (Wx):
		mask=maskGen(Wx)
		sigmaG1=1.0
		sigmaG2=sigmaG_arr[-1]
		fmask2=fmask2_arr[Wx-1]
		fsky=fsky_arr[Wx-1]
		map1=kmapGen(Wx)*mask
		map2=CIBGen(Wx)*mask
		cc = WLanalysis.CrossCorrelate(map1, map2, edges=edgesGen(Wx), sigmaG1=sigmaG1, sigmaG2=sigmaG2)[1]/fmask2
		err = theory_err(map1, map2, Wx, fmask2, fsky, sigmaG1, sigmaG2)
		return cc, err
	fn = tSZ_dir+'cc_kx857.npy'
	#cc_kx857 = array(map(kx857_fcn, range(1,5)))
	#save(fn, cc_kx857)
	cc_arr = load(fn)
	plot_cc_err (cc_arr, 'Cell_kappa_857')
	
if cc_yxk:
	print 'yxk'
	ell_JCH, halo1, halo2, C_tot = genfromtxt(tSZ_dir+'CellykappaCFHTLS_WMAP9_Jiadndz_zcut13.txt').T#CellykappaCFHTLS_Planck15_Jiadndz_zcut13.txt
	for ip in range(4):
		def txk (Wx):
			sigmaG1=sigmaG_arr[Wx]
			sigmaG2=1.0
			tmap = tmapGen(prefix_arr[ip], Wx)
			if ip==3:
				tmap[abs(tmap)>1.0]=0
				mask = JCHmaskGen(Wx)
			else:
				mask = maskGen(Wx)
			kmap = kmapGen(Wx)*mask
			tmap *=mask
			fmask2 = sum(mask**2)/sizes[Wx-1]**2
			fsky = sum(mask)/sizes[Wx-1]**2*sizedeg_arr[Wx-1]/41253.0
			edges = edgesGen(Wx)
			CC_signal = WLanalysis.CrossCorrelate(tmap, kmap, edges=edges, sigmaG1=sigmaG1, sigmaG2=sigmaG2)[1]/fmask2
			CC_err = theory_err (tmap, kmap, Wx, fmask2, fsky, sigmaG1, sigmaG2)
			return CC_signal, CC_err
		fn = tSZ_dir+'cc_yxk_%s.npy'%(prefix_arr[ip])
		#cc_all = array(map(txk, range(1,5)))
		#save(fn, cc_all)
		cc_arr = load(fn)
		plot_cc_err (cc_arr, 'Cell_kappa_y_%s'%(prefix_arr[ip]), theorycurve=[ell_JCH, C_tot])

if SNR_calc:
	ell_JCH0, halo1, halo2, C_tot0 = genfromtxt(tSZ_dir+'CellykappaCFHTLS_WMAP9_Jiadndz_zcut13.txt').T
	ell_JCH = concatenate([[0,],ell_JCH0,[1e6,]])
	C_tot = concatenate([[0,],C_tot0,[0,]])
	Cinterp = interpolate.interp1d(ell_JCH, C_tot)
	def theoryGen(Wx):
		size = sizes[Wx-1]
		y, x = np.indices((size, size))
		center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
		center+=0.5
		r = np.hypot(x - center[0], y - center[1])
		r *= 360./sqrt(sizedeg_arr[Wx-1])
		Cmat = Cinterp(r)
		Cth = WLanalysis.azimuthalAverage(Cmat, edges = edgesGen(Wx))[1][:-1]
		return Cth
	theory_all = concatenate(map(theoryGen, range(1,5)))
	chisq_model_fcn = lambda A, CC, err: sum((CC-A*theory_all)**2/err**2)
	figure(figsize=(16,10))
	for ip in range(4):
		factor=2.0*pi/(1+ell_arr)[:-1]
		factors=concatenate(repeat(factor,4).reshape(-1,4).T)
		ell_arr4 = concatenate(repeat(ell_arr,4).reshape(-1,4).T)
		fn = tSZ_dir+'cc_yxk_%s.npy'%(prefix_arr[ip])
		cc_arr = concatenate(load(fn)[:,0,:-1])*factors
		err_arr = concatenate(load(fn)[:,1,:-1])*factors
		A_out = op.minimize(chisq_model_fcn, 1.0, args=(cc_arr, err_arr))
		A_min = A_out.x
		chisq_model = A_out.fun
		chisq_null = sum((cc_arr/err_arr)**2)
		SNR = sqrt(chisq_null-chisq_model)
		temp = r'{4}: A={0:.2f}, $\chi^2_0$={1:.2f}, $\chi^2_A$={2:.2f}, SNR={3:.2f}'.format(float(A_min), chisq_null, chisq_model, SNR, prefix_arr[ip])
		print temp
		
		subplot(2,2,ip+1)
		#errorbar(range(40), cc_arr, err_arr, label='data')
		#plot(range(40), theory_all, label='theory (A=1)')
		#plot(range(40), theory_all*A_min, label='theory (A=%.2f)'%(A_min))
		#for aa in (9.5, 19.5, 29.5):
			#plot([aa,aa],[-amax(cc_arr)*1.2, amax(cc_arr)*1.2],'k-')

		
		######## inverse weighted
		errK_arr = err_arr.reshape(4,-1)
		CC_arr = cc_arr.reshape(4,-1)
		Cth_arr = theory_all.reshape(4,-1)
		weightK = 1/errK_arr**2/sum(1/errK_arr**2, axis=0)
		CC_mean = sum(CC_arr*weightK,axis=0)
		err_mean = sqrt(1.0/sum(1/errK_arr**2, axis=0))
		Cth_mean = sum(Cth_arr*weightK,axis=0)
		errorbar(ell_arr[:-1], CC_mean, err_mean, label='data')
		plot(ell_arr[:-1], Cth_mean, label='theory (A=1)')
		plot(ell_arr[:-1], Cth_mean*A_min, label='theory (A=%.2f)'%(A_min))

		plot((0,2000),zeros(2),'k--')
		title(temp)
		legend(fontsize=10)
		ylabel('ell*C')
		if ip>1:
			xlabel('ell')
	savefig(plot_dir+'fitmodel_data_inverseSum.jpg')
	close()
	
	
		
		