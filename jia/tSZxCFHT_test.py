##########################################################
### This code is for cross correlate CFHT with Planck tSZ. 
### It does the following:
### 1) put tSZ to a grid same as CFHT size
### 2) cross correlate CFHT with tSZ

import numpy as np
from scipy import *
from pylab import *
import os
import WLanalysis
from scipy import interpolate


###################### knobs###########################
plot_crosscorrelate_all = 1
testCC = 0
test_powspec = 0
#######################################################


kSZ_dir = '/Users/jia/CFHTLenS/kSZ/newfil/'
#kSZ_dir = '/Users/jia/CFHTLenS/kSZ/ellmax3000/'
plot_dir = '/Users/jia/CFHTLenS/plot/tSZxCFHT/newfil/'#ellmax3000/'
kSZCoordsGen = lambda i: WLanalysis.readFits(kSZ_dir+'LGMCA_W%i_flipper8192_kSZfilt_squared_toJia.fit'%(i))
#kSZCoordsGen = lambda i: WLanalysis.readFits(kSZ_dir+'kSZ2_W%i_ellmax3000.fit'%(i))
#noiseCoordsGen = lambda i: WLanalysis.readFits(kSZ_dir+'kSZ2_noise_W%i_ellmax3000.fit'%(i))
#offsetCoordsGen = lambda i: WLanalysis.readFits(kSZ_dir+'kSZ2_W%i_ellmax3000_offset.fit'%(i))

kmapGen = lambda i: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/W%i_KS_1.3_lo_sigmaG10.fit'%(i))
bmodeGen = lambda i: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/W%i_Bmode_1.3_lo_sigmaG05.fit'%(i))
galnGen = lambda i: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/W%i_galn_1.3_lo_sigmaG05.fit'%(i))
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
sizes = (1330, 800, 1120, 950)
PPR512=8468.416479647716
PPA512=2.4633625
edgesGen = lambda Wx: linspace(5,80,11)*sizes[Wx-1]/1330.0
rad2pix=lambda x, size: around(size/2.0-0.5 + x*PPR512).astype(int)
sigmaG_arr = [0.0, 1.0, 2.0, 5.0]
fmask_arr = [0.74221415941996738, 0.64173278932078071, 0.66676100127555149, 0.4807035716069451]
#fmask2=sum(maskGen(Wx)**2)/sizes[Wx-1]**2
fmask2_arr = [0.68441507160425807, 0.53272247188649302, 0.59883023828975701, 0.41809094369516109]

#######################################################
###### convert from txt to fits #######################
#######################################################
#for i in range(1,5):
	#print 'txt2fits',i
	#fn = kSZ_dir+'LGMCA_W%i_flipper8192_kSZfilt_squared_toJia'%(i)
	#data = genfromtxt(fn+'.txt')
	#WLanalysis.writeFits(data, fn+'.fit')
	
#kSZCoordsGen = lambda i: genfromtxt(kSZ_dir+'kSZ2_W%i_ellmax3000.txt'%(i))
#noiseCoordsGen = lambda i: genfromtxt(kSZ_dir+'kSZ2_noise_W%i_ellmax3000.txt'%(i))
#offsetCoordsGen = lambda i: genfromtxt(kSZ_dir+'kSZ2_W%i_ellmax3000_offset.txt'%(i))

#txt2fits_kSZ = lambda Wx: WLanalysis.writeFits(kSZCoordsGen(Wx),kSZ_dir+'kSZ2_W%i_ellmax3000.fit'%(Wx))
#txt2fits_noise = lambda Wx: WLanalysis.writeFits(noiseCoordsGen(Wx),kSZ_dir+'kSZ2_noise_W%i_ellmax3000.fit'%(Wx))
#txt2fits_offset = lambda Wx: WLanalysis.writeFits(offsetCoordsGen(Wx),kSZ_dir+'kSZ2_W%i_ellmax3000_offset.fit'%(Wx))
#map(txt2fits_kSZ,range(1,5))
#map(txt2fits_noise,range(1,5))
#map(txt2fits_offset,range(1,5))
#######################################################


def list2coords(radeclist, Wx, offset=False):
	size=sizes[Wx-1]
	xy = zeros(shape = radeclist.shape)
	if offset:
		center = 0.5*(amin(radeclist,axis=0)+amax(radeclist, axis=0))
	else:
		center = centers[Wx-1]
	f_Wx = WLanalysis.gnom_fun(center)
	#xy = degrees(array(map(f_Wx,radeclist)))
	xy = array(map(f_Wx,radeclist))
	xy_pix = rad2pix(xy, size)
	return xy_pix

def interpGridpoints (xy, values, newxy, method='nearest'):
	newvalues = interpolate.griddata(xy, values, newxy, method=method)
	return newvalues

def plotimshow(img,ititle,vmin=0, vmax=0.05):		 
	 #if vmin == None and vmax == None:
	imgnonzero=img[nonzero(img)]
	if vmin == None:
		std0 = std(imgnonzero)
		x0 = median(imgnonzero)
		vmin = x0-3*std0
		vmax = x0+3*std0
	im=imshow(img,interpolation='nearest',origin='lower',aspect=1,vmin=vmin,vmax=vmax)
	colorbar()
	title(ititle,fontsize=16)
	savefig(plot_dir+'%s.jpg'%(ititle))
	close()	

def kSZmapGen (Wx, noise=False, offset=False, method='nearest'):
	#print Wx
	size=sizes[Wx-1]
	if noise:
		
		kSZmap_fn = kSZ_dir+'kSZmap_noise_W%i_%s.fit'%(Wx,method)
	elif offset:
		kSZmap_fn = kSZ_dir+'kSZmap_offset_W%i_%s.fit'%(Wx,method)
	else:	
		kSZmap_fn = kSZ_dir+'kSZmap_W%i_%s.fit'%(Wx,method)
	isfile_kmap, kSZmap = WLanalysis.TestFitsComplete(kSZmap_fn, return_file = True)
	if isfile_kmap == False:
		if noise:
			kSZCoord = noiseCoordsGen(Wx)
		elif offset:
			kSZCoord = offsetCoordsGen(Wx)
		else:
			kSZCoord = kSZCoordsGen(Wx)
		radeclist = kSZCoord[:,:-1]
		values = kSZCoord.T[-1]
		xy = list2coords(radeclist, Wx, offset=offset)
		X,Y=meshgrid(range(size),range(size))
		X=X.ravel()
		Y=Y.ravel()
		newxy=array([X,Y]).T
		newvalues = interpGridpoints (xy, values, newxy,method=method)
		kSZmap = zeros(shape=(size,size))
		kSZmap[Y,X]=newvalues
		WLanalysis.writeFits(kSZmap, kSZmap_fn)
	#plotimshow(kSZmap, 'kSZmap_W%i_%s_Noise%s'%(Wx,method,noise))
	kSZmap[isnan(kSZmap)]=0.0
	if offset:
		kSZmap = kSZmap.T
	return kSZmap

def KSxkSZ (Wx, method='nearest', sigmaG = 0.0):
	print 'KSxkSZ',Wx
	KS = kmapGen(Wx)
	Bmode = bmodeGen(Wx)
	kSZ = kSZmapGen (Wx, method=method)
	noise = kSZ
	offset = kSZ
	###adhoc fix 2014-09-05
	#noise = kSZmapGen (Wx, method=method, noise=True)
	#offset = kSZmapGen (Wx, method=method, offset=True)

	## masking
	galn = galnGen(Wx)
	mask = ones(shape=galn.shape)
	idx = where(galn<0.5)
	mask[idx] = 0
	## smoothed masking
	mask_smooth = WLanalysis.smooth(mask, sigmaG*PPA512)
	KS *= mask_smooth
	noise *= mask_smooth
	kSZ *= mask_smooth
	offset *= mask_smooth
	Bmode *= mask_smooth
	
	#plotimshow(KS,'Masked_KS_%s_sigmaG%s_W%i'%(method,sigmaG,Wx), vmin=-0.01, vmax=0.07)
	#plotimshow(noise,'Masked_noise_%s_sigmaG%s_W%i'%(method,sigmaG,Wx),vmin=None,vmax=None)
	#plotimshow(offset,'Masked_offset_%s_sigmaG%s_W%i'%(method,sigmaG,Wx),vmin=None,vmax=None)
	#plotimshow(kSZ,'Masked_kSZ_%s_sigmaG%s_W%i'%(method,sigmaG,Wx),vmin=None,vmax=None)
	
	sizedeg = (sizes[Wx-1]/512.0)**2*12.0
	#fmask = sum(mask_smooth)/galn.shape[0]**2
	fmask = fmask_arr[Wx-1]
	fsky = fmask*sizedeg/41253.0# 41253.0 deg is the full sky in degrees

	edges = edgesGen(Wx)
	ell_arr, CCK = WLanalysis.CrossCorrelate (KS,kSZ,edges=edges)
	ell_arr, CCB = WLanalysis.CrossCorrelate (KS,noise,edges=edges)
	ell_arr, CCO = WLanalysis.CrossCorrelate (KS,offset,edges=edges)
	ell_arr, CCBMODE = WLanalysis.CrossCorrelate (Bmode, kSZ, edges=edges)
	CCK /= fmask2_arr[Wx-1]
	CCB /= fmask**2
	CCO /= fmask**2
	CCBMODE /= fmask**2

	# error
	autoK = WLanalysis.PowerSpectrum(KS, sizedeg = sizedeg, edges=edges)[-1]/fmask**2
	autokSZ = WLanalysis.PowerSpectrum(kSZ, sizedeg = sizedeg, edges=edges)[-1]/fmask**2
	autoB = WLanalysis.PowerSpectrum(noise, sizedeg = sizedeg, edges=edges)[-1]/fmask**2
	autoO = WLanalysis.PowerSpectrum(offset, sizedeg = sizedeg, edges=edges)[-1]/fmask**2
	autoBMODE = WLanalysis.PowerSpectrum(Bmode, sizedeg = sizedeg, edges=edges)[-1]/fmask**2

	d_ell = ell_arr[1]-ell_arr[0]
	errK = sqrt(autoK*autokSZ/fsky/(2*ell_arr+1)/d_ell)
	errB = sqrt(autoK*autoB/fsky/(2*ell_arr+1)/d_ell)
	errO = sqrt(autoK*autoO/fsky/(2*ell_arr+1)/d_ell)
	errBMODE = sqrt(autoBMODE*autokSZ/fsky/(2*ell_arr+1)/d_ell)
	return ell_arr, CCK, CCB, errK, errB, CCO, errO, CCBMODE, errBMODE

def CrossPower(CCK, CCB, errK, errB, method='nearest', sigmaG=1.0, noise='noise'):
	
	f=figure(figsize=(8,6))
	ax=f.add_subplot(111)

	ax.errorbar(ell_arr, CCK, errK, fmt='o',color='b', label=r'$\kappa\times\,kSZ$  ')
	ax.errorbar(ell_arr, CCB, errB, fmt='o',color='r',label=r'$%s\times\,kSZ$'%(noise))

	leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=0)
	leg.get_frame().set_visible(False)
	#ax.set_xscale('log')
	ax.set_xlim(0,3000)
	ax.set_xlabel(r'$\ell$', fontsize=16)
	ax.set_ylabel(r'$\ell(\ell+1)P_{n\kappa}(\ell)/2\pi$', fontsize=16)
	ax.set_title('%s, %s arcmin'%(method, sigmaG))
	ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	savefig(plot_dir+'kSZxCFHT_%s_sigmaG%s_%s.jpg'%(method,sigmaG, noise))
	close()

if plot_crosscorrelate_all:
	for method in ('nearest',):#'linear','cubic'):
		for sigmaG in (0.0,):#sigmaG_arr:#
			print 'method, sigmaG', method, sigmaG
			CC_arr = array([KSxkSZ(Wx, method=method, sigmaG=sigmaG) for Wx in range(1,5)])
			#CC_arr rows: 0 ell_arr, 1 CCK, 2 CCB, 3 errK, 4 errB, 5 CCO, 6 errO, 7 CCBMODE, 8 errBMODE

			errK_arr = CC_arr[:,3]
			errB_arr = CC_arr[:,4]#pure instrumentation noise
			errO_arr = CC_arr[:,6]#using offset map
			errBMODE_arr = CC_arr[:,8]
			
			weightK = 1/errK_arr/sum(1/errK_arr, axis=0)
			weightB = 1/errB_arr/sum(1/errB_arr, axis=0)
			weightO = 1/errO_arr/sum(1/errO_arr, axis=0)
			weightBMODE = 1/errBMODE_arr/sum(1/errBMODE_arr, axis=0)
			
			errK = 1/sum(1/errK_arr, axis=0)
			errB = 1/sum(1/errB_arr, axis=0)
			errO = 1/sum(1/errO_arr, axis=0)
			errBMODE = 1/sum(1/errBMODE_arr, axis=0)
			
			ell_arr = CC_arr[0,0]
			CCBMODE = sum(CC_arr[:,7]*weightBMODE,axis=0)
			CCK = sum(CC_arr[:,1]*weightK,axis=0)
			CCB = sum(CC_arr[:,2]*weightB,axis=0)#pure instrumentation noise
			CCO = sum(CC_arr[:,5]*weightO,axis=0)#using offset map
			
			#CrossPower(CCK, CCB, errK, errB, method=method, sigmaG=sigmaG, noise='noise')
			#CrossPower(CCK, CCO, errK, errO, method=method, sigmaG=sigmaG, noise='offset')
			CrossPower(CCK, CCBMODE, errK, errBMODE, method=method, sigmaG=sigmaG, noise='Bmode')
			
			#text_arr = array([ell_arr, CCK, CCO, CCB, CCBMODE, errK, errO, errB, errBMODE]).T
			#savetxt(kSZ_dir+'CrossCorrelate_%s_sigmaG%02d.txt'%(method, sigmaG*10), text_arr, header='ell\tkSZxkappa\toffsetxkappa\tnoisexkappa\tkSZxBmode\terr(kSZxkappa)\terr(offsetxkappa)\terr(noisexkappa)\terr(kSZxBmode)')
			
			
			#####################################################
			##test 7/28/2014 plot out each power spectrum:
			#f=figure(figsize=(8,6))
			#ax=f.add_subplot(111)
			#colors=['r','b','g','m']
			#for i in range(4):
				#plot (ell_arr, CC_arr[i,1], colors[i]+'-', label='k x kSZ W%i'%(i+1))
				#plot (ell_arr, CC_arr[i,2], colors[i]+'.', label='k x noise W%i'%(i+1))
				#plot (ell_arr, CC_arr[i,5], colors[i]+'--', label='k x offset W%i'%(i+1))
			#leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
			#leg.get_frame().set_visible(False)
			##ax.set_xscale('log')
			#ax.set_xlabel(r'$\ell$', fontsize=16)
			#ax.set_ylabel(r'$\ell(\ell+1)P_{n\kappa}(\ell)/2\pi$', fontsize=16)
			#ax.set_title('%s, %s arcmin'%(method, sigmaG))
			#ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
			##show()
			#savefig(plot_dir+'kSZxCFHT_byWx_%s_sigmaG%s_Bmode.jpg'%(method,sigmaG))
			#close()
			#####################################################

if testCC:
	Wx=1
	#kSZCoord=genfromtxt('/Users/jia/CFHTLenS/kSZ/Jiatest_W1.txt')
	#kSZCoord=genfromtxt('/Users/jia/CFHTLenS/kSZ/ellmax3000/Jiatest_W1_hires.txt')
	#radeclist = kSZCoord[:,:-1]
	#values = kSZCoord.T[-1]
	
	#size=sizes[Wx-1]
	#xy = list2coords(radeclist,Wx)
	#X,Y=meshgrid(range(size),range(size))
	#X=X.ravel()
	#Y=Y.ravel()
	#newxy=array([X,Y]).T
	
	#def plot_test_ps (method):
		#print method
		#newvalues = interpGridpoints (xy, values, newxy, method=method)
		#kSZmap = zeros(shape=(size,size))
		#kSZmap[Y,X]=newvalues
		#edges = edgesGen(Wx)
		#sizedeg = (sizes[Wx-1]/512.0)**2*12.0
		#ell_arr, ps = WLanalysis.PowerSpectrum(kSZmap, sizedeg=sizedeg, edges=edges)
		##WLanalysis.writeFits(kSZmap, kSZ_dir+'test_hires_powerspec_%s.fit'%(method))
		
		##plotimshow(kSZmap, 'test_powerspec_%s'%(method), vmin=None, vmax=None)
		
		#savetxt(kSZ_dir+'test_hires_powspec_W1_%s.txt'%(method),array([ell_arr, ps, ps/(ell_arr*(ell_arr+1)/(2*pi))]).T)
		#return ell_arr, ps
	sizedeg = (sizes[Wx-1]/512.0)**2*12.0
	kSZmap = lambda method: WLanalysis.readFits(kSZ_dir+'test_hires_powerspec_%s.fit'%(method))
	ps_prebin = lambda method: WLanalysis.PowerSpectrum(kSZmap(method),sizedeg=sizedeg)
	ps_afterbin = lambda method: WLanalysis.PowerSpectrum_Pell_binning(kSZmap(method),sizedeg=sizedeg)

	method_arr = ('nearest','linear','cubic')
	ellps_arr = array(map(ps_prebin, method_arr))
	ellps_after_arr = array(map(ps_afterbin, method_arr))
	ell_arr = ellps_arr[0][0]
	
	f=figure(figsize=(8,6))
	ax=f.add_subplot(111)
	ax.set_xlabel(r'$\ell$', fontsize=16)
	ax.set_ylabel(r'$\ell(\ell+1)P_{\kappa\kappa}(\ell)/2\pi$', fontsize=16)
	ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	i=0
	for ps1 in (ellps_arr[0,1],):
		ax.plot(ell_arr,ps1,label='bin P '+method_arr[i])
		i+=1
	i=0
	for ps in (ellps_after_arr[0,1],):
		ax.plot(ell_arr,ps,label='bin P(l+1)l '+method_arr[i])
		i+=1
	#ax.set_title(method)
	legend(loc=0)
	#ax.set_xscale('log')
	ax.set_xlim(0,3500)
	savefig(plot_dir+'test_hires_powspec_W1_binning.jpg')
	close()

if test_powspec:
	f=figure(figsize=(10,8))
	for Wx in range(1,5):
		print Wx
		kSZ = kSZmapGen (Wx)
		mask = zeros(shape=kSZ.shape)
		mask[25:-25,25:-25] = 1
		mask10 = WLanalysis.smooth(mask,10)
		mask20 = WLanalysis.smooth(mask,20)
		#plotimshow(kSZ,'kSZ_map_W%i'%(Wx),vmin=None,vmax=None)
		sizedeg = (sizes[Wx-1]/512.0)**2*12.0
		edges = edgesGen(Wx)
		ell_arr, autokSZ = WLanalysis.PowerSpectrum(kSZ, sizedeg = sizedeg, edges=edges)
		ell_arr, autokSZ_smooth10 = WLanalysis.PowerSpectrum(kSZ*mask10, sizedeg = sizedeg, edges=edges)
		ell_arr, autokSZ_smooth20 = WLanalysis.PowerSpectrum(kSZ*mask20, sizedeg = sizedeg, edges=edges)
		
		ax=f.add_subplot(2,2,Wx)
		ax.plot(ell_arr,autokSZ,label='no smooth')
		ax.plot(ell_arr,autokSZ_smooth10,'--',label='4 arcmin')
		ax.plot(ell_arr,autokSZ_smooth20,'-.',label='8 arcmin')
		#ax.set_xscale('log')
		#ax.set_yscale('log')
		ax.set_xlabel(r'$\ell$')
		ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$')
		ax.set_xlim(ell_arr[0],ell_arr[-1])
		title('W%i'%(Wx))
		if Wx==1:
			leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':12},loc=0)
			leg.get_frame().set_visible(False)
		#savefig(plot_dir+'test_powspec_W%i.jpg'%(Wx))
		#savetxt(kSZ_dir+'test_smooth_AutoPowspec_W%i.txt'%(Wx), array([ell_arr,autokSZ_smooth]).T, header='ell\tell(ell+1)P/2pi')
	savefig(plot_dir+'test_powspec_smooth_mult.jpg')
	close()