#!python
# Jia Liu 2015/02/19
# This code calculates the model for cmb lensing x weak lensing

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

cmb_dir = '/Users/jia/weaklensing/cmblensing/'
plot_dir = '/Users/jia/Desktop/cmblensing/'
#######################################
########## cosmo params ###############
#######################################

z0, z1 = genfromtxt(cmb_dir+'dndz_CFHT_nocutPeak.txt').T
#z0, z1 = genfromtxt(cmb_dir+'dndz_CFHT.txt').T
#z0, z1 = genfromtxt(cmb_dir+'dndz_CFHT_nocut.txt').T

##z_arr, W_wl0, W_cmb_arr = load(cmb_dir+'test_W_planck.npy')#test_W_hinshaw
##W_wl = interpolate.interp1d(z_arr, W_wl0)
##W_cmb = interpolate.interp1d(z_arr, W_cmb_arr)

####### planck 2015 TT, TE, EE + lowP
#OmegaM = 0.3156
#H0 = 67.27
#Ptable = genfromtxt(cmb_dir+'P_delta_smith03_revised')

########### colin params ##############
OmegaM = 0.317 
H0 = 65.74
Ptable = genfromtxt(cmb_dir+'P_delta_smith03_revised_colinparams')

############ Hinshaw ##################
#OmegaM = 0.282
#H0 = 69.7
#Ptable = genfromtxt(cmb_dir+'P_delta_Hinshaw')

########### WMAP9+BAO #################
#OmegaM = 0.293
#H0 = 68.8
#Ptable = genfromtxt(cmb_dir+'P_delta_WMAPBAO')

########## Heymann++ ##################
#OmegaM = 0.271
#H0 = 73.8
#Ptable = genfromtxt(cmb_dir+'P_delta_Heymans')

#######################################
####### constants & derived params
#######################################
OmegaV = 1.0-OmegaM
h = H0/100.0
c = 299792.458#km/s
Gnewton = 6.674e-8#cgs cm^3/g/s

H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV))
DC = lambda z: c*quad(H_inv, 0, z)[0] # comoving distance Mpc
z_ls = 1100 #last scattering

#######################################
##### find my own curve for dn/dz
#######################################
#dndz_Hand = lambda z: 0.688*(z**0.531+z**(0.531*7.810))/(z**7.810+0.517)
#dndz_VW = lambda z: 1.5*exp(-(z-0.7)**2/0.32**2)+0.2*exp(-(z-1.2)**2/0.46**2)
########### my dndz interpolation 

########## lensing kernel using dndz ##########

z0 = concatenate([[0,], z0, linspace(z0[-1]*1.2, 1200,100)])
z1 = concatenate([[0,], z1, 1e-128*ones(100)])
dndz_interp = interpolate.interp1d(z0, z1,kind='cubic')
integrand = lambda zs, z: dndz_interp(zs)*(1-DC(z)/DC(zs))
W_wl_fcn = lambda z: 1.5*OmegaM*H0**2*(1+z)*H_inv(z)*DC(z)/c*quad(integrand, z, 4.5, args=(z,))[0]
z_arr = linspace(1e-5, 4.5, 100)
W_wl0 = array(map(W_wl_fcn, z_arr))
print 'done interpolating W_wl'
W_wl = interpolate.interp1d(z_arr, W_wl0)
W_cmb = lambda z: 1.5*OmegaM*H0**2*(1+z)*H_inv(z)*DC(z)/c*(1-DC(z)/DC(z_ls))

aa = array([1/1.05**i for i in arange(33)]) # scale factor
zz = 1.0/aa-1 # redshifts
kk = Ptable.T[0]
iZ, iK = meshgrid(zz,kk)
Z, K = iZ.flatten(), iK.flatten()
P_deltas = Ptable[:,1:].flatten()
#Pmatter_interp = interpolate.CloughTocher2DInterpolator(array([K, Z]).T, P_deltas/(K/2.0/pi)**3)
#Pmatter = lambda k, z: Pmatter_interp (k/h, z)*h**3
Pmatter_interp = interpolate.CloughTocher2DInterpolator(array([K*h, Z]).T, 2.0*pi**2*P_deltas/(K*h)**3)
Pmatter = lambda k, z: Pmatter_interp (k, z)

Ckk_integrand = lambda z, ell: 1.0/(H_inv(z)*c*DC(z)**2)*W_wl(z)*W_cmb(z)*Pmatter(ell/DC(z), z)

ell_arr = linspace(1e-5, 2000, 200)
Ckk_arr = array([quad(Ckk_integrand, 0.002, 2.0 , args=(iell))[0] for iell in ell_arr])#3.7
#plot(ell_arr, Ckk_arr*ell_arr)
#show()
#save(cmb_dir+'model_planck2015+Hchi_Hinshaw.npy',array([ell_arr, Ckk_arr]))



######################
W_cmb_arr = array(map(W_cmb, z_arr))
CH_ell,CH_1h, CH_2h, CH_total, CH_linear = genfromtxt(cmb_dir+'CellkappaCMBkappaCFHTLS_Jia_nocutPeak0227.txt').T
z_arr_CH, wl_CH, cmb_CH = genfromtxt(cmb_dir+'Wkappa_gal_CMB_nocutPeak.txt').T

figure(figsize=(8,8))
subplot(211)
plot(z_arr, W_wl0,'.', label='Jia (gal)')
plot(z_arr_CH, wl_CH, label='Colin (gal)')
plot(z_arr, W_cmb_arr, '.',label='Jia (cmb)')
plot(z_arr_CH, cmb_CH, label='Colin (cmb)')
xlabel('z')
ylabel('W_wl')
legend(loc=0, fontsize=12)
title('Colin Params, nocutPeak')
subplot(212)
plot(ell_arr, Ckk_arr*ell_arr*1e6,'-', label='Jia')
#plot(ell_arr, Ckk_arrZ2*ell_arr*1e6,'.', label='Jia (Colin params, z=[0,2])')
plot(CH_ell, CH_total*1e6, label='Colin')
xlabel('ell')
ylabel('ell*C*1e6')
xlim(0,2000)
legend(loc=0, fontsize=12)
show()
#savefig(plot_dir+('C_comparison_colinparams_nocutPeak.jpg'))
#close()

###################################
############# SNR #################
###################################
#jj=0
#nocut=1
#ell_arr_data = 40.0*WLanalysis.edge2center(linspace(1,50,6))[jj:]
##40=512.0/1330.0*360./(sqrt(12.0))

#ell_arr, Ckk_arr = load(cmb_dir+'model_Hinshaw_nozcutPeak.npy')[:,jj:]

###### colin model
##ell,CH_1h, CH_2h, CH_total, CH_linear = genfromtxt(cmb_dir+'CellkappaCMBkappaCFHTLS_Jia_nocutPeak.txt').T
##ell_arr = ell
##Ckk_arr = CH_total/ell_arr

#model_raw = interpolate.interp1d(ell_arr, Ckk_arr)(ell_arr_data)
#MM = model_raw*ell_arr_data
#model_fit = lambda A:  A*concatenate([MM,MM,MM,MM])
#chisq_model_fcn = lambda A, CC, err: sum((CC-model_fit(A))**2/err**2)
#chisq_model_single_fcn = lambda A, CC, err: sum((CC-A*MM)**2/err**2)

#def plot_elems (Wx, return_chisq_null = False, nocut=nocut):
	#if nocut:
		#idir=cmb_dir+'CC_noZcut/'		
	#else:
		#idir=cmb_dir+'CC_0213/'
	#CC_noise = load(idir+'CFHTxPlanck_lensing_500sim_W%s.npy'%(Wx))[:,jj:]/(ell_arr_data+1)*2*pi
	#CC_signal =load(idir+'CFHTxPlanck_lensing_W%s.npy'%(Wx))[jj:]/(ell_arr_data+1)*2*pi
		
	#CC_err = std(CC_noise,axis=0)
	#CC_noise_mean = mean(CC_noise,axis=0)
	#CCN_cov = np.cov(CC_noise,rowvar=0)
	#chisq_null = sum(mat(CC_signal)*mat(CCN_cov).I*mat(CC_signal).T)
	#if return_chisq_null:
		#return chisq_null
	#else:
		#return CC_signal, CC_err, CC_noise_mean

#datacube = array([plot_elems(Wx, nocut=nocut) for Wx in range(1,5)])
#CC_arr = datacube[:,0,:]
#errK_arr = datacube[:,1,:]

#def find_SNR (CC_arr, errK_arr, nocut=1):
	#weightK = 1/errK_arr**2/sum(1/errK_arr**2, axis=0)
	#CC_mean = sum(CC_arr*weightK,axis=0)
	#err_mean = sqrt(1.0/sum(1/errK_arr**2, axis=0))
	#std_mean = std(CC_arr, axis=0)#0.5*(amax(CC_arr, axis=0)-amin(CC_arr, axis=0))#
	#A_out = op.minimize(chisq_model_fcn, 1.0, args=(concatenate(CC_arr), concatenate(errK_arr)))
	#A_min = A_out.x#0.61741426
	#chisq_model = A_out.fun
	#chisq_null=sum(array([plot_elems(Wx, return_chisq_null=1, nocut=nocut) for Wx in range(1,5)]))
	#SNR = sqrt(chisq_null-chisq_model)
	#print chisq_null, chisq_model
	#return A_min, SNR, CC_mean, err_mean, std_mean
#A_min, SNR, CC_mean, err_mean, std_mean=find_SNR(CC_arr, errK_arr, nocut=nocut)

###### plots #########
#f=figure(figsize=(8,6))
#ax=f.add_subplot(111)
#ax.bar(ell_arr_data, 2*err_mean*1e6, bottom=(CC_mean-err_mean)*1e6, width=ones(len(ell_arr_data))*80, align='center',ec='brown',fc='none',linewidth=1.5, alpha=1.0)#

##bar(ell_arr_data+20, 2*std_mean*1e6, bottom=(CC_mean-std_mean)*1e6, width=ones(len(ell_arr_data))*40, align='center',edgecolor='c',color='c',alpha=0.5)
#seed(584)#good seeds: 6, 16, 25, 41, 53, 128, 502, 584
#ax.plot(ell_arr, zeros(len(ell_arr)), 'k--', linewidth=1)

#for Wx in range(1,5):
	#cc=rand(3)#colors[Wx-1]
	#ax.errorbar(ell_arr_data+(Wx-2.5)*15, CC_arr[Wx-1]*1e6, errK_arr[Wx-1]*1e6, fmt='o',ecolor=cc,mfc=cc, mec=cc, label=r'$\rm W%i$'%(Wx), linewidth=1.2, capsize=0)
#ell_arr[0]=0
#Ckk_arr[0]=0
#ax.plot(ell_arr ,Ckk_arr*ell_arr*1e6,'k-',linewidth=1.2, label=r'$WMAP+{\rm eCMB+BAO}+H_0$')#Hinshaw++2013
#handles, labels = ax.get_legend_handles_labels()
#handles = [handles[i] for i in (1,2,3,4, 0)]
#labels = [labels[i] for i in (1,2,3,4, 0)]
#leg=ax.legend(handles, labels,loc=3,fontsize=14)
#leg.get_frame().set_visible(False)
#ax.set_xlabel(r'$\ell$',fontsize=18)
#ax.set_xlim(1,2000)
#ax.set_ylabel(r'$\ell C_{\ell}^{\kappa_{cmb}\kappa_{gal}}$',fontsize=18)
#ax.set_title('dn/dz nocutPeak, A=%.2f, SNR=%.2f'%(A_min, SNR))#, noZcut
##show()
#savefig(cmb_dir+'paper/CC.jpg')
#close()


########## fit to C, not C*ell_arr

#ell_arr_data = 40.0*WLanalysis.edge2center(logspace(0,log10(50),7))[1:]
##40=512.0/1330.0*360./(sqrt(12.0))

#ell_arr, Ckk_arr = load(cmb_dir+'model_planck2015_nozcutPeak.npy')[:,1:]#
#model_raw = interpolate.interp1d(ell_arr, Ckk_arr)(ell_arr_data)
#MM = model_raw
#model_fit = lambda A:  A*concatenate([MM,MM,MM,MM])
#chisq_model_fcn = lambda A, CC, err: sum((CC-model_fit(A))**2/err**2)
#chisq_model_single_fcn = lambda A, CC, err: sum((CC-A*MM)**2/err**2)

#def plot_elems (Wx, return_chisq_null = False, nocut=1):
	#if nocut:
		#idir=cmb_dir+'CC_noZcut/'
	#else:
		#idir=cmb_dir+'CC_0213/'
	#CC_noise = load(idir+'CFHTxPlanck_logbins_lensing_500sim_W%s.npy'%(Wx))[:,1:]/ell_arr_data**2*2*pi
	#CC_signal =load(idir+'CFHTxPlanck_logbins_lensing_W%s.npy'%(Wx))[1:]/ell_arr_data**2*2*pi
	#CC_err = std(CC_noise,axis=0)
	#CC_noise_mean = mean(CC_noise,axis=0)
	#CCN_cov = np.cov(CC_noise,rowvar=0)
	#chisq_null = sum(mat(CC_signal)*mat(CCN_cov).I*mat(CC_signal).T)
	#if return_chisq_null:
		#print chisq_null
		#return chisq_null
	#else:
		#return CC_signal, CC_err, CC_noise_mean

#datacube = array([plot_elems(Wx) for Wx in range(1,5)])
#CC_arr = datacube[:,0,:]
#errK_arr = datacube[:,1,:]

#def find_SNR (CC_arr, errK_arr, nocut=1):
	#weightK = 1/errK_arr**2/sum(1/errK_arr**2, axis=0)
	#CC_mean = sum(CC_arr*weightK,axis=0)
	##CC_mean = mean(CC_arr,axis=0)
	##err_mean = sqrt(1.0/sum(1/errK_arr**2, axis=0))
	#err_mean = std(CC_arr, axis=0)
	#A_out = op.minimize(chisq_model_fcn, 1.0, args=(concatenate(CC_arr), concatenate(errK_arr)))
	##A_out = op.minimize(chisq_model_single_fcn, 1.0, args=(CC_mean, err_mean))
	#A_min = A_out.x#0.61741426
	#chisq_model = A_out.fun
	#chisq_null=sum(array([plot_elems(Wx, return_chisq_null=1, nocut=nocut) for Wx in range(1,5)]))
	#SNR = sqrt(chisq_null-chisq_model)
	#return A_min, SNR, CC_mean, err_mean
#A_min, SNR, CC_mean, err_mean=find_SNR(CC_arr, errK_arr, nocut=1)

#bar(ell_arr_data, 2*err_mean*ell_arr_data*1e6, bottom=(CC_mean-err_mean)*ell_arr_data*1e6, width=ones(5)*40, align='center',edgecolor='y',color='y',alpha=0.7)

#colors=('r','b','m','g')
#plot(ell_arr, Ckk_arr*ell_arr*1e6,'k-',linewidth=2, label='planck2015, noZcut')
#for Wx in range(1,5):
	#cc=colors[Wx-1]
	#errorbar(ell_arr_data+(Wx-1), CC_arr[Wx-1]*ell_arr_data*1e6, errK_arr[Wx-1]*ell_arr_data*1e6, fmt='o',ecolor=cc,mfc=cc, mec=cc, label='W%i'%(Wx))
##errorbar(ell_arr_data, CC_mean*1e6, err_mean*1e6, label='mean',ecolor='k',mfc='k', mec='k',)
#xlabel('ell')
#ylabel('ell*C*1e6')
#legend(loc=0,fontsize=12)
#show()

##################################
### dN/dz for nocut ##############
##################################

#PzFull = genfromtxt('/Users/jia/Documents/weaklensing/CFHTLenS/catalogue/junk/full_subfields/full_subfield1')[:,-70:]
#zPDF = mean(PzFull, axis=0)
#zcenter = arange(0.025, 3.525, 0.05)
#zcenter_long = concatenate([[0,],zcenter, linspace(3.6, 6.0, 10)])
#zPDF_long = concatenate([[0,],zPDF/0.05,1e-128*ones(10)])
#save(cmb_dir+'dndz_CFHT_nocut.npy',array([zcenter_long, zPDF_long]))

#############################################
###### cross CMB lensing with CFHT galn #####
#############################################

#data_ell_arr = array([  380.06143435, 740.11963531, 1100.17783627,  1460.23603723, 1820.29423819])
#cat_dir = '/Users/jia/weaklensing/CFHTLenS/catalogue/'
#maskGen = lambda Wx: np.load(cmb_dir+'mask/W%i_mask.npy'%(Wx))
#kmapGen = lambda Wx: load(cat_dir+'kmap_W%i_sigma10_noZcut.npy'%(Wx))
##galnGen = lambda i: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_galn_1.3_lo_sigmaG10.fit'%(i))
#galnGen = lambda Wx: load(cmb_dir+'OmoriHolder_galn_W%i.npy'%(Wx))
#galnSGen = lambda Wx: load(cmb_dir+'OmoriHolder_galnSmooth_W%i.npy'%(Wx))
#ptsrcGen = lambda i: np.load(cmb_dir + 'planck/'+'kappamask_flipper2048_CFHTLS_W%i_map.npy'%(i))
#cmblGen = lambda Wx: np.load(cmb_dir+'planck/dat_kmap_flipper2048_CFHTLS_W%i_map.npy'%(Wx))
##fmask2_arr = [0.69589653344034097, 0.56642933419641017, 0.5884087465608, 0.43618153901026568]
#fmask2_arr = [0.7005031375431059, 0.8165671875, 0.6909159757653062, 0.4467756232686981]
#sizes = (1330, 800, 1120, 950)
#PPR512=8468.416479647716
#PPA512=2.4633625
#testedgesGen = lambda Wx: logspace(0,log10(50),15)*sizes[Wx-1]/1330.0#linspace(5,80,11)
#colors=('r','b','m','g')
#for Wx in range(1,5):
##def galnxcmb (Wx):
	#imask=maskGen(Wx)#ptsrcGen(Wx)
	#kmap = kmapGen(Wx)*imask
	#cmblmap = cmblGen(Wx)*imask
	
	## set mean to 0
	#kmap -= mean(kmap)
	#cmblmap -= mean(cmblmap)
	
	#edges = testedgesGen(Wx)
	#ell_arr, CC_signal = WLanalysis.CrossCorrelate(kmap, cmblmap,edges=edges)
	#CC_signal /= fmask2_arr[Wx-1]

	#cc=colors[Wx-1]
	#errorbar(ell_arr, CC_signal/ell_arr*2*pi,1e-6*ones(len(ell_arr)),fmt='o',label='W%i'%(Wx),ecolor=cc,mfc=cc, mec=cc)
	##return ell_arr, CC_signal/ell_arr*1e5*2*pi
##all_galnxcmb = array(map(galnxcmb, range(1,5)))
##ell_arr = all_galnxcmb[0,0]
##errorbar (ell_arr, mean(all_galnxcmb[:,1,:],axis=0), std(all_galnxcmb[:,1,:],axis=0))
#xlabel('ell')
#ylabel('ell*C')
#legend(loc=0,fontsize=12)
#show()

########################################## 
######### various tests ##################
##########################################

## (1) test dndz_interp - pass #########
#z_arr = linspace(1e-2, 4, 100)
#plot(z_arr, dndz_Hand(z_arr),label='Hand+2014')
#plot(z_arr, dndz_VW(z_arr),label='van Waerbeke+2013')
#plot(z_hist[0],z_hist[1]/0.05,label='CFHT',drawstyle='steps-post')
#plot(z_arr, dndz_interp(z_arr), label='JL interpolation')
#legend()
#xlabel('z')
#ylabel('dn/dz')
#xlim(0,2.5)
#show()

########### (2) my attempt to fit to dndz, aborted
#def chisq_dndz_JL (abcA):
	#a, b, c, A = abcA
	#dndz = A*(z0**a+z0**(a*b))/(z0**b+c)
	#diff = sum(abs(dndz - z1))
	#return diff
#abcA_guess = (0.531, 7.810, 0.517, 0.688)
#out = op.minimize(chisq_dndz_JL, abcA_guess)r
#abcA_JL = [ 0.28391205,  6.79531638,  0.73779074,  0.65322337]
####### do interpolation directly

########## (3) calculate  W_wl ##################
######### prepare for interpolation ############

#z0, z1 = genfromtxt(cmb_dir+'dndz_CFHT_nocut.txt').T#dndz_CFHT.txt for zcut0213
#z0 = concatenate([[0,], z0, linspace(z0[-1]*1.2, 1200,100)])
#z1 = concatenate([[0,], z1, 1e-128*ones(100)])
#dndz_interp = interpolate.interp1d(z0, z1,kind='cubic')
##########quad(dndz_interp, 0,3.7)
#integrand = lambda zs, z: dndz_interp(zs)*(1-DC(z)/DC(zs))
######integrand = lambda zs, z: dndz_Hand(zs)*(1-DC(z)/DC(zs))
#W_wl = lambda z: 1.5*OmegaM*H0**2*(1+z)*H_inv(z)*DC(z)/c*quad(integrand, z, 6.0, args=(z,))[0]
#z_arr200 = linspace(1e-5, 4, 200)
#W_wl_arr = array(map(W_wl, z_arr200))
#save(cmb_dir+'W_wl_interp_nocut.npy',array([z_arr200, W_wl_arr]))

##save(cmb_dir+'W_wl_interp_Hand.npy',array([z_arr200, W_wl_arr]))


### (4) W_cmb & W_wl - test pass #############
#z_arr = linspace(1e-2, 4, 100)
#W_cmb_arr = array(map(W_cmb, z_arr))
#plot(z_arr,W_cmb_arr, label='Planck')
#plot(z_arr,W_wl(z_arr)*amax(W_cmb_arr)/amax(W_wl0), label='CFHT')
#legend(loc=0)
#xlabel('z')
#ylabel('W')
#show()

########(4B) compare  W with Colin - somewhat pass

#z_arr_CH, wl_CH, cmb_CH = genfromtxt('/Users/jia/Desktop/cmblensing/Wkappa_gal_CMB.txt.2').T
#z_arrB, W_wl0B = load(cmb_dir+'W_wl_interp_colin.npy')
#Wcmbb=array(map(W_cmb_fcn,z_arr))

#plot(z_arr, W_wl0, 'r-',label='WL (Jia)')
#plot(z_arrB, W_wl0B, 'r.',label='WL (Jia, Colin params)')
#plot(z_arr, Wcmbb, 'r--',label='CMB (Jia)')

#plot(z_arr_CH, wl_CH,'b-',label='WL (Colin)')
#plot(z_arr_CH, cmb_CH,'b--',label='CMB (Colin)')

#xlim(0,2)
##ylim(0, 1.2)
#xlabel('z')
#legend(loc=0)
#ylabel('W')
#show()

#### (5) nicaea matter powspec #######
#for fn in ('WMAPBAO','Heymans', 'smith03_revised', 'smith03_revised_colinparams'):
	#Ptable = genfromtxt(cmb_dir+'P_delta_'+fn)[::5,]#_colinparams,'P_delta_WMAPBAO',P_delta_smith03_revised
	#aa = array([1/1.05**i for i in arange(33)]) # scale factor
	#zz = 1.0/aa-1 # redshifts
	#kk = Ptable.T[0]
	##for i range(len(zz)):
	##loglog(kk, Ptable[:,i+1]/(kk/2/pi)**3)
	#loglog(kk, Ptable[:,6]/(kk/2/pi)**3,label=fn)
	##loglog(kk, Ptable[:,1]/(kk/2/pi)**3,'--')

#xlabel('k [Mpc/h]')
#ylabel('P_delta / (k/2pi)^3')
#legend(loc=0, fontsize=10)
#show()

#### (6) compare C with colin
#ell,CH_1h, CH_2h, CH_total, CH_linear = genfromtxt('/Users/jia/Desktop/cmblensing/CellkappaCMBkappaCFHTLS_Jia.txt.1').T
#Ckk_arr_JL = array([quad(Ckk_integrand, 0.00502525, 1.98992489 , args=(iell))[0] for iell in ell])
#plot(ell, Ckk_arr_JL*ell,label='JL')#(use Colin W^cfht)')
#plot(ell, CH_1h,label='1 halo (Colin)')
#plot(ell, CH_2h,label='2 halo (Colin)')
#plot(ell, CH_total,label='Total (Colin)')
#plot(ell, CH_linear,label='Linear (Colin)')
#xlabel('ell')
#ylabel('ell*C')
#xlim(0,2000)
#legend(loc=0)
#show()

#### (7) check matter power spectrum interpolation 

#kk_newarr = logspace(log10(amin(kk)/h), log10(amax(kk)/h), 100)
#Pz000_interp = array([Pmatter(ikk, 0.05) for ikk in kk_newarr])
#Pz002_interp = array([Pmatter(ikk, 0.2) for ikk in kk_newarr])
#loglog(kk/h, Ptable[:,1:][:,1]/(kk/h/2.0/pi)**3,'b.',label='nicaea (z=0.05)')
#loglog(kk_newarr, Pz000_interp,'b-', label='z=0.05')
#loglog(kk_newarr, Pz002_interp,'r--', label='z=0.2')
#legend(loc=0)
#xlabel('k (Mpc-1)')
#ylabel('P_delta / k^3')
#show()

### (8) matter power spectrum with colin
#####k [h/Mpc]   P(k) [(Mpc/h)^3]

#k_CH, P_CH = genfromtxt('/Users/jia/Desktop/cmblensing/wmap9baosn_max_likelihood_Colinz0_ext_1648_matterpower.dat').T
#f=figure()
#ax=f.add_subplot(211)
#ax2=f.add_subplot(212)
#P_JL = array([Pmatter(ik*h, 0.0) for ik in k_CH])
#ax.loglog(k_CH, P_CH, label='linear (Colin)')
#ax.loglog(k_CH, P_JL*h**3, label='smith03 (Jia)')
#ax.legend(loc=0,fontsize=10)
#ax2.plot(k_CH, P_JL*h**3/P_CH-1)
#ax2.set_xscale('log')
#ax2.set_ylim(-1,1)
#ax2.set_xlabel('ell')
#show()

########(9) lensing kernel for W_gal, zcut vs noZcut
#z_arr200, W_wl_arr=load(cmb_dir+'W_wl_interp_nocut.npy') 
#z_nocutPeak, W_wl_nocutPeak = load(cmb_dir+'W_wl_interp_nocutPeak.npy') 
#z0213, wwl_0213 = load(cmb_dir+'W_wl_interp.npy') 
#z_heymans, wwl_heymans=load(cmb_dir+'W_wl_interp_nocut_heymans.npy')
#plot(z0213,wwl_0213,'-',label='0.2<z<1.3')
#plot(z_arr200, W_wl_arr,'.',label='no z cut')
#plot(z_nocutPeak, W_wl_nocutPeak, '--', label='no z cut (peak)')
#plot(z_heymans, wwl_heymans, '-', label='no z cut (heymans)')
#xlabel('z')
#ylabel('W_gal')
#legend(loc=0)
#show()

######## (10) dndz comparison ###########
#textfile = load(cat_dir+'CFHTLenS_downloads/All_RA_Dec_e12_w_z_m_c.npy')[1:]
#RA, DEC, e1, e2, weight, zB, m, c2 = textfile.T

#zcenter_nocut, zpdf_nocut = genfromtxt(cmb_dir+'dndz_CFHT_nocut.txt').T
#zcenter_0213, zpdf_0213 = genfromtxt(cmb_dir+'dndz_CFHT.txt').T

#hist(zB, range=(0,3.5),bins=50, histtype='step',normed=True,label='peak (no z cut)')
#plot(zcenter_nocut, zpdf_nocut, drawstyle='steps-mid', label='PDF (no z cut)')
#plot(zcenter_0213, zpdf_0213, drawstyle='steps-mid', label='PDF (0.2<z<1.3)')
#xlim(0,3.5)
#legend(loc=0)
#xlabel('z')
#ylabel('dn/dz')
#show()

####### (11) model comparison ########
##prefix_arr = ('planck2015','planck2015_nozcut','planck2015_nozcutPeak','WMAPBAO_nozcut','WMAPBAO_nozcutPeak','Heymans_nozcut')
##prefix_arr = ('H065','H068','H071','H065_Peak','H071_Peak')

#prefix_arr = ('planck2015_nozcutPeak', 'planck2015+P_Hinshaw', 'planck2015+W_Hinshaw','planck2015+Hchi_Hinshaw',)
#i=0
#for ip in prefix_arr:
	#print ip
	#ell, C = load(cmb_dir+'model_%s.npy'%(ip))
	#plot(ell, C*ell*1e6,label=prefix_arr[i])
	#i+=1
#xlabel('ell')
#ylabel('ell*C*1e6')
#legend(loc=0)
##show()
#savefig(plot_dir+'model_planck15_vs_hinshaw.jpg')
#close()

###### (12) understand the model sensitivity to H0
#z_arr = logspace(-5,log10(4),10)
#def testH0 (H0):
	#H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV))
	#DC = lambda z: c*quad(H_inv, 0, z)[0]
	#out = array([1.0/H_inv(z)/DC(z)**2 for z  in z_arr])
	#return out
#H0_arr=linspace(65,73, 20)
#out=array(map(testH0,H0_arr))

#def testOm (OmegaM, H0=67):
	#H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+(1-OmegaM)))
	#DC = lambda z: c*quad(H_inv, 0, z)[0]
	#out = array([1.0/H_inv(z)/DC(z)**2 for z  in z_arr])
	#return out
#out=array(map(testOm,linspace(0.26, 0.33, 5)))

#f=figure()
#ax=f.add_subplot(111)
#for H0 in (65, 67, 69, 71, 73):
	#ax.plot(z_arr, testH0(H0),label=H0)
#legend(loc=0,fontsize=12)
#ax.set_yscale('log')
#ax.set_xscale('log')
#ax.set_xlabel('z')
#ax.set_ylabel('H(z)/Chi(z)**2')
#show()

############## Wcmb sensitivity to H0 - none
#z_arr = linspace(0,4,100)
#def testWcmb (H0, OmegaM=0.3):
	#H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV))
	#DC = lambda z: c*quad(H_inv, 0, z)[0] 
	#W_cmb = lambda z: 1.5*OmegaM*H0**2*(1+z)*H_inv(z)*DC(z)/c*(1-DC(z)/DC(z_ls))
	#out = array([W_cmb(z) for z in z_arr])
	#return out
#f=figure()
#ax=f.add_subplot(111)
#for H0 in (65, 67, 69, 71, 73):
	#ax.plot(z_arr, testWcmb(H0),label=H0)
#legend(loc=0,fontsize=12)
##ax.set_yscale('log')
##ax.set_xscale('log')
#ax.set_xlabel('z')
#ax.set_ylabel('W^cmb')
#show()	