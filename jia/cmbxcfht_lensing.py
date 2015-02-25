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
#######################################
####### planck 2015 TT, TE, EE + lowP
#######################################
H0 = 67.27
OmegaM = 0.3156

########### colin params ##############
#OmegaM = 0.317 
#H0 = 65.74

#######################################
####### constants & derived params
#######################################
OmegaV = 1.0-OmegaM
H0_cgs = H0*1e5/3.08567758e24
# assume 0 radiation
h = H0/100.0
c = 299792.458#km/s
Gnewton = 6.674e-8#cgs cm^3/g/s
rho_c0 = 3.0/8.0/pi/Gnewton*H0_cgs**2#9.9e-30

H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV))
# luminosity distance Mpc
DL = lambda z: (1+z)*c*quad(H_inv, 0, z)[0] 
# comoving distance Mpc
DC_fcn = lambda z: c*quad(H_inv, 0, z)[0] 
#DC = interpolate.interp1d(logspace(-5,log10(1.2e3),1000), array([DC_fcn(z) for z in logspace(-5,log10(1.2e3),1000)]))
DC = DC_fcn
z_ls = 1100 #last scattering

#######################################
##### find my own curve for dn/dz
#######################################
###### Hand+2014: a, b, c, A = 0.531, 7.810, 0.517, 0.688
#dndz_Hand = lambda z: 0.688*(z**0.531+z**(0.531*7.810))/(z**7.810+0.517)
########### van Waerbeke fitting
#dndz_VW = lambda z: 1.5*exp(-(z-0.7)**2/0.32**2)+0.2*exp(-(z-1.2)**2/0.46**2)
########### my dndz interpolation 

z_arr, W_wl0 = load(cmb_dir+'W_wl_interp.npy')#W_wl_interp_Hand.npy
z_arr = concatenate([z_arr, logspace(log10(5), log10(1200), 100)])
W_wl0 = concatenate([W_wl0, ones(100)*1e-128])
W_cmb_fcn = lambda z: 1.5*OmegaM*H0**2*(1+z)*H_inv(z)*DC(z)/c*(1-DC(z)/DC(z_ls))
###W_cmb = interpolate.interp1d(z_arr, array([W_cmb_fcn(z) for z in z_arr]))
W_cmb = W_cmb_fcn
W_wl = interpolate.interp1d(z_arr, W_wl0)

Ptable = genfromtxt(cmb_dir+'P_delta_smith03_revised')[::5,]#_colinparams
aa = array([1/1.05**i for i in arange(33)]) # scale factor
zz = 1.0/aa-1 # redshifts
kk = Ptable.T[0]
iZ, iK = meshgrid(zz,kk)
Z, K = iZ.flatten(), iK.flatten()
P_deltas = Ptable[:,1:].flatten()

Pmatter_interp = interpolate.CloughTocher2DInterpolator(array([K, Z]).T, P_deltas/(K/2.0/pi)**3)
Pmatter = lambda k, z: Pmatter_interp (k/h, z)*h**3

Ckk_integrand = lambda z, ell: 1.0/(H_inv(z)*c*DC(z)**2)*W_wl(z)*W_cmb(z)*Pmatter(ell/DC(z), z)
ell_arr = linspace(1e-5, 2000, 100)

#Ckk_arr = array([quad(Ckk_integrand, 0, 2.5 , args=(iell))[0] for iell in ell_arr])
#save(cmb_dir+'model_planck2015.npy',array([ell_arr, Ckk_arr]))


##################################
############ SNR #################
##################################
data_ell_arr = array([  380.06143435, 740.11963531, 1100.17783627,  1460.23603723, 1820.29423819])
ell_arr, Ckk_arr = load(cmb_dir+'model_planck2015.npy')
model_raw = interpolate.interp1d(ell_arr, Ckk_arr)(data_ell_arr)
MM = model_raw*data_ell_arr
model_fit = lambda A:  A*concatenate([MM,MM,MM,MM])
chisq_model_fcn = lambda A, CC, err: sum((CC-model_fit(A))**2/err**2)

def plot_elems (Wx, A=0.61741426, return_chisq_null = False):
	CC_noise = load(cmb_dir+'CFHTxPlanck_lensing_500sim_W%s.npy'%(Wx))/(data_ell_arr+1)*2*pi
	CC_signal =load(cmb_dir+'CFHTxPlanck_lensing_W%s.npy'%(Wx))/(data_ell_arr+1)*2*pi
	CC_err = std(CC_noise,axis=0)
	CC_noise_mean = mean(CC_noise,axis=0)
	CCN_cov = np.cov(CC_noise,rowvar=0)
	chisq_null = sum(mat(CC_signal)*mat(CCN_cov).I*mat(CC_signal).T)
	if return_chisq_null:
		print chisq_null
		return chisq_null
	else:
		return CC_signal, CC_err, CC_noise_mean

datacube = array([plot_elems(Wx) for Wx in range(1,5)])
CC_arr = datacube[:,0,:]
errK_arr = datacube[:,1,:]
#weightK = 1/errK_arr**2/sum(1/errK_arr**2, axis=0)
#CC_mean = sum(CC_arr*weightK,axis=0)
#err_mean = sqrt(1.0/sum(1/errK_arr**2, axis=0))

#A_out = op.minimize(chisq_model_fcn, 1.0, args=(concatenate(CC_arr), concatenate(errK_arr)))
#A_min = A_out.x#0.61741426
#chisq_model = A_out.fun
#chisq_null=sum(array([plot_elems(Wx, return_chisq_null=1) for Wx in range(1,5)]))

##################################
### mode coupling ################
##################################

#############################################
###### cross CMB lensing with CFHT galn #####
#############################################
data_ell_arr = array([  380.06143435, 740.11963531, 1100.17783627,  1460.23603723, 1820.29423819])
cat_dir = '/Users/jia/weaklensing/CFHTLenS/catalogue/'
maskGen = lambda Wx: np.load(cmb_dir+'mask/W%i_mask.npy'%(Wx))
kmapGen = lambda Wx: load(cat_dir+'kmap_W%i_sigma10_noZcut.npy'%(Wx))
galnGen = lambda i: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_galn_1.3_lo_sigmaG10.fit'%(i))
cmblGen = lambda Wx: np.load(cmb_dir+'planck/dat_kmap_flipper2048_CFHTLS_W%i_map.npy'%(Wx))
fmask2_arr = [0.69589653344034097, 0.56642933419641017, 0.5884087465608, 0.43618153901026568]
sizes = (1330, 800, 1120, 950)
PPR512=8468.416479647716
PPA512=2.4633625
edgesGen = lambda Wx: linspace(5,50,6)*sizes[Wx-1]/1330.0#linspace(5,80,11)
colors=('r','b','m','g')
for Wx in range(1,5):
	imask=maskGen(Wx)
	kmap = kmapGen(Wx)*imask
	#kmap = kmap/mean(kmap)-1
	cmblmap = cmblGen(Wx)*imask
	
	## set mean to 0
	kmap -= mean(kmap)
	cmblmap -= mean(cmblmap)
	
	edges = edgesGen(Wx)
	CC_signal = WLanalysis.CrossCorrelate(kmap, cmblmap,edges=edges)[1]/fmask2_arr[Wx-1]
	#np.save(cmb_dir+'galnxPlanck_lensing_W%s.npy'%(Wx), CC_signal)
	#plot(data_ell_arr+Wx*10, CC_signal/data_ell_arr*2*pi*1e6,'o',label='W%i'%(Wx))
	cc=colors[Wx-1]
	errorbar(data_ell_arr+(Wx-1)*30, CC_signal/data_ell_arr*2*pi*1e6,errK_arr[Wx-1]*1e6,fmt='o',ecolor=cc,mfc=cc, mec=cc, label='W%i'%(Wx))
	errorbar(data_ell_arr+(Wx-1)*30+10, CC_arr[Wx-1]*1e6, errK_arr[Wx-1]*1e6, fmt='*',ecolor=cc,mfc=cc, mec=cc)#, label='W%i'%(Wx))
title('circle: no z cut; star: 0.2<z<1.3')
xlabel('ell')
ylabel('ell*C*1e6')
legend(loc=0,fontsize=12)
show()

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
########## prepare for interpolation ############

#z_hist = genfromtxt(cmb_dir+'nz_sumpdf.hist').T
#z0, z1 = z_hist[0,:-1]+0.025, z_hist[1,:-1]/0.05
#z1 /= sum(z1)*0.05

#z0, z1 = genfromtxt(cmb_dir+'dndz_CFHT.txt').T
#z0 = concatenate([[0,], z0, linspace(z0[-1]*1.2, 1200,100)])
#z1 = concatenate([[0,], z1, 1e-128*ones(100)])

#dndz_interp = interpolate.interp1d(z0, z1,kind='cubic')

#integrand = lambda zs, z: dndz_interp(zs)*(1-DC(z)/DC(zs))
######integrand = lambda zs, z: dndz_Hand(zs)*(1-DC(z)/DC(zs))
#W_wl = lambda z: 1.5*OmegaM*H0**2*(1+z)*H_inv(z)*DC(z)/c*quad(integrand, z, 6.0, args=(z,))[0]
#z_arr200 = linspace(1e-5, 4, 50)
#W_wl_arr = array(map(W_wl, z_arr200))
#save(cmb_dir+'W_wl_interp_colin.npy',array([z_arr200, W_wl_arr]))
####save(cmb_dir+'W_wl_interp_Hand.npy',array([z_arr200, W_wl_arr]))


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
#for i in range(len(zz)):
    #loglog(kk, Ptable[:,i+1]/(kk/2/pi)**3)
#xlabel('k [Mpc/h]')
#ylabel('P_delta / (k/2pi)^3')
#title('smith03_revises')
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
#####Pmatter_interp = interpolate.CloughTocher2DInterpolator(array([K, Z]).T, P_deltas/(K/2.0/pi)**3)
#####Pmatter = lambda k, z: Pmatter_interp (k, z)

#k_CH, P_CH = genfromtxt('/Users/jia/Desktop/cmblensing/wmap9baosn_max_likelihood_Colinz0_ext_1648_matterpower.dat').T

#P_JL = array([Pmatter(ik*h, 0.0) for ik in k_CH])
#loglog(k_CH, P_CH, label='linear (Colin)')
#loglog(k_CH, P_JL*h**3, label='smith03 (Jia)')
#legend(loc=0)
#show()