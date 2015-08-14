### python code to compute non-Gaussianity of CMB lensing maps
### Jia Liu 2015/06/12
### works on Stampede
### 2015/06/30: update, write function that takes one directory, 
### for each file inside that dir, do:
### (1) compute ps
### (2) generate a GRF from that ps - maybe generate from avg ps instead
### (3) compute PDF for file and GRF, for 5 smoothings
### (4) compute peaks for file and GRF, for 5 smoothings

import WLanalysis
import glob, os, sys
import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack
from emcee.utils import MPIPool

from scipy import interpolate
import random

############ for 31 cosmos #########
iii = int(sys.argv[1])
cosmo_arr = genfromtxt('/work/02977/jialiu/CMB_batch/success.txt',dtype='string')
cosmo = cosmo_arr[iii]
kmapGen = lambda r: WLanalysis.readFits('/home1/02977/jialiu/scratch/CMB_batch_storage/%s/1024b600/Maps/WLconv_z40.00_%04dr.fits'%(cosmo, r))
####################################

########### fiducial cosmos #########
#cosmo='Om0.260_Ol0.740_Ob0.046_w-1.000_ns0.960_si0.800'
#kmapGen = lambda r: WLanalysis.readFits(b600_dir+'WLconv_z38.00_%04dr.fits'%(r))
#####################################

CMBlensing_dir = '/work/02977/jialiu/CMBnonGaussian/'

ends = [0.5, 0.22, 0.18, 0.1, 0.08]
PDFbin_arr = [linspace(-end, end, 101) for end in ends]
kmap_stds = [0.06, 0.05, 0.04, 0.03, 0.02] #[0.014, 0.011, 0.009, 0.006, 0.005]
peak_bins_arr = [linspace(-3*istd, 6*istd, 26) for istd in kmap_stds]

sizedeg = 3.5**2
PPA = 2048.0/(sqrt(sizedeg)*60.0) #pixels per arcmin
sigmaG_arr = array([0.5, 1.0, 2.0, 5.0, 8.0])
sigmaP_arr = sigmaG_arr*PPA #smoothing scale in pixels

b600_dir =  '/work/02918/apetri/kappaCMB/Om0.260_Ol0.740_Ob0.046_w-1.000_ns0.960_si0.800/1024b600/Maps/'
#Pixels on a side: 2048
#Pixel size: 6.15234375 arcsec
#Total angular size: 3.5 deg
#lmin=1.0e+02 ; lmax=1.5e+05


def PDFGen (kmap, PDF_bins):
	all_kappa = kmap[~isnan(kmap)]
	PDF = histogram(all_kappa, bins=PDF_bins)[0]
	PDF_normed = PDF/float(len(all_kappa))
	return PDF_normed

def peaksGen (kmap, peak_bins):
	peaks = WLanalysis.peaks_list(kmap)
	peaks_hist = histogram(peaks, bins=peak_bins)[0]
	return peaks_hist

def compute_GRF_PDF_ps_pk (r):
	'''for a convergence map with filename fn, compute the PDF and the power spectrum. sizedeg = 3.5**2, or 1.7**2'''
	print cosmo, r
	kmap = kmapGen(r)
	#kmap = load(CMBlensing_dir+'GRF_fidu/'+'GRF_fidu_%04dr.npy'%(r))
	
	kmap_smoothed = [WLanalysis.smooth(kmap, sigmaP) for sigmaP in sigmaP_arr]
	i_arr = arange(len(sigmaP_arr))
	ps = WLanalysis.PowerSpectrum(kmap_smoothed[0])[1]
	PDF = [PDFGen(kmap_smoothed[i], PDFbin_arr[i]) for i in i_arr]
	peaks = [peaksGen(kmap_smoothed[i], peak_bins_arr[i]) for i in i_arr]

	###### generate GRF
	#random.seed(r)
	#GRF = WLanalysis.GRF_Gen(kmap)
	#save(CMBlensing_dir+'GRF_fidu/'+'GRF_fidu_%04dr.npy'%(r), GRF)		
	#GRF = load(CMBlensing_dir+'GRF_fidu/'+'GRF_fidu_%04dr.npy'%(r))
	#GRF_smoothed = [WLanalysis.smooth(GRF, sigmaP) for sigmaP in sigmaP_arr]
	#PDF_GRF = [PDFGen(GRF_smoothed[i], PDFbin_arr[i]) for i in i_arr]
	#peaks_GRF = [peaksGen(GRF_smoothed[i], peak_bins_arr[i]) for i in i_arr]
	#############

	return ps, PDF, peaks#, PDF_GRF, peaks_GRF
		
pool=MPIPool()
a=pool.map(compute_GRF_PDF_ps_pk,range(1, 1025))
save(CMBlensing_dir+'%s_PDF_pk_600b_GRF'%(cosmo), a)
print '---DONE---DONE---'


########### cosmology constraints #############
# first get average of everything
# covariance matrix
#cosmo_arr = genfromtxt('/work/02977/jialiu/CMB_batch/success.txt',dtype='string')
#pspkPDF_avg = 
#pspkPDF_fidu_mat = 

#def return_interp_cosmo_for_idx (idx):
	#ps_CFHT_test = ps_CFHT0[idx]
	#idx = idx[where(ps_CFHT_test>0)[0]]#rid of zero bins	
	#ps_CFHT = ps_CFHT0[idx]
	#ps_fidu_mat = ps_fidu_mat0[:,idx]
	#ps_avg = ps_avg0[:,idx]
	#cov_mat = cov(ps_fidu_mat,rowvar=0)
	#cov_inv = mat(cov_mat).I

	#spline_interps = list()
	#for ibin in range(ps_avg.shape[-1]):
		#ps_model = ps_avg[:,ibin]
		#iinterp = interpolate.Rbf(im, iw, s, ps_model)
		#spline_interps.append(iinterp)

	#def interp_cosmo (params, method = 'multiquadric'):
		#'''Interpolate the powspec for certain param.
		#Params: list of 3 parameters = (om, w, si8)
		#Method: "multiquadric" for spline (default), and "GP" for Gaussian process.
		#'''
		#mm, wm, sm = params
		#gen_ps = lambda ibin: spline_interps[ibin](mm, wm, sm)
		#ps_interp = array(map(gen_ps, range(ps_avg.shape[-1])))
		#ps_interp = ps_interp.reshape(-1,1).squeeze()
		#return ps_interp
	#return interp_cosmo, cov_mat, cov_inv, ps_CFHT








############ plot on local laptop ##############

#cd ~/Desktop/CMBnonGaussian/
#mat_kappa=load('PDF_pk_600b_kappa.npy')
#mat_GRF=load('PDF_pk_600b_GRF.npy')

#f=figure(figsize=(15,25))
#for i in arange(len(sigmaG_arr)):
	#ax=f.add_subplot(5,2,i*2+1)
	#ax2=f.add_subplot(5,2,i*2+2)
	#iPDF_kappa = array([mat_kappa[x][0][i] for x in range(1024)])
	#ipeak_kappa = array([mat_kappa[x][1][i] for x in range(1024)])
	#iPDF_GRF = array([mat_GRF[x][0][i] for x in range(1024)])
	#ipeak_GRF = array([mat_GRF[x][1][i] for x in range(1024)])
	
	#ax.errorbar(PDFbin_arr[i][1:], mean(iPDF_kappa, axis=0), std(iPDF_kappa, axis=0)/sqrt(3e4/12), label='kappa',)
	#ax.errorbar(PDFbin_arr[i][1:], mean(iPDF_GRF, axis=0), std(iPDF_GRF, axis=0)/sqrt(3e4/12), label='GRF')
	#ax2.errorbar(peak_bins_arr[i][1:], mean(ipeak_kappa, axis=0), std(ipeak_kappa, axis=0)/sqrt(3e4/12), label='kappa')
	#ax2.errorbar(peak_bins_arr[i][1:], mean(ipeak_GRF, axis=0), std(ipeak_GRF, axis=0)/sqrt(3e4/12), label='GRF')
	#ax.set_yscale('log')
	#ax2.set_yscale('log')
	#meanPDF = mean(iPDF_kappa, axis=0)
	#meanPK = mean(ipeak_kappa, axis=0)
	#ax.set_ylim(amax([amin(meanPDF), 1e-7]), amax(meanPDF)*1.5)
	#ax2.set_ylim(amin(meanPK), amax(meanPK)*1.5)
	#ax.set_title('PDF(%.1farcmin)'%(sigmaG_arr[i]))
	#ax2.set_title('peaks(%.1farcmin)'%(sigmaG_arr[i]))
	#if i ==0:
		#leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=0)
		#leg.get_frame().set_visible(False)
	#if i == 4:
		#ax.set_xlabel('kappa')
		#ax2.set_xlabel('kappa')
	
#savefig('/Users/jia/Desktop/CMBnonGaussian/PK_PDF.jpg')
#close()

#mat_kappa=load('PDF_pk_600b_kappa.npy')
#mat_GRF=load('PDF_pk_600b_GRF.npy')


#for i in arange(len(sigmaG_arr)):
	#ax=f.add_subplot(5,2,i*2+1)
	#ax2=f.add_subplot(5,2,i*2+2)
	#iPDF_kappa = array([mat_kappa[x][0][i] for x in range(1024)])
	#ipeak_kappa = array([mat_kappa[x][1][i] for x in range(1024)])
	#iPDF_GRF = array([mat_GRF[x][0][i] for x in range(1024)])
	#ipeak_GRF = array([mat_GRF[x][1][i] for x in range(1024)])

	#mean_pk_GRF = mean(ipeak_GRF,axis=0)
	#mean_pk_kappa = mean(ipeak_kappa,axis=0)
	##idx_pk = nonzero(mean_pk_GRF)[0]
	#idx_pk = nonzero(mean_pk_kappa)[0]
	
	#mean_PDF_GRF = mean(iPDF_GRF,axis=0)
	#mean_PDF_kappa = mean(iPDF_kappa,axis=0)
	##idx_PDF = where(mean_PDF_GRF>5e-4)[0]
	#idx_PDF = where(mean_PDF_kappa>5e-4)[0]
	
	##covI_PDF = mat(cov(iPDF_GRF[:,idx_PDF]/sqrt(3e4/12),rowvar=0)).I
	##covI_peak = mat(cov(ipeak_GRF[:,idx_pk]/sqrt(3e4/12),rowvar=0)).I
	
	#covI_PDF = mat(cov(iPDF_kappa[:,idx_PDF]/sqrt(3e4/12),rowvar=0)).I
	#covI_peak = mat(cov(ipeak_kappa[:,idx_pk]/sqrt(3e4/12),rowvar=0)).I

	#dN_PDF = mat((mean_PDF_GRF-mean_PDF_kappa)[idx_PDF])
	#dN_pk =  mat((mean_pk_GRF-mean_pk_kappa)[idx_pk])
	#chisq_PDF = dN_PDF*covI_PDF*dN_PDF.T
	#chisq_peak = dN_pk*covI_peak*dN_pk.T
	#print 'sigmaG = %.1f arcmin, SNR(PDF) = %.2f, SNR(peaks) = %.2f' % (sigmaG_arr[i], sqrt(chisq_PDF), sqrt(chisq_peak))
############# test plots ######################

##a=WLanalysis.readFits('/Users/jia/Documents/weaklensing/map_conv_shear_sample/WL-conv_m-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.798_4096xy_0001r_0029p_0100z_og.gre.fit')

##b=GRF_Gen(a)

##ell, ps_a = WLanalysis.PowerSpectrum(a)
##ell, ps_b = WLanalysis.PowerSpectrum(b)

##from pylab import *
##print 'hi'
##f=figure()
##ax=f.add_subplot(111)
##ax.plot(ell, ps_a,'b-',lw=2,label='Convergence Map')
##ax.plot(ell, ps_b,'r--',lw=2,label='Gaussian Random Field')
##ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$',fontsize=20)
##ax.set_xscale('log')
##ax.set_yscale('log')
##ax.set_xlabel(r'$\ell$',fontsize=20)
##ax.set_xlim(ell[6],ell[-5])
##ax.set_ylim(3e-6,1e-3)
##leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=0)
##leg.get_frame().set_visible(False)
##show()

##asmooth=WLanalysis.smooth(a, 2.5*4)
##bsmooth=WLanalysis.smooth(b, 2.5*4)
#f=figure()
#xxx=0.01
#subplot(121)
#imshow(asmooth, vmin=-2*xxx, vmax=8*xxx)
##title('Convergence Map',fontsize=20)
#colorbar()
#subplot(122)
#imshow(bsmooth, vmin=-2*xxx, vmax=8*xxx)
##title('Gaussian Random Field',fontsize=20)
#colorbar()
#show()

######################################################
############### J U N K ##############################
######################################################

#img = a.astype(float)
#e1, ps_kappa = WLanalysis.PowerSpectrum(a)
#GRF = GRF_Gen (ell_arr_center, psd1D0, size)
#e1, ps_GRF = WLanalysis.PowerSpectrum(GRF)

#loglog(e1, ps_kappa, label='kappa')
##loglog(e1, ps_GRF, label='GRF')
##xxxx=array([WLanalysis.PowerSpectrum(GRF_Gen (ell_arr_center, psd1D0, size))[1] for i in range(5)])
#loglog(e1,mean(xxxx,axis=0), label='GRF')
#legend()
#xlabel('ell')
#ylabel('ell**2*P')
#show()

#subplot(121)
#imshow(a,vmin=-.06,vmax=.06)
#title('kappa')
#colorbar()
#subplot(122)
#imshow(xxx,vmin=-.06,vmax=.06)
#title('GRF')
#colorbar()
#show()

#b300_dir = '/work/02918/apetri/kappaCMB/Om0.260_Ol0.740_Ob0.046_w-1.000_ns0.960_si0.800/1024b300/Maps/'
##Pixels on a side: 2048
##Pixel size: 2.98828125 arcsec
##Total angular size: 1.7 deg
##lmin=2.1e+02 ; lmax=3.1e+05


#ell600 = array([110.50448683,    126.93632224,    145.81154455,    167.49348136,
          #192.39948651,    221.00897366,    253.87264448,    291.62308909,
          #334.98696272,    384.79897302,    442.01794731,    507.74528896,
          #583.24617818,    669.97392544,    769.59794604,    884.03589463,
         #1015.49057792,   1166.49235637,   1339.94785088,   1539.19589208,
         #1768.07178925,   2030.98115583,   2332.98471274,   2679.89570175,
         #3078.39178417,   3536.14357851,   4061.96231167,   4665.96942547,
         #5359.79140351,   6156.78356833,   7072.28715702,   8123.92462333,
         #9331.93885094,  10719.58280701,  12313.56713667,  14144.57431404,
        #16247.84924667,  18663.87770189,  21439.16561402,  24627.13427334,
        #28289.14862808,  32495.69849334,  37327.75540378,  42878.33122805,
        #49254.26854668,  56578.29725615,  64991.39698667,  74655.51080755,
        #85756.6624561 ,  98508.53709335])

#ell300 = array([227.50923759,     261.33948696,     300.20023877,
           #344.83952045,     396.11658987,     455.01847518,
           #522.67897393,     600.40047754,     689.67904089,
           #792.23317975,     910.03695035,    1045.35794786,
          #1200.80095508,    1379.35808178,    1584.4663595 ,
          #1820.0739007 ,    2090.71589571,    2401.60191017,
          #2758.71616357,    3168.932719  ,    3640.14780141,
          #4181.43179142,    4803.20382034,    5517.43232714,
          #6337.86543799,    7280.29560281,    8362.86358284,
          #9606.40764068,   11034.86465428,   12675.73087598,
         #14560.59120563,   16725.72716569,   19212.81528136,
         #22069.72930855,   25351.46175197,   29121.18241125,
         #33451.45433138,   38425.63056271,   44139.45861711,
         #50702.92350393,   58242.36482251,   66902.90866275,
         #76851.26112542,   88278.91723422,  101405.84700787,
        #116484.72964502,  133805.8173255 ,  153702.52225084,
        #176557.83446844,  202811.69401573])

########## operation on stampede 	
#pool = MPIPool()

#out600 = pool.map(compute_PDF_ps, [(fn, 3.5**2) for fn in glob.glob(b600_dir+'*.fits')])
#save(CMBlensing_dir+'out600.npy',out600)

#ps600 = array([out600[i][1] for i in range(len(out600))])
#save(CMBlensing_dir+'ps600.npy',ps600)
#for j in range(len(sigmaG_arr)):
	#PDF600 = array([out600[i][0][j][0] for i in range(len(out600))])
	#mean600 = array([out600[i][0][j][1] for i in range(len(out600))])
	#std600 = array([out600[i][0][j][2] for i in range(len(out600))])
	#save(CMBlensing_dir+'PDF600%02d.npy'%(sigmaG_arr[j]*10),PDF600)
	#save(CMBlensing_dir+'mean600%02d.npy'%(sigmaG_arr[j]*10),mean600)
	#save(CMBlensing_dir+'std600%02d.npy'%(sigmaG_arr[j]*10),std600)


#out300 = pool.map(compute_PDF_ps, [(fn, 1.7**2) for fn in glob.glob(b300_dir+'*.fits')])
#save(CMBlensing_dir+'out300.npy',out300)

#ps300 = array([out300[i][1] for i in range(len(out300))])
#save(CMBlensing_dir+'ps300.npy',ps300)
#for j in range(len(sigmaG_arr)):
	#PDF300 = array([out300[i][0][j][0] for i in range(len(out300))])
	#mean300 = array([out300[i][0][j][1] for i in range(len(out300))])
	#std300 = array([out300[i][0][j][2] for i in range(len(out300))])
	#save(CMBlensing_dir+'PDF300%02d.npy'%(sigmaG_arr[j]*10),PDF300)
	#save(CMBlensing_dir+'mean300%02d.npy'%(sigmaG_arr[j]*10),mean300)
	#save(CMBlensing_dir+'std300%02d.npy'%(sigmaG_arr[j]*10),std300)

#print 'DONE-DONE-DONE'

#######################################
###### local laptop plotting ##########
#######################################

#import matplotlib.pyplot as plt
#from pylab import *

#ell_arr = [ell600, ell300]
#plot_dir = '/Users/jia/Desktop/CMBnonGaussian/plot/'
#i=0
#gaussian = lambda x, mu, sig: np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))/sig/sqrt(2.0*pi)

#f=figure(figsize=(8,6))
#ax=f.add_subplot(111)
#for res in ('600','300'):
	#res_dir = '/Users/jia/Desktop/CMBnonGaussian/b%s/'%(res)
	#ps = load(res_dir+'ps%s.npy'%(res))	
	#ax.errorbar(ell_arr[i], mean(ps,axis=0),std(ps,axis=0), label='Gadget (box size = %s Mpc/h)'%(res))
	#i+=1
	
#ell_nicaea, P_kappa_smith = genfromtxt('/Users/jia/Documents/code/nicaea_2.5/Demo/P_kappa_smithrevised').T

#ell_nicaea, P_kappa_linear = genfromtxt('/Users/jia/Documents/code/nicaea_2.5/Demo/P_kappa_linear').T

#ax.plot(ell_nicaea, P_kappa_smith, label='Nicaea2.5 (smith03)')
#ax.plot(ell_nicaea, P_kappa_linear, label='Nicaea2.5 (linear)')
#ax.set_xlim(ell_arr[0][0],ell_arr[1][-1])
#ax.set_ylim(1e-4, 1e-2)
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_xlabel(r'$\ell$')
#ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$')
#leg=ax.legend(loc=0)
#leg.get_frame().set_visible(False)
#savefig(plot_dir+'ps_nicaea.jpg')
#close()

#i=0
#for res in ('600','300'):	
	##f=figure(figsize=(12,8))
	##for j in range(4):
		##ax=f.add_subplot(2,2,j+1)
		##sigmaG = sigmaG_arr[j]
		##iPDF = load(res_dir+'PDF%s%02d.npy'%(res,sigmaG_arr[j]*10))
		##imean =mean(load(res_dir+'mean%s%02d.npy'%(res,sigmaG_arr[j]*10)))
		##istd = mean(load(res_dir+'std%s%02d.npy'%(res,sigmaG_arr[j]*10)))
		##PDF_center = WLanalysis.edge2center(PDFbin_arr[j])
		##norm = 1.0/(PDF_center[-1]-PDF_center[-2])
		##ax.errorbar(PDF_center, mean(iPDF,axis=0)*norm, std(iPDF,axis=0)*norm/sqrt(3e4/12))
		
		##xbins = linspace(PDFbin_arr[j][0],PDFbin_arr[j][-1], 100)
		##ax.plot(xbins, gaussian(xbins, imean, istd))
		##ax.set_xlabel(r'$\kappa$')
		##ax.set_ylabel('PDF')
		##ax.annotate('$\sigma_G = %s$'%(sigmaG), xy=(0.05, 0.85),xycoords='axes fraction',color='k',fontsize=16)
		
		##ax.set_yscale('log')
		##ax.set_ylim(1e-4, 50)
		##ax.set_xlim(-0.2, 0.3)
		
	##savefig(plot_dir+'PDF_log_scaled_b%s.jpg'%(res))
	##close()
	##i+=1
	