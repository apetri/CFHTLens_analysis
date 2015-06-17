### python code to compute non-Gaussianity of CMB lensing maps
### Jia Liu 2015/06/12
### works on Stampede

import WLanalysis
import glob, os, sys
import numpy as np
from scipy import *
from emcee.utils import MPIPool

CMBlensing_dir = '/work/02977/jialiu/CMBnonGaussian/'

#PDF_bins = linspace(-0.3, 0.5, 51)
ends = [0.7, 0.6, 0.5, 0.4]
PDFbin_arr = [linspace(-end, end, 101) for end in ends]

sigmaG_arr = (0.5, 1.0, 2.0, 5.0)

b300_dir = '/work/02918/apetri/kappaCMB/Om0.260_Ol0.740_Ob0.046_w-1.000_ns0.960_si0.800/1024b300/Maps/'
#Pixels on a side: 2048
#Pixel size: 2.98828125 arcsec
#Total angular size: 1.7 deg
#lmin=2.1e+02 ; lmax=3.1e+05

b600_dir =  '/work/02918/apetri/kappaCMB/Om0.260_Ol0.740_Ob0.046_w-1.000_ns0.960_si0.800/1024b600/Maps/'
#Pixels on a side: 2048
#Pixel size: 6.15234375 arcsec
#Total angular size: 3.5 deg
#lmin=1.0e+02 ; lmax=1.5e+05

ell600 = array([110.50448683,    126.93632224,    145.81154455,    167.49348136,
          192.39948651,    221.00897366,    253.87264448,    291.62308909,
          334.98696272,    384.79897302,    442.01794731,    507.74528896,
          583.24617818,    669.97392544,    769.59794604,    884.03589463,
         1015.49057792,   1166.49235637,   1339.94785088,   1539.19589208,
         1768.07178925,   2030.98115583,   2332.98471274,   2679.89570175,
         3078.39178417,   3536.14357851,   4061.96231167,   4665.96942547,
         5359.79140351,   6156.78356833,   7072.28715702,   8123.92462333,
         9331.93885094,  10719.58280701,  12313.56713667,  14144.57431404,
        16247.84924667,  18663.87770189,  21439.16561402,  24627.13427334,
        28289.14862808,  32495.69849334,  37327.75540378,  42878.33122805,
        49254.26854668,  56578.29725615,  64991.39698667,  74655.51080755,
        85756.6624561 ,  98508.53709335])

ell300 = array([227.50923759,     261.33948696,     300.20023877,
           344.83952045,     396.11658987,     455.01847518,
           522.67897393,     600.40047754,     689.67904089,
           792.23317975,     910.03695035,    1045.35794786,
          1200.80095508,    1379.35808178,    1584.4663595 ,
          1820.0739007 ,    2090.71589571,    2401.60191017,
          2758.71616357,    3168.932719  ,    3640.14780141,
          4181.43179142,    4803.20382034,    5517.43232714,
          6337.86543799,    7280.29560281,    8362.86358284,
          9606.40764068,   11034.86465428,   12675.73087598,
         14560.59120563,   16725.72716569,   19212.81528136,
         22069.72930855,   25351.46175197,   29121.18241125,
         33451.45433138,   38425.63056271,   44139.45861711,
         50702.92350393,   58242.36482251,   66902.90866275,
         76851.26112542,   88278.91723422,  101405.84700787,
        116484.72964502,  133805.8173255 ,  153702.52225084,
        176557.83446844,  202811.69401573])

def PDFGen(kmap, PDF_bins):
	all_kappa = kmap[~isnan(kmap)]
	PDF = histogram(all_kappa, bins=PDF_bins)[0]
	PDF_normed = PDF/float(len(all_kappa))
	return PDF_normed, mean(kmap), std(kmap)
	
def compute_PDF_ps (fnsizedeg):
	'''for a convergence map with filename fn, compute the PDF and the power spectrum. sizedeg = 3.5**2, or 1.7**2'''
	fn, sizedeg = fnsizedeg
	print fn, sizedeg
	kmap = WLanalysis.readFits(fn)
	PPA = 2048.0/(sqrt(sizedeg)*60.0) #pixels per arcmin
	PDF10 = [PDFGen(WLanalysis.smooth(kmap, PPA*sigmaG_arr[i]), PDFbin_arr[i]) for i in range(len(sigmaG_arr))]
	ell_arr, powspec = WLanalysis.PowerSpectrum(kmap, sizedeg = sizedeg)
	return PDF10, powspec

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

###### local laptop plotting ##########
import matplotlib.pyplot as plt
from pylab import *

ell_arr = [ell600, ell300]
plot_dir = '/Users/jia/Desktop/CMBnonGaussian/plot/'
i=0
gaussian = lambda x, mu, sig: np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))/sig/sqrt(2.0*pi)

f=figure(figsize=(8,6))
ax=f.add_subplot(111)
for res in ('600','300'):
	res_dir = '/Users/jia/Desktop/CMBnonGaussian/b%s/'%(res)
	ps = load(res_dir+'ps%s.npy'%(res))	
	ax.errorbar(ell_arr[i], mean(ps,axis=0),std(ps,axis=0), label='Gadget (box size = %s Mpc/h)'%(res))
	i+=1
	
ell_nicaea, P_kappa_smith = genfromtxt('/Users/jia/Documents/code/nicaea_2.5/Demo/P_kappa_smithrevised').T

ell_nicaea, P_kappa_linear = genfromtxt('/Users/jia/Documents/code/nicaea_2.5/Demo/P_kappa_linear').T

ax.plot(ell_nicaea, P_kappa_smith, label='Nicaea2.5 (smith03)')
ax.plot(ell_nicaea, P_kappa_linear, label='Nicaea2.5 (linear)')
ax.set_xlim(ell_arr[0][0],ell_arr[1][-1])
ax.set_ylim(1e-4, 1e-2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$')
leg=ax.legend(loc=0)
leg.get_frame().set_visible(False)
savefig(plot_dir+'ps_nicaea.jpg')
close()

i=0
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
	