import numpy as np
from scipy import *
import WLanalysis
from pylab import *
import os
import scipy.ndimage as snd
import matplotlib.pyplot as plt

test_45_vs_randrot = 1
test_noise_signal = 0

plot_dir = '/Users/jia/weaklensing/CFHTLenS/plot/'
test_dir = '/Users/jia/weaklensing/CFHTLenS/mass/'
PPA512 =2.4633625#pixels per arcmin, PPR/degrees(1)/60

if test_45_vs_randrot:
	i=1
	#1. get y, x, e1, e2 for all subfields
	#2. create KS maps, first rotate 45, then rand rotate
	#3. plot out histogram for pixels
	for i in arange(1,14):
		sG_arr = array([1, 3.5, 5.3, 8.9])
		for sG in sG_arr:
			KS_45_fn = test_dir+'KS_45_%i_%i.fit'%(i,sG)
			KS_rand_fn = test_dir+'KS_rand_%i_%i.fit'%(i,sG)
			print i, sG
			if os.path.isfile(KS_45_fn) and os.path.isfile(KS_rand_fn):
				KS_45=WLanalysis.readFits(KS_45_fn)
				KS_rand=WLanalysis.readFits(KS_rand_fn)
			
			else:
				print 'generating KS'
				y, x, e1, e2, w = WLanalysis.readFits(test_dir+'yxew_subfield%i_zcut0213.fit'%(i)).T
				e1_45, e2_45 = WLanalysis.rndrot(e1, e2, deg=45)
				e1_rand, e2_rand = WLanalysis.rndrot(e1, e2, iseed=0)
				mat_e1_45,mat_e2_45,mat_e1_rand,mat_e2_rand, Mw = WLanalysis.coords2grid(x, y, array([e1_45*w, e2_45*w, e1_rand*w, e2_rand*w, w]) )[0]

				mat_e1_45_smoothed  = WLanalysis.weighted_smooth(mat_e1_45  , Mw, sigmaG=sG)
				mat_e2_45_smoothed  = WLanalysis.weighted_smooth(mat_e2_45  , Mw, sigmaG=sG)
				mat_e1_rand_smoothed= WLanalysis.weighted_smooth(mat_e1_rand, Mw, sigmaG=sG)
				mat_e2_rand_smoothed= WLanalysis.weighted_smooth(mat_e2_rand, Mw, sigmaG=sG)
				KS_45 =WLanalysis.KSvw(mat_e1_45_smoothed,mat_e2_45_smoothed)
				KS_rand=WLanalysis.KSvw(mat_e1_rand_smoothed,mat_e2_rand_smoothed)
				
				
				WLanalysis.writeFits(KS_45,KS_45_fn)
				WLanalysis.writeFits(KS_rand,KS_rand_fn)
			
			kappa_45 = KS_45.flatten()
			std_45=std(kappa_45)
			#hist_45,binedges45 = histogram(kappa_45,bins=100,range=(-5*std_45,5*std_45))
			#savetxt(test_dir+'hist45_%i_%i.ls'%(i,sG),array([hist_45,binedges45[:-1]]).T)
			
			kappa_rand = KS_rand.flatten()
			#std_rand=std(kappa_rand)
			#hist_rand,binedgesrand = histogram(kappa_rand,bins=100,range=(-5*std_rand,5*std_rand))
			#savetxt(test_dir+'histrand_%i_%i.ls'%(i,sG),array([hist_rand,binedgesrand[:-1]]).T)
			
			peaks_45 = WLanalysis.peaks_list(KS_45)
			peaks_rand=WLanalysis.peaks_list(KS_rand)
			
			figure(figsize=(6,8))
			subplot(211)
			hist(kappa_45,bins=20,range=(-3*std_45,3*std_45),histtype='step',label='45 deg')
			hist(kappa_rand,bins=20,range=(-3*std_45,3*std_45),histtype='step',label='random')
			title('PDF(top)/peaks(bottom) #%i, %.1f\''%(i,sG),fontsize=20)
			#xlabel('kappa')
			legend()

			subplot(212)
			hist(peaks_45,bins=20,range=(-3*std_45,3*std_45),histtype='step',label='45 deg')
			hist(peaks_rand,bins=20,range=(-3*std_45,3*std_45),histtype='step',label='random')
			#title('peaks')
			xlabel('kappa',fontsize=20)
			#legend()
			
			savefig(plot_dir+'test_galnoise_subfield%i_smooth%.1f.jpg'%(i,sG))
			close()
			

if test_noise_signal:
	# plot out the actual convergence map, for with or without noise
	sG=1 # smoothing = 3 arcmin
	
	idx = WLanalysis.readFits(test_dir+'zcut_idx_subfield1.fit')
	y, x, e1, e2, w = WLanalysis.readFits(test_dir+'yxew_zcut0213_subfield1.fit').T
	print 'a'
	sim = WLanalysis.readFits(test_dir+'raytrace_subfield1_WL-only_mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800_4096xy_0001r.fit')[idx]
	print 'b'
	k, s1, s2 = sim.T[[0,1,2]]

	e1_45, e2_45 = WLanalysis.rndrot(e1, e2, deg=45)
	e1_45b = e1_45.copy()
	e2_45b = e2_45.copy()
	e1_45 += s1
	e2_45 += s2
	e1_45 *= w
	e2_45 *= w

	e1_rand, e2_rand = WLanalysis.rndrot(e1, e2, iseed=0)
	e1_randb = e1_rand.copy()
	e2_randb = e2_rand.copy()
	e1_rand += s1
	e2_rand += s2
	e1_rand *= w
	e2_rand *= w
	print 'c'
	Mk,galn=WLanalysis.coords2grid(x, y, array([e1_45,e2_45,e1_rand,e2_rand,e1_45b,e2_45b,e1_randb,e2_randb,k*w,w,s1*w,s2*w]))

	mat_e1_45,mat_e2_45,mat_e1_rand,mat_e2_rand,mat_e1_45b,mat_e2_45b,mat_e1_randb,mat_e2_randb,mat_k,Mw,Ms1,Ms2 = Mk

	print 'd'
	mat_e1_45_smoothed  = WLanalysis.weighted_smooth(mat_e1_45  , Mw, sigmaG=sG)
	mat_e2_45_smoothed  = WLanalysis.weighted_smooth(mat_e2_45  , Mw, sigmaG=sG)
	mat_e1_rand_smoothed= WLanalysis.weighted_smooth(mat_e1_rand, Mw, sigmaG=sG)
	mat_e2_rand_smoothed= WLanalysis.weighted_smooth(mat_e2_rand, Mw, sigmaG=sG)
	mat_e1_45b_smoothed  = WLanalysis.weighted_smooth(mat_e1_45b  , Mw, sigmaG=sG)
	mat_e2_45b_smoothed  = WLanalysis.weighted_smooth(mat_e2_45b  , Mw, sigmaG=sG)
	mat_e1_randb_smoothed= WLanalysis.weighted_smooth(mat_e1_randb, Mw, sigmaG=sG)
	mat_e2_randb_smoothed= WLanalysis.weighted_smooth(mat_e2_randb, Mw, sigmaG=sG)

	Ms1_smoothed= WLanalysis.weighted_smooth(Ms1, Mw, sigmaG=sG)
	Ms2_smoothed= WLanalysis.weighted_smooth(Ms2, Mw, sigmaG=sG)
	print 'e'

	mat_k_smoothed = WLanalysis.weighted_smooth(mat_k, Mw,sigmaG=sG)

	KS_s = WLanalysis.KSvw(Ms1_smoothed,Ms2_smoothed)
	KS_45 =WLanalysis.KSvw(mat_e1_45_smoothed,mat_e2_45_smoothed)
	KS_rand=WLanalysis.KSvw(mat_e1_rand_smoothed,mat_e2_rand_smoothed)

	KS_45_nosig = WLanalysis.KSvw(mat_e1_45b_smoothed,mat_e2_45b_smoothed)
	KS_rand_nosig = WLanalysis.KSvw(mat_e1_randb_smoothed,mat_e2_randb_smoothed)

	plot_dir = '/Users/jia/Documents/weaklensing/CFHTLenS/plot/'
	def plotimshow(img,ititle,vmin=None,vmax=None):         
	#if vmin == None and vmax == None:
		imgnonzero=img[nonzero(img)]
		if vmin == None:
			std0 = std(imgnonzero)
			x0 = median(imgnonzero)
			vmin = x0-3*std0
			vmax = x0+3*std0
		im=imshow(img,interpolation='nearest',origin='lower',aspect=1,vmin=vmin,vmax=vmax)#,extent=[-25,25,-25,25])
		title(ititle)
		colorbar()
		savefig(plot_dir+'%s.jpg'%(ititle))
		close()    
	titles = ('test_galrot_conv','test_galrot_45','test_galrot_rand','test_galrot_45_noshear','test_galrot_rand_noshear','test_galrot_ShearOnly')
	i = 0
	for img in (mat_k_smoothed,KS_45,KS_rand,KS_45_nosig,KS_rand_nosig,KS_s):
		plotimshow(img,titles[i]+str(sG)+'arcmin')
		i+=1
