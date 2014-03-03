#!/vega/astro/users/jl3509/tarball/anacondaa/bin/python
# Jia Liu 2014/2/18
# Overview: this code creates mass maps from CFHT catalogues
################ steps #####################
#1) smoothing (on the fly?), take into account c2, m, w correction
#2) KS93, KSvw, aperture mass(this needs smaller smoothing)
#3) count peaks, MF, powerspectrum 

import WLanalysis
from emcee.utils import MPIPool
import os

########## define constants ############
ngal_arcmin = 5.0
zmax=1.3
zmin=0.2

ngal_cut = ngal_arcmin*(60**2*12)/512**2# = 0.82, cut = 5 / arcmin^2
PPR512=8468.416479647716#pixels per radians
rad2pix=lambda x: around(512/2.0-0.5 + x*PPR512).astype(int) #from radians to pixel location

full_dir = '/vega/astro/users/jl3509/CFHT_cat/full_subfields/'
KS_dir = '/vega/astro/users/jl3509/CFHT_cat/KS/'
sigmaG_arr = (0.5, 1, 1.8, 3.5, 5.3, 8.9)

	
########### functions #########
def fileGen(i):
	'''i range from (1, 2..13)
	'''
	Me1_fn = KS_dir+'CFHT_subfield%02d_Me1.fits'%(i)
	Me2_fn = KS_dir+'CFHT_subfield%02d_Me2.fits'%(i)
	Mw_fn = KS_dir+'CFHT_subfield%02d_Mw.fits'%(i)
	galn_fn = KS_dir+'CFHT_subfield%02d_galn.fits'%(i)
	
	print 'fileGen', i
	if WLanalysis.TestComplete((Me1_fn,Me2_fn,Mw_fn,galn_fn),rm = True):
		Me1 = WLanalysis.readFits(Me1_fn)
		Me2 = WLanalysis.readFits(Me2_fn)
		Mw =  WLanalysis.readFits(Mw_fn)
		galn =WLanalysis.readFits(galn_fn)
	else:
		ifile = np.genfromtxt(full_dir+'full_subfield'+str(i) ,usecols=[0, 1, 2, 3, 4, 9, 10, 11, 16, 17])
		# cols: y, x, z_peak, z_rnd1, z_rnd2, e1, e2, w, m, c2

		#redshift cut 0.2< z <1.3
		zs = ifile[:,[2,3,4]]
		idx = where((amax(zs,axis=1) <= zmax) & (amin(zs,axis=1) >= zmin))[0]
		
		y, x, z_peak, z_rnd1, z_rnd2, e1, e2, w, m, c2 = ifile[idx].T

		k = array([e1*w, (e2-c2)*w, (1+m)*w])
		Ms, galn = WLanalysis.coords2grid(x, y, k)
		Me1, Me2, Mw = Ms
		WLanalysis.writeFits(Me1,Me1_fn)
		WLanalysis.writeFits(Me2,Me2_fn)
		WLanalysis.writeFits(Mw,Mw_fn)
		WLanalysis.writeFits(galn,galn_fn)
	return Me1, Me2, Mw, galn

####### smooth and KS inversion #########

def KSmap (i):
	Me1, Me2, Mw, galn = fileGen(i)
	for sigmaG in sigmaG_arr:	
		print 'KSmap i, sigmaG', i, sigmaG
		KS_fn = KS_dir+'CFHT_KS_sigma%02d_subfield%02d.fits'%(sigmaG,i)
		mask_fn = KS_dir+'CFHT_mask_ngal%i_sigma%02d_subfield%02d.fits'%(ngal_arcmin,sigmaG,i)
		
		if WLanalysis.TestComplete((KS_fn,mask_fn),rm=True):
			kmap = WLanalysis.readFits(KS_fn)
			Mmask = WLanalysis.readFits(mask_fn)
		else:
			Me1_smooth = WLanalysis.weighted_smooth(Me1, Mw, PPA=PPA512, sigmaG=sigmaG)
			Me2_smooth = WLanalysis.weighted_smooth(Me2, Mw, PPA=PPA512, sigmaG=sigmaG)
			galn_smooth = WLanalysis.weighted_smooth(galn, Mw, PPA=PPA512, sigmaG=sigmaG)
			## KS
			kmap = WLanalysis.KSvw(Me1_smooth, Me2_smooth)
			## mask
			maskidx = where(galn_smooth < ngal_cut) #cut at ngal=5
			Mmask = ones(shape=galn.shape)
			Mmask[maskidx]=0
			
			WLanalysis.writeFits(kmap, KS_fn)
			WLanalysis.writeFits(Mmask, KS_dir)
			
KSmap (1)
savetxt(KS_dir+'done.ls',zeros(5))
# Initialize the MPI pool
#pool = MPIPool()

# Make sure the thread we're running on is the master
#if not pool.is_master():
    #pool.wait()
    #sys.exit(0)
#logger.debug("Running with MPI...")

#pool.map()