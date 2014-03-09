#!/afs/rhic.bnl.gov/@sys/opt/astro/SL64/anaconda/bin
# yeti: /vega/astro/users/jl3509/tarball/anacondaa/bin/python
# Jia Liu 2014/3/7
# Overview: this code creates mass maps from simulation
################ steps #####################
#1) smoothing, use random galaxy direction, and w as wegith
#2) KSvw
#3) count peaks, MF, powerspectrum 

import WLanalysis
from emcee.utils import MPIPool
import os
import numpy as np
from scipy import *
import scipy.ndimage as snd

########## define constants ############
ngal_arcmin = 5.0
zmax=1.3
zmin=0.2

ngal_cut = ngal_arcmin*(60**2*12)/512**2# = 0.82, cut = 5 / arcmin^2
PPR512=8468.416479647716#pixels per radians
PPA512=2.4633625
rad2pix=lambda x: around(512/2.0-0.5 + x*PPR512).astype(int) #from radians to pixel location

full_dir = '/direct/astro+astronfs01/workarea/jia/CFHT/full_subfields/'
KS_dir = '/direct/astro+astronfs03/workarea/jia/CFHT/KSsim/'
sim_dir = '/direct/astro+astronfs01/workarea/jia/CFHT/galaxy_catalogue_128R/'
fidu='mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800'
hi_w='mQ3-512b240_Om0.260_Ol0.740_w-0.800_ns0.960_si0.800'
hi_s='mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.850'
hi_m='mQ3-512b240_Om0.290_Ol0.710_w-1.000_ns0.960_si0.800'

SIMfn= lambda i, cosmo, R: sim_dir+'%s/raytrace_subfield%i_WL-only_%s_4096xy_%04dr.fit'%(cosmo, i, cosmo, R)
KSfn = lambda i, cosmo, R, sigmaG, zg: KS_dir+'%s/SIM_KS_sigma%02d_subfield%i_%s_%s_%04dr.fit'%(cosmo, sigmaG*10, i, zg, cosmo,R)#i=subfield, cosmo, R=realization, sigmaG=smoothing, zg=zgroup=(pz, rz, rz2)

sigmaG_arr = (0.5, 1, 1.8, 3.5, 5.3, 8.9)

## generate random rotation while preserve galaxy size and shape info

def rndrot (e1, e2, iseed=None):
	'''rotate galaxy with ellipticity (e1, e2), by a random angle.
	'''
	if iseed:
		random.seed(iseed)
	ells = e1+1j*e2
	ells_new = -ells*exp(-4j*pi*rand(len(e1)))
	return real(ells_new), imag(ells_new)

#### test
# a=rand(10)+rand(10)*1j
# b=rndrot(a)
# abs(a)==abs(b) #return True

## create index file for z cut ###
def zcut_idx (i, zmin=zmin, zmax=zmax):
	'''return index for z cut
	'''
	fn = full_dir+'zcut_idx_subfield%i'%(i)
	
	if WLanalysis.TestComplete(fn):
		idx = genfromtxt(fn)
	
	else:
		zs = np.genfromtxt(full_dir+'full_subfield'+str(i) ,usecols=[2, 3, 4])
		# z_peak, z_rnd1, z_rnd2

		# redshift cut 0.2< z <1.3
		print 'zs', i
		idx = np.where((amax(zs,axis=1) <= zmax) & (amin(zs,axis=1) >= zmin))[0]
		savetxt(fn,idx)
	return idx

def eobs_fun (g1, g2, k, e1, e2):
	g = (g1+1j*g2)/(1-k)
	eint = e1+1j*e2
	eobs = (g+eint)/(1-g*eint)
	return real(eobs), imag(eobs)

### 3/9, put field 1 config here ########
i=1
idx = zcut_idx (i)
y, x, e1, e2, w, c2 = (np.genfromtxt(full_dir+'full_subfield'+str(i) ,usecols=[0, 1, 9, 10, 11, 17])[idx]).T
### end field 1 config ###

def fileGen(i, R, cosmo):
	'''
	Input:
	i: subfield range from (1, 2..13)
	R: realization range from (1..128)
	cosmo: one of the 4 cosmos (fidu, hi_m, hi_w, hi_s)
	Return:
	3 Me1 = e1*w # 3 are for 3 redshift groups
	3 Me2 = e2*w
	Mw = w
	'''
	### these are from simulation
	Ms1_pz_fn  = KS_dir+'%s/SIM_Ms1_pz_subfield%i_%s_%04dr.fit'%(cosmo, i, cosmo, R)
	Ms2_pz_fn  = KS_dir+'%s/SIM_Ms2_pz_subfield%i_%s_%04dr.fit'%(cosmo, i, cosmo, R)
	Ms1_rz1_fn = KS_dir+'%s/SIM_Ms1_rz1_subfield%i_%s_%04dr.fit'%(cosmo, i, cosmo, R)
	Ms2_rz1_fn = KS_dir+'%s/SIM_Ms2_rz1_subfield%i_%s_%04dr.fit'%(cosmo, i, cosmo, R)
	Ms1_rz2_fn = KS_dir+'%s/SIM_Ms1_rz2_subfield%i_%s_%04dr.fit'%(cosmo, i, cosmo, R)
	Ms2_rz2_fn = KS_dir+'%s/SIM_Ms2_rz2_subfield%i_%s_%04dr.fit'%(cosmo, i, cosmo, R)
	Mw_fn = KS_dir+'SIM_Mw_subfield%i.fit'%(i) # same for all R

	Marr = (Mw_fn, Ms1_pz_fn, Ms2_pz_fn, Ms1_rz1_fn, Ms2_rz1_fn, Ms1_rz2_fn, Ms2_rz2_fn)
	print 'fileGen', i, str(R)+'r', cosmo
	if WLanalysis.TestComplete(Marr, rm = False):
		Mw = WLanalysis.readFits(Mw_fn)
		Ms1_pz  = WLanalysis.readFits(Ms1_pz_fn )
		Ms2_pz  = WLanalysis.readFits(Ms2_pz_fn )
		Ms1_rz1 = WLanalysis.readFits(Ms1_rz1_fn)
		Ms2_rz1 = WLanalysis.readFits(Ms2_rz1_fn)
		Ms1_rz2 = WLanalysis.readFits(Ms1_rz2_fn)
		Ms2_rz2 = WLanalysis.readFits(Ms2_rz2_fn)
		createfiles = 0
	
	elif WLanalysis.TestComplete((Mw_fn,),rm = False):
		#Mw = WLanalysis.readFits(Mw_fn)
		WLanalysis.TestComplete(Marr[1:], rm = True)
		createfiles = 1 #flag to create Ms's
	else:
		createfiles = 2 #flag to create everything

	if createfiles:
		#idx = zcut_idx (i)#redshift cut
		#simfile = WLanalysis.readFits(SIMfn(i,cosmo,R))[idx, [0,1,2,4,5,6,8,9,10]]#simulation file at redshift cut
		#s1_pz, s2_pz, k_pz, s1_rz1, s2_rz1, k_rz1, s1_rz2, s2_rz2, k_rz2 = simfile.T
		
		s1_pz, s2_pz, k_pz, s1_rz1, s2_rz1, k_rz1, s1_rz2, s2_rz2, k_rz2 = (WLanalysis.readFits(SIMfn(i,cosmo,R))[idx].T)[[0,1,2,4,5,6,8,9,10]]
		
		eint1, eint2 = rndrot(e1, e2-c2, iseed=R)#random rotation
			
		## get reduced shear
		e1_pz, e2_pz = eobs_fun(s1_pz, s2_pz, k_pz, eint1, eint2)
		e1_rz1, e2_rz1 = eobs_fun(s1_rz1, s2_rz1, k_rz1, eint1, eint2)
		e1_rz2, e2_rz2 = eobs_fun(s1_rz2, s2_rz2, k_rz2, eint1, eint2)
			
		kk = array([k_rz1, e1_pz*w, e2_pz*w, e1_rz1*w, e2_rz1*w, e1_rz2*w, e2_rz2*w, w])
		print 'coords2grid'
		Mk, Ms1_pz, Ms2_pz, Ms1_rz1, Ms2_rz1, Ms1_rz2, Ms2_rz2, Mw = WLanalysis.coords2grid(x, y, kk)[0]
		if createfiles == 2:
			WLanalysis.writeFits(Mw, Mw_fn)
		#Marr = (Mw_fn, Ms1_pz_fn, Ms2_pz_fn, Ms1_rz1_fn, Ms2_rz1_fn, Ms1_rz2_fn, Ms2_rz2_fn)
		j = 1
		for iM in (Ms1_pz, Ms2_pz, Ms1_rz1, Ms2_rz1, Ms1_rz2, Ms2_rz2):
			WLanalysis.writeFits(iM, Marr[j])
			j+=1
		### add Mk just for comparison ###
		Mk_fn = KS_dir+'%s/SIM_Mk_rz1_subfield%i_%s_%04dr.fit'%(cosmo, i, cosmo, R)
		WLanalysis.writeFits(Mk, Mk_fn)
		
	return Ms1_pz, Ms2_pz, Ms1_rz1, Ms2_rz1, Ms1_rz2, Ms2_rz2, Mw

### test, pass, still need to check actual map 
# Ms1_pz, Ms2_pz, Ms1_rz1, Ms2_rz1, Ms1_rz2, Ms2_rz2, Mw = fileGen(1, 1, fidu)

####### smooth and KS inversion #########
	
def KSmap(iiRcosmo):
	'''Input:
	i: subfield range from (1, 2..13)
	R: realization range from (1..128)
	cosmo: one of the 4 cosmos (fidu, hi_m, hi_w, hi_s)
	Return:
	KS inverted map
	'''
	i, R, cosmo = iiRcosmo
	Ms1_pz, Ms2_pz, Ms1_rz1, Ms2_rz1, Ms1_rz2, Ms2_rz2, Mw = fileGen(i, R, cosmo)
	Ms_arr = ((Ms1_pz, Ms2_pz), (Ms1_rz1, Ms2_rz1), (Ms1_rz2, Ms2_rz2))
	zgs=('pz', 'rz1', 'rz2')
	for sigmaG in sigmaG_arr:
		for j in range(3):
			print 'KSmap i, R, sigmaG, cosmo', i, R, sigmaG, cosmo
			KS_fn = KSfn(i, cosmo, R, sigmaG, zgs[j])
			Me1, Me2 = Ms_arr[j]
			
			if os.path.isfile(KS_fn):
				kmap = WLanalysis.readFits(KS_fn)
			else:
				Me1_smooth = WLanalysis.weighted_smooth(Me1, Mw, PPA=PPA512, sigmaG=sigmaG)
				Me2_smooth = WLanalysis.weighted_smooth(Me2, Mw, PPA=PPA512, sigmaG=sigmaG)
				kmap = WLanalysis.KSvw(Me1_smooth, Me2_smooth)
				WLanalysis.writeFits(kmap, KS_fn)

# test KSmap(1, 1, fidu), pass

#for i in (1,2):
	#j=0

	#idx = zcut_idx (i)
	#y, x, e1, e2, w, c2 = (np.genfromtxt(full_dir+'full_subfield'+str(i) ,usecols=[0, 1, 9, 10, 11, 17])[idx]).T
	
	#print 'idx'
	#for cosmo in (fidu,hi_m,hi_w,hi_s):
		#print 'cosmo', cosmo
		#iRcosmo=[[1,1,''],]*128
		#for R in arange(1,129):
			#iRcosmo[j]=[i,R,cosmo]
				#j+=1

	### Initialize the MPI pool
		#pool = MPIPool()
		### Make sure the thread we're running on is the master
		#if not pool.is_master():
			#pool.wait()
			#sys.exit(0)
		### logger.debug("Running with MPI...")
		#pool.map(KSmap, iRcosmo)


	
print 'idx'
i=1
for cosmo in (fidu, hi_m, hi_w, hi_s):
	print 'cosmo', cosmo
	iRcosmo=[[1,1,''],]*128
	j=0
	for R in arange(1,129):
		iRcosmo[j]=[i,R,cosmo]
		j+=1

	## Initialize the MPI pool
	pool = MPIPool()
	## Make sure the thread we're running on is the master
	#if not pool.is_master():
		#pool.wait()
		#sys.exit(0)
	#### logger.debug("Running with MPI...")
	
	pool.map(KSmap, iRcosmo)
		
savetxt(KS_dir+'done.ls',zeros(5))