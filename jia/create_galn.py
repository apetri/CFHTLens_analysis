import numpy as np
from scipy import *
import WLanalysis

sigmaG = 0.5
emucat_dir = '/direct/astro+astronfs03/workarea/jia/CFHT/CFHT/raytrace_subfields/emulator_galpos_zcut0213/'
KS_dir = '/direct/astro+astronfs03/workarea/jia/CFHT/KSsim/'
Mk_fn = lambda i, cosmo, R: KS_dir+'SIM_Mk/SIM_Mk_rz1_subfield%i_%s_%04dr.fit'%(i, cosmo, R)
galn_fn = lambda i: KS_dir+'galn_subfield%i.fit'%(i)
fidu='mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.800'
hi_w='mQ3-512b240_Om0.260_Ol0.740_w-0.800_ns0.960_si0.800'
hi_s='mQ3-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.850'
hi_m='mQ3-512b240_Om0.290_Ol0.710_w-1.000_ns0.960_si0.800'
cosmo_arr=(fidu,hi_m,hi_w,hi_s)

def galn_gen(i):
	print i
	y, x, z = WLanalysis.readFits(emucat_dir+'emulator_subfield%i_zcut0213.fit'%(i)).T
	Mz, galn = WLanalysis.coords2grid(x, y, array([z,]))
	WLanalysis.writeFits(galn, galn_fn)
	
# map(galn_gen,range(1,14))

# power spectrum for all of the rz1, 4 cosmo
def ps (R):
	Mk = WLanalysis.readFits(Mk_fn(i, cosmo, R))
	galn = WLanalysis.readFits(galn_fn(i))
	Mk_smooth = WLanalysis.weighted_smooth(Mk, galn, sigmaG)
	ps = WLanalysis.PowerSpectrum(Mk_smooth)
	return ps[1]
	
for i in range(1, 14):
	for cosmo in cosmo_arr:
		print i, cosmo
		pmat = np.arary(map(ps, arange(1,1001)))
		pmat_fn = KS_dir + 'powspec_Mk/SIM_powspec_sigma05_subfield%i_rz1_%s_1000R.fit'%(i,cosmo)
		WLanalysis.writeFits(pmat,pmat_fn)
print 'done'