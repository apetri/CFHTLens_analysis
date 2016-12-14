### Jia Liu 2016/12/13
### works on Stampede
### compute the bispectrum in the squeezed limit kappa^2 x kappa
### Gaussian smoothing 1,2,5 arcmin
#### on stampede
#### idev -m 60
#### ibrun python squeezed_bispec.py

import WLanalysis
import glob, os, sys
import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack

from scipy import interpolate
import random

machine ='stampede'# 'local'#
if machine == 'stampede':
    from emcee.utils import MPIPool
    main_dir = '/work/02977/jialiu/squeeze/'
    file_dir = '/work/02977/jialiu/CMBL_maps_46cosmo/'
    all_points = genfromtxt(file_dir+'model_point.txt')
    cosmo_arr = array(['Om%.3f_Ol%.3f_w-1.000_si%.3f'%(cosmo[0],1-cosmo[0], cosmo[1]) for cosmo in all_points])  
    kmapGen = lambda cosmo, r: WLanalysis.readFits(file_dir+'%s/WLconv_z1100.00_%04dr.fits'%(cosmo, r))

else:
    main_dir = '/Users/jia/Documents/weaklensing/sqeeze/'
    import pickle
    from pylab import *

fidu_cosmo = 'Om0.296_Ol0.704_w-1.000_si0.786'

PPA=2048.0/3.5/60.0

def BispecGen (r, cosmo = fidu_cosmo, R_arr = [1.0, 2.0, 5.0]):
    '''Generate the squeezed bispectrum, with smoothing scales R_arr
    '''
    print cosmo, r
    ikmap = kmapGen(cosmo, r)
    ikmap -= mean(ikmap) ## set mean to 0
    bs_arr = zeros(shape=(3,50))
    
    for i in range(len(R_arr)):
        ikmap_smooth = WLanalysis.smooth(ikmap,  R_arr[i]*PPA)
        bs_arr[i] = WLanalysis.CrossCorrelate(ikmap_smooth**2, ikmap_smooth, sizedeg = 12.25, PPA=PPA)[1]
    return bs_arr

if machine == 'stampede':
    pool=MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    a=pool.map(BispecGen, arange(1,1025))
    save(main_dir+'%s_BS.npy'%(fidu_cosmo), a)
    pool.close()
    print '---DONE---DONE---'

else:
    ell_arr2048=WLanalysis.PowerSpectrum(zeros(shape=(2048,2048)), bins=50)[0]
    BS_sims = load(main_dir+'Om0.296_Ol0.704_w-1.000_si0.786_BS.npy')
    BS_sims_R1_mean,  BS_sims_R2_mean, BS_sims_R5_mean= mean(BS_sims,axis=0) *2*pi/ell_arr2048**2
    BS_sims_R1_std,  BS_sims_R2_std, BS_sims_R5_std= std(BS_sims,axis=0)*2*pi/ell_arr2048**2

    ell_theo1, BS_R1_theory = pickle.load(open(main_dir+'reintegratedbispectrum/kappakappa_sq_linlog_full_new_theta_1e-2_nlPS_Bfit_Jias_Simulation_kmin7.560000e-03_kmax3.528000e 01_Filter_1_R1arcmin.pkl'))[1:]
    BS_R2_theory = pickle.load(open(main_dir+'reintegratedbispectrum/kappakappa_sq_linlog_full_new_theta_1e-2_nlPS_Bfit_Jias_Simulation_kmin7.560000e-03_kmax3.528000e 01_Filter_1_R2arcmin.pkl'))[2]
    BS_R5_theory = pickle.load(open(main_dir+'reintegratedbispectrum/kappakappa_sq_linlog_full_new_theta_1e-2_nlPS_Bfit_Jias_Simulation_kmin7.560000e-03_kmax3.528000e 01_Filter_1_R5arcmin.pkl'))[2]
    BS_R1_PBtheory = pickle.load(open(main_dir+'reintegratedbispectrum/kappakappa_sq_linlog_full_new_theta_1e-2_nlPS_Bfit_Jias_Simulation_kmin7.560000e-03_kmax3.528000e 01_Filter_1_postBorn_R1arcmin.pkl'))[2]
    BS_R2_PBtheory = pickle.load(open(main_dir+'reintegratedbispectrum/kappakappa_sq_linlog_full_new_theta_1e-2_nlPS_Bfit_Jias_Simulation_kmin7.560000e-03_kmax3.528000e 01_Filter_1_postBorn_R2arcmin.pkl'))[2]
    BS_R5_PBtheory = pickle.load(open(main_dir+'reintegratedbispectrum/kappakappa_sq_linlog_full_new_theta_1e-2_nlPS_Bfit_Jias_Simulation_kmin7.560000e-03_kmax3.528000e 01_Filter_1_postBorn_R5arcmin.pkl'))[2]
    
    f=figure(figsize=(8,6))
    ax=f.add_subplot(111)
    ax.errorbar(ell_arr2048, BS_sims_R1_mean, BS_sims_R1_std, color='b', label = 'sim R1')
    ax.errorbar(ell_arr2048, BS_sims_R2_mean, BS_sims_R2_std, color='g', label = 'sim R2')
    ax.errorbar(ell_arr2048, BS_sims_R5_mean, BS_sims_R5_std, color='r', label = 'sim R5')
    
    ax.plot(ell_theo1, BS_R1_theory, 'b--', label='theory R1')
    ax.plot(ell_theo1, BS_R2_theory, 'g--', label='theory R2')
    ax.plot(ell_theo1, BS_R5_theory, 'r--', label='theory R5')
    
    ax.plot(ell_theo1, BS_R1_PBtheory, 'b:', label='theory R1 (PostBorn)')
    ax.plot(ell_theo1, BS_R2_PBtheory, 'g:', label='theory R2 (PostBorn)')
    ax.plot(ell_theo1, BS_R5_PBtheory, 'r:', label='theory R5 (PostBorn)')
    
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$C_\ell ^{\kappa^2 \times \kappa}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e2, 1e4)
    savefig(main_dir+'compare_theory_sim.png')
    close()
    
