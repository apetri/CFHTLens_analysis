#!python
# Jia Liu (10/15/2015)
# multiplicative bias project with Alvaro and Colin
# This code does:
# (1) calculate 2 cross-correlations x 3 cuts = 6
#     (galn x CFHT, galn x Planck CMB lensing);
# (2) calculate their theoretical error;
# (3) generate 100 GRF from galn maps, to get error estimation (600 in tot)
# (4) compute the model
# (5) calculate SNR
# (6) estimate m

import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack
from scipy.fftpack import fftfreq, fftshift
import os
import scipy.ndimage as snd
import WLanalysis
import sys

########## knobs #############
main_dir = '/Users/jia/weaklensing/multiplicative/'
#main_dir = '/work/02977/jialiu/multiplicative/'

prefix = '_nonzeroOffDiagCov'##''# or '' for zero off-diag cov
compute_data_points = 0
compute_sim_err = 0
compute_model = 0
plot_omori_comp = 0
compute_m_fit, bmodel = 1, 2#bmodels:0-constb, 1-b0(1+z), 2-b0(1+z)-z
## official plots ########
plot_cc = 0
plot_contour = 0
plot_kernel = 0
#################### constants and small functions ##################
sizes = (1330, 800, 1120, 950)

######## referee
#galnGen = lambda Wx, cut: load (main_dir+'cfht_galn/W%i_cut%i_ludoweight.npy'%(Wx, cut))
galnGen = lambda Wx, cut: load (main_dir+'referee/galn_W%i_cut%i_LensfitWNonzero.npy'%(Wx,cut))

CkappaGen = lambda Wx: WLanalysis.readFits (main_dir+'cfht_kappa/W%s_KS_1.3_lo_sigmaG10.fit'%(Wx))
PkappaGen = lambda Wx: load (main_dir+'planck2015_kappa/dat_kmap_flipper2048_CFHTLS_W%s_map.npy'%(Wx))
#CmaskGen = lambda Wx: load (main_dir+'cfht_mask/Mask_W%s_0.7_sigmaG10.npy'%(Wx))
CmaskGen = lambda Wx: load (main_dir+'mask_ludo/ludomask_weight0_manu_W%i.npy'%(Wx))
PmaskGen = lambda Wx: load (main_dir+'planck2015_mask/kappamask_flipper2048_CFHTLS_W%s_map.npy'%(Wx))
maskGen = lambda Wx: CmaskGen(Wx)*PmaskGen(Wx)
PlanckSim15Gen = lambda Wx, r: load('/work/02977/jialiu/cmblensing/planck/sim15/sim_%04d_kmap_CFHTLS_W%s.npy'%(r, Wx))



edgesGen = lambda Wx: linspace(1,60,7)*sizes[Wx-1]/1330.0
### omori & holder bin edges #####
#edgesGen = lambda Wx: linspace(1.25, 47.49232195,21)*sizes[Wx-1]/1330.0

edges_arr = map(edgesGen, range(1,5))
sizedeg_arr = array([(sizes[Wx-1]/512.0)**2*12.0 for Wx in range(1,5)])
####### test: ell_arr = WLanalysis.PowerSpectrum(CmaskGen(1), sizedeg = sizedeg_arr[0],edges=edges_arr[0])[0]
ell_arr = WLanalysis.edge2center(edgesGen(1))*360.0/sqrt(sizedeg_arr[0])

factor = (ell_arr+1)*ell_arr/(2.0*pi)
mask_arr = map(maskGen, range(1,5))
fmask_arr = array([sum(mask_arr[Wx-1])/sizes[Wx-1]**2 for Wx in range(1,5)])
fmask2_arr = array([sum(mask_arr[Wx-1]**2)/sizes[Wx-1]**2 for Wx in range(1,5)])
fsky_arr = fmask_arr*sizedeg_arr/41253.0
d_ell = ell_arr[1]-ell_arr[0]
#################################################

def theory_CC_err(map1, map2, Wx):
    map1*=mask_arr[Wx-1]
    map2*=mask_arr[Wx-1]
    #map1-=mean(map1)
    #map2-=mean(map2)
    auto1 = WLanalysis.PowerSpectrum(map1, sizedeg = sizedeg_arr[Wx-1], edges=edges_arr[Wx-1],sigmaG=1.0)[-1]/fmask2_arr[Wx-1]/factor
    auto2 = WLanalysis.PowerSpectrum(map2, sizedeg = sizedeg_arr[Wx-1], edges=edges_arr[Wx-1],sigmaG=1.0)[-1]/fmask2_arr[Wx-1]/factor    
    err = sqrt(auto1*auto2/fsky_arr[Wx-1]/(2*ell_arr+1)/d_ell)
    CC = WLanalysis.CrossCorrelate(map1, map2, edges = edges_arr[Wx-1], sigmaG1=1.0, sigmaG2=1.0)[1]/fmask2_arr[Wx-1]/factor
    return CC, err

def find_SNR (CC_arr, errK_arr):
    '''Find the mean of 4 fields, and the signal to noise ratio (SNR).
    Input:
    CC_arr: array of (4 x Nbin) in size, for cross correlations;
    errK_arr: error bar array, same dimension as CC_arr;
    Output:
    SNR = signal to noise ratio
    CC_mean = the mean power spectrum of the 4 fields, an Nbin array.
    err_mean = the mean error bar of the 4 fields, an Nbin array.
    '''
    weightK = 1/errK_arr**2/sum(1/errK_arr**2, axis=0)
    CC_mean = sum(CC_arr*weightK,axis=0)
    err_mean = sqrt(1.0/sum(1/errK_arr**2, axis=0))
    SNR = sqrt( sum(CC_mean**2/err_mean**2) )
    SNR2 = sqrt(sum (CC_arr**2/errK_arr**2))
    return SNR, SNR2, CC_mean, err_mean

if compute_data_points:
                                             
    def theory_CC_err(map1, map2, Wx, cut):
        mask=maskGen(Wx)
        #map1*=mask
        #map2*=mask
        fmask2=sum(mask)/sizes[Wx-1]**2
        fsky = fmask2*sizedeg_arr[Wx-1]/41253.0
        auto1 = WLanalysis.PowerSpectrum(map1, sizedeg = sizedeg_arr[Wx-1], edges=edges_arr[Wx-1],sigmaG=0.0)[-1]/fmask2/factor
        auto2 = WLanalysis.PowerSpectrum(map2, sizedeg = sizedeg_arr[Wx-1], edges=edges_arr[Wx-1],sigmaG=0.0)[-1]/fmask2/factor    
        err = sqrt(auto1*auto2/fsky/(2*ell_arr+1)/d_ell)
        CC0 = WLanalysis.CrossCorrelate(map1, map2, edges = edges_arr[Wx-1])[1]/fmask2/factor
        CC1 = WLanalysis.CrossCorrelate(map2, map1, edges = edges_arr[Wx-1])[1]/fmask2/factor
        CC=0.5*(CC0+CC1)
        return CC, err

    ######### this part is to test old work in Liu&Hill 2015 ############
    #cut=22
    #cmb_dir = '/Users/jia/Documents/weaklensing/cmblensing/'
    #CkappaGen_nocut = lambda Wx: np.load('/Users/jia/Documents/weaklensing/cmblensing/cfht/kmap_W%i_sigma10_noZcut.npy'%(Wx))
    
    #maskGen = lambda Wx: PmaskGen(Wx)*np.load(cmb_dir+'mask/W%i_mask1315_noZcut.npy'%(Wx))
    #CmaskGen = lambda Wx: np.load(cmb_dir+'mask/W%i_mask1315_noZcut.npy'%(Wx))
    
    #kappa_CC_err3 = array([theory_CC_err(CkappaGen_nocut(Wx)*CmaskGen(Wx), PkappaGen(Wx)*PmaskGen(Wx), Wx, cut) for Wx in range(1,5)])
    #save('/Users/jia/Desktop/cfhtplancklensing_CC_err_%s_diffmask.npy'%(cut), kappa_CC_err3)

    ###########################################
    for cut in (22, 23, 24): ######## 3 field cross power spectrum
        ## compute C_ell, only needed once
        planck_CC_err = array([theory_CC_err(PkappaGen(Wx), galnGen(Wx,cut), Wx, cut) for Wx in range(1,5)])
        cfht_CC_err = array([theory_CC_err(CkappaGen(Wx), galnGen(Wx,cut), Wx, cut) for Wx in range(1,5)])
        ####### referee
        #save(main_dir+'powspec/planck_CC_err_%s_ludo.npy'%(cut), planck_CC_err)
        #save(main_dir+'powspec/cfht_CC_err_%s_ludo.npy'%(cut), cfht_CC_err)
        save(main_dir+'referee/planck_CC_err_%s_ludo.npy'%(cut), planck_CC_err)
        save(main_dir+'referee/cfht_CC_err_%s_ludo.npy'%(cut), cfht_CC_err)
        
    
########### compute sim error ######################
if compute_sim_err:
    from scipy.fftpack import fftfreq, fftshift,ifftshift
    from random import gauss
    from emcee.utils import MPIPool
    p = MPIPool()
    #class GRF_Gen:
        #'''return a random gaussian field that has the same power spectrum as img.
        #'''
        #def __init__(self, kmap):
            #self.size = kmap.shape[0]
            #self.GRF = rand(self.size,self.size)
            #self.p2D_mean, self.p2D_std = ps1DGen(kmap)
        
        #def newGRF(self):
            #self.psd2D_GRF = gauss(self.p2D_mean, self.p2D_std)
            #self.rand_angle = rand(self.size**2).reshape(self.size,self.size)*2.0*pi
            #self.psd2D_GRF_Fourier = sqrt(self.psd2D_GRF) * [cos(self.rand_angle) + 1j * sin(self.rand_angle)]
            #self.GRF_image = fftpack.ifft2(ifftshift(self.psd2D_GRF_Fourier))[0]
            #self.GRF = sqrt(2)*real(self.GRF_image)
            #return self.GRF
        
    seednum=0
    Wx, cut = int(sys.argv[1]), int(sys.argv[2])
    Pkmap = PkappaGen(Wx)*mask_arr[Wx-1]

    print 'Wx, cut', Wx, cut
    galn = galnGen(Wx, cut)*mask_arr[Wx-1]
    igaln = galn.copy()
    
    random.seed(seednum)
    #x = WLanalysis.GRF_Gen(galn)    
    #Ckmap0 = CkappaGen(Wx)*mask_arr[Wx-1]
    #CFHTx = WLanalysis.GRF_Gen (Ckmap0)
    
    def iCC (i):
        
        #igaln = x.newGRF()*mask_arr[Wx-1]
        ######## # use Planck sim map, and CFHT GRF map
        Pkmap = PlanckSim15Gen(Wx, i)*mask_arr[Wx-1]
        #Ckmap = CFHTx.newGRF()
        Ckmap = load('/work/02977/jialiu/kSZ/CFHT/Noise/W%i_Noise_sigmaG10_%04d.npy'%(Wx, i))
        #############
        
        CCP = WLanalysis.CrossCorrelate(Pkmap, igaln, edges = edges_arr[Wx-1], sigmaG1=1.0, sigmaG2=1.0)[1]/fmask2_arr[Wx-1]/factor
        CCC = WLanalysis.CrossCorrelate(Ckmap, igaln, edges = edges_arr[Wx-1], sigmaG1=1.0, sigmaG2=1.0)[1]/fmask2_arr[Wx-1]/factor
        return CCP, CCC

    if not p.is_master():
        p.wait()
        sys.exit(0)
    CCsim_err_arr = array(p.map(iCC, range(100)))
    
    ####### referee
    #save(main_dir+'powspec/CC_Plancksim_CFHTrot_ludomask_cut%i_W%i.npy'%(cut, Wx), CCsim_err_arr)
    save(main_dir+'referee/CC_Plancksim_CFHTrot_ludomask_cut%i_W%i.npy'%(cut, Wx), CCsim_err_arr)

    p.close()
############# finish compute sim error #####################


############ calculate theory #################
if compute_model:
    for cut in (22,23,24):#cut=22
        print cut
        from scipy.integrate import quad
        z_center= arange(0.025, 3.5, 0.05)
        #referee
        dndzgal = load(main_dir+'dndz/dndz_0213_cut%s_noweight.npy'%(cut))[:,1]
        #dndzgal = load('/Users/jia/weaklensing/multiplicative/referee/dndz_0213_cut%s_noweight_lensfitWnonzero.npy'%(cut))[:,1]
        
        dndzkappa = load(main_dir+'dndz/dndz_0213_weighted.npy')[:,1]
        
        ######### Planck 15
        OmegaM = 0.3156#
        H0 = 67.27
        Ptable = genfromtxt('/Users/jia/weaklensing/cmblensing/P_delta_Planck15')
        ######## WMAP
        #Ptable = genfromtxt('/Users/jia/weaklensing/cmblensing/P_delta_Hinshaw')
        #OmegaM = 0.282#Planck15 0.3156#
        #H0 = 69.7 #Planck15 67.27
        
        #Ptable = genfromtxt('/Users/jia/weaklensing/cmblensing/P_delta_smith03_revised_colinparams')
        z_center = concatenate([[0,], z_center, [4.0,]])
        dndzgal = concatenate([[0,], dndzgal, [0,]])
        dndzkappa = concatenate([[0,], dndzkappa, [0,]])
        dndzgal /= 0.05*sum(dndzgal)
        dndzkappa /= 0.05*sum(dndzkappa)
        dndzgal_interp = interpolate.interp1d(z_center,dndzgal ,kind='cubic')
        dndzkappa_interp = interpolate.interp1d(z_center,dndzkappa ,kind='cubic')

        OmegaV = 1.0-OmegaM
        h = H0/100.0
        c = 299792.458#km/s
        H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV))
        DC_fcn = lambda z: c*quad(H_inv, 0, z)[0] # comoving distance Mpc
        z_ls = 1100 #last scattering
        z_arr = linspace(0, 4.0, 200)
        #zz=linspace(0,1100,300)
        DC4interp=array([DC_fcn(iz) for iz in z_arr])
        DC = interpolate.interp1d(z_arr, DC4interp)
        DC_ls = DC_fcn(z_ls)
        
        integrand = lambda zs, z: dndzkappa_interp(zs)*(1-DC(z)/DC(zs))
        W_wl_fcn = lambda z: quad(integrand, z, 4.0, args=(z,))[0]
        W_wl0 = array(map(W_wl_fcn, z_arr))
        W_wl = interpolate.interp1d(z_arr, W_wl0)
        W_cmb = lambda z: (1-DC(z)/DC_ls)
        ####### save for plotting #############
        #Wcfht_plot = W_wl0 * (1+z_arr) * H_inv(z_arr) * DC(z_arr)
        #Wcmb_plot = W_cmb(z_arr) * (1+z_arr) * H_inv(z_arr) * DC(z_arr)
        #save(main_dir+'plot/z_Wgal.npy',array([z_arr, Wcfht_plot]).T)
        #save(main_dir+'plot/z_Wcmb.npy',array([z_arr, Wcmb_plot]).T)
        ######################
        aa = array([1/1.05**i for i in arange(33)])
        zz = 1.0/aa-1 # redshifts
        kk = Ptable.T[0]
        iZ, iK = meshgrid(zz,kk)
        Z, K = iZ.flatten(), iK.flatten()
        P_deltas = Ptable[:,1:34].flatten()

        Pmatter_interp = interpolate.CloughTocher2DInterpolator(array([K*h, Z]).T, 2.0*pi**2*P_deltas/(K*h)**3)
        Pmatter = lambda k, z: Pmatter_interp (k, z)
        
        ell_arr2 = linspace(10, 2300, 200)

        ####### constant b ############
        Cplan_integrand = lambda z, ell: (1.0+z)/DC(z)*dndzgal_interp(z)*W_cmb(z)*Pmatter(ell/DC(z), z)
        Ccfht_integrand = lambda z, ell: (1.0+z)/DC(z)*dndzgal_interp(z)*W_wl(z)*Pmatter(ell/DC(z), z)
        ####### b=b0(1+z)
        #Ccfht_integrand = lambda z, ell: (1.0+z)**2/DC(z)*dndzgal_interp(z)*W_wl(z)*Pmatter(ell/DC(z), z)
        #Cplan_integrand = lambda z, ell: (1.0+z)**2/DC(z)*dndzgal_interp(z)*W_cmb(z)*Pmatter(ell/DC(z), z)
        
        #C_gal_auto_integrand = lambda z, ell: 1.0/DC(z)**2/(c*H_inv(z))*dndzgal_interp(z)**2*Pmatter(ell/DC(z), z)
        
        print 'Cplan_arr'
        Cplan_arr = 1.5*OmegaM*(H0/c)**2*array([quad(Cplan_integrand, 0.002, 3.5 , args=(iell))[0] for iell in ell_arr2])
        
        print 'Ccfht_arr'
        Ccfht_arr = 1.5*OmegaM*(H0/c)**2*array([quad(Ccfht_integrand, 0.002, 3.5 , args=(iell))[0] for iell in ell_arr2])
        
        ######### referee
        #save(main_dir+'powspec/Cplanck_cut%s_arr.npy'%(cut), array([ell_arr2, Cplan_arr]).T)
        #save(main_dir+'powspec/Ccfht_cut%s_arr.npy'%(cut), array([ell_arr2, Ccfht_arr]).T)
        
        #save(main_dir+'referee/Cplanck_cut%s_arr.npy'%(cut), array([ell_arr2, Cplan_arr]).T)
        #save(main_dir+'referee/Ccfht_cut%s_arr.npy'%(cut), array([ell_arr2, Ccfht_arr]).T)
        
        
        #print 'C_gal_auto'
        #C_gal_auto_arr = array([quad(C_gal_auto_integrand, 0.002, 3.5 , args=(iell))[0] for iell in ell_arr2])
        ##C_gal_auto_arr = 1.5*OmegaM*(H0/c)**2*array([quad(C_gal_auto_integrand, 0.002, 3.5 , args=(iell))[0] for iell in ell_arr2])
        #save(main_dir+'powspec/C_GalAuto_cut%s_arr.npy'%(cut), array([ell_arr2, C_gal_auto_arr]).T)

############ done theory ######################

################ test against omori and holder ##########

#for cut in (22,23, 24):
    #planck_CC_err = array([theory_CC_err(PkappaGen(Wx), galnGen(Wx,cut), Wx) for Wx in range(1,5)])

    #cfht_CC_err = array([theory_CC_err(CkappaGen(Wx), galnGen(Wx,cut), Wx) for Wx in range(1,5)])
    
    
    #planck_SNR = find_SNR (planck_CC_err[:,0,:], planck_CC_err[:,1,:])    
    #cfht_SNR = find_SNR (cfht_CC_err[:,0,:], cfht_CC_err[:,1,:])
    
    #print 'i<%i\tSNR(planck)=%.2f\tSNR(cfht)=%.2f (using mean, or 20 bins between ell=[50,1900])'%(cut,planck_SNR[0],cfht_SNR[0])
    #print 'i<%i\tSNR(planck)=%.2f\tSNR(cfht)=%.2f (all 20x4=80bins)'%(cut,planck_SNR[1],cfht_SNR[1])
    
    
############# plotting: 2 cross-correlation and theory #################
if compute_m_fit:
    ell_arr = ell_arr[:5]
    for cut in (22, 23, 24):
        ###### referee, with w>0 gals only
        #planck_CC_err = load(main_dir+'referee/planck_CC_err_%s_ludo.npy'%(cut))
        #cfht_CC_err = load(main_dir+'referee/cfht_CC_err_%s_ludo.npy'%(cut))
        planck_CC_err = load(main_dir+'powspec/planck_CC_err_%s_ludo.npy'%(cut))[:,:,:5]
        cfht_CC_err = load(main_dir+'powspec/cfht_CC_err_%s_ludo.npy'%(cut))[:,:,:5]
        
        #errK_arr2 = array([std(load(main_dir+'powspec/CC_Plancksim_CFHTrot_cut%i_W%i.npy'%(cut, Wx)), axis=0) for Wx in range(1,5)])*1e5*ell_arr        
        #planck_simerrdiag = errK_arr2[:,0,:]
        #cfht_simerrdiag = errK_arr2[:,1,:]
        #SNR, SNR2, CC_mean, err_mean = find_SNR(CC_arr/(1e5*ell_arr), errK_arr/(1e5*ell_arr))
        
        ########## referee
        #CC_sims = array([load(main_dir+'referee/CC_Plancksim_CFHTrot_ludomask_cut%i_W%i.npy'%(cut, Wx)) for Wx in range(1,5)])
        CC_sims = array([load(main_dir+'powspec/CC_Plancksim_CFHTrot_ludomask_cut%i_W%i.npy'%(cut, Wx))[:,:,:5] for Wx in range(1,5)])
        
        CC_conc=concatenate(swapaxes(concatenate(swapaxes(CC_sims, 1,-1)),0,1))
        ############ debugging ##########
        ## CC_sims.shape=(4, 100, 2, 6), need (2, 4, 6, 100)
        ## swapaxes(CC_sims, 1,-1)=(4, 6, 2, 100)
        ## concatenate(swapaxes(CC_sims, 1,-1)).shape=(24, 2, 100)
        ## swapaxes(concatenate(swapaxes(CC_sims, 1,-1)),0,1).shape = (2, 24, 100)
        ## CC_conc.shape=(48,100)
        #################################
        cov_sims = cov(CC_conc)
        
        
        CC_data = concatenate([concatenate(planck_CC_err[:,0,:]),concatenate(cfht_CC_err[:,0,:])])# shape=(2,4,6)

        err_data = concatenate([concatenate(planck_CC_err[:,1,:]),concatenate(cfht_CC_err[:,1,:])])
        
        ######## referee
        #ell_arr2, Cplanck_theory = load(main_dir+'referee/Cplanck_cut%s_arr.npy'%(cut)).T
        #ell_arr2, Ccfht_theory = load(main_dir+'referee/Ccfht_cut%s_arr.npy'%(cut)).T
        if bmodel ==0:
            ell_arr2, Cplanck_theory = load(main_dir+'powspec/Cplanck_cut%s_arr.npy'%(cut)).T
            ell_arr2, Ccfht_theory = load(main_dir+'powspec/Ccfht_cut%s_arr.npy'%(cut)).T        
        if bmodel == 1 or bmodel==2:
            ell_arr2, Cplanck_theory = load(main_dir+'powspec/Cplanck_cut%s_arr_bz.npy'%(cut)).T
            ell_arr2, Ccfht_theory = load(main_dir+'powspec/Ccfht_cut%s_arr_bz.npy'%(cut)).T
        
        
        #####################
        if bmodel < 2: # const, or b=b0(1+z)
            theorypoints_planck0 = interpolate.interp1d(ell_arr2, Cplanck_theory)(ell_arr)
            theorypoints_cfht0 = interpolate.interp1d(ell_arr2, Ccfht_theory)(ell_arr)
            theorypoints_planck = lambda b: b*concatenate(repeat(theorypoints_planck0.reshape(1,-1),4,axis=0))
            theorypoints_cfht = lambda b, m: b*m*concatenate(repeat(theorypoints_cfht0.reshape(1,-1),4,axis=0))
       
        else:######## use b=b0(1+z)-z = b0*z+b0-z; bug before calculates b=b0*z+1-z
            ell_arr2, Cplanck_theory_const = load(main_dir+'powspec/Cplanck_cut%s_arr.npy'%(cut)).T
            ell_arr2, Ccfht_theory_const = load(main_dir+'powspec/Ccfht_cut%s_arr.npy'%(cut)).T
            
            #theorypoints_planck1 = lambda b: interpolate.interp1d(ell_arr2, Cplanck_theory*b- Cplanck_theory+b*Cplanck_theory_const)(ell_arr)
            #theorypoints_cfht1 = lambda b,m: interpolate.interp1d(ell_arr2, Ccfht_theory*b*m-m*(Ccfht_theory-b*Ccfht_theory_const))(ell_arr)
            
            theorypoints_planck1 = lambda b: interpolate.interp1d(ell_arr2, Cplanck_theory*(b-1)+Cplanck_theory_const)(ell_arr)
            theorypoints_cfht1 = lambda b,m: interpolate.interp1d(ell_arr2, Ccfht_theory*(b-1)*m+m*Ccfht_theory_const)(ell_arr)
            
            theorypoints_planck = lambda b: concatenate(repeat(theorypoints_planck1(b).reshape(1,-1),4,axis=0))
            theorypoints_cfht = lambda b, m: concatenate(repeat(theorypoints_cfht1(b,m).reshape(1,-1),4,axis=0))
        ########################################
        
        ####################### 
        b_arr = linspace(0.0, 3.0, 1001)#301)#referee
        m_arr = linspace(0.0, 3.0, 1000)#300)#
        #m_arr = ones(2)###### to do quick fit for cfht or cmb only

        ######### sanity check against omori & holder ###########
        ###### for check: comment out this section
        idxhalf=len(cov_sims)/2
        if prefix == '_nonzeroOffDiagCov':
            covI_sims = mat(cov_sims).I
            covI_sims*=(100-cov_sims.shape[0]-2)/(100.-1)
            def chisq_func(b, m): 
                dn = mat(CC_data-concatenate([theorypoints_planck(b), theorypoints_cfht(b,m)]))
                return dn*covI_sims*dn.T
        else:
            
            covI_sims_block1 = (100-cov_sims.shape[0]/2.0-2)/(100.-1)*mat(cov_sims[:idxhalf,:idxhalf]).I
            covI_sims_block2 = (100-cov_sims.shape[0]/2.0-2)/(100.-1)*mat(cov_sims[idxhalf:,idxhalf:]).I
            def chisq_func(b, m): 
                #dn = mat(CC_data-concatenate([theorypoints_planck(b), theorypoints_cfht(b,m)]))
                dn1 = mat(CC_data[:idxhalf]-theorypoints_planck(b))
                dn2 = mat(CC_data[idxhalf:]-theorypoints_cfht(b,m))
                return dn1*covI_sims_block1*dn1.T+dn2*covI_sims_block2*dn2.T
        ###### uncomment this section: using planck x gal only
        #covI_sims20 = (100-cov_sims.shape[0]/2.0-2)/(100.-1)*mat(cov(CC_conc[:idxhalf,:])).I 
        #def chisq_func (b, m):
            #dn = mat(CC_data[:idxhalf]-theorypoints_planck(b))
            #return dn*covI_sims20*dn.T
        ####### or uncomment this section: using cfht x gal only
        #covI_sims20 = (100-cov_sims.shape[0]/2.0-2)/(100.-1)*mat(cov(CC_conc[idxhalf:,:])).I
        #def chisq_func (b, m):
            #dn = mat(CC_data[idxhalf:]-theorypoints_cfht(b,1))
            #return dn*covI_sims20*dn.T
        ########################################################
        
        
        ############## this block: calculate chisq for best fits #######
        prob=load(main_dir+'referee/prob%s_cut%i_%s_40bins.npy'%(prefix,cut,['Planck','bz','bTEGMARK'][bmodel]))
        idx = where(prob==amax(prob))
        b_best, m_best = b_arr[idx[0]], m_arr[idx[1]]
        ichisq = chisq_func(b_best, m_best)
        print ['const b','b=b0(1+z)','b=b0(1+z)-z'][bmodel], cut,float(ichisq), float(ichisq/(len(cov_sims)-2.0))
        ################################################################
        
        
        ############## this block: prob plane calculation
        #heatmap, prob = WLanalysis.prob_plane(chisq_func, b_arr, m_arr)
        #P_b = sum(prob,axis=1)
        #V_b = WLanalysis.findlevel(P_b)
        ####### dummy setting for comparing to omori holder ####       
        #P_m = sum(prob,axis=0)
        #V_m = WLanalysis.findlevel(P_m)
        ##################################
        
########## print out for latex marginalized errors #############
        #string = '''cut i<%i 
#68CL: b = $%.2f\substack{+%.2f \\\\ -%.2f}$ &$%.2f\substack{+%.2f \\\\ -%.2f}$
#95CL: b = $%.2f\substack{+%.2f \\\\ -%.2f}$ &$%.2f\substack{+%.2f \\\\ -%.2f}$
#99CL: b = $%.2f\substack{+%.2f \\\\ -%.2f}$ &$%.2f\substack{+%.2f \\\\ -%.2f}$
        #'''%(cut, 
       #b_arr[argmax(P_b)], b_arr[P_b>V_b[0]][-1]-b_arr[argmax(P_b)], b_arr[argmax(P_b)]-b_arr[P_b>V_b[0]][0], 
       #m_arr[argmax(P_m)], m_arr[P_m>V_m[0]][-1]-m_arr[argmax(P_m)], m_arr[argmax(P_m)]-m_arr[P_m>V_m[0]][0], 
       #b_arr[argmax(P_b)], b_arr[P_b>V_b[1]][-1]-b_arr[argmax(P_b)], b_arr[argmax(P_b)]-b_arr[P_b>V_b[1]][0], 
       #m_arr[argmax(P_m)], m_arr[P_m>V_m[1]][-1]-m_arr[argmax(P_m)], m_arr[argmax(P_m)]-m_arr[P_m>V_m[1]][0], 
       #b_arr[argmax(P_b)], b_arr[P_b>V_b[2]][-1]-b_arr[argmax(P_b)], b_arr[argmax(P_b)]-b_arr[P_b>V_b[2]][0], 
       #m_arr[argmax(P_m)], m_arr[P_m>V_m[2]][-1]-m_arr[argmax(P_m)], m_arr[argmax(P_m)]-m_arr[P_m>V_m[2]][0]) 
        
        #string = '''cut i<%i 
#68CL: b = $%.2f\substack{+%.2f \\\\ -%.2f}$
#95CL: b = $%.2f\substack{+%.2f \\\\ -%.2f}$
#99CL: b = $%.2f\substack{+%.2f \\\\ -%.2f}$
        #'''%(cut, 
       #b_arr[argmax(P_b)], b_arr[P_b>V_b[0]][-1]-b_arr[argmax(P_b)],b_arr[argmax(P_b)]-b_arr[P_b>V_b[0]][0],  
       #b_arr[argmax(P_b)], b_arr[P_b>V_b[1]][-1]-b_arr[argmax(P_b)],b_arr[argmax(P_b)]-b_arr[P_b>V_b[1]][0],  
       #b_arr[argmax(P_b)], b_arr[P_b>V_b[2]][-1]-b_arr[argmax(P_b)],b_arr[argmax(P_b)]-b_arr[P_b>V_b[2]][0],  
       #)
        #print ['const b','b=b0(1+z)','b=b0(1+z)-z'][bmodel]
        #print string   
#########end: print out for latex marginalized errors #############        
        #if bmodel==0:
            #save(main_dir+'referee/prob%s_cut%i_Planck_40bins.npy'%(prefix,cut),prob)
        #elif bmodel==1:
            #save(main_dir+'referee/prob%s_cut%i_bz_40bins.npy'%(prefix,cut),prob)
        #elif bmodel==2:
            #save(main_dir+'referee/prob%s_cut%i_bTEGMARK_40bins.npy'%(prefix,cut),prob)
        

    ##################  plotting for correlation mat, or probability plane###########  
        #from pylab import *
        #imshow(WLanalysis.corr_mat(cov_sims),interpolation='nearest',origin='lower')
        #title('correlation matrix (i<%s)'%(cut))
        #colorbar()
        #savefig(main_dir+'plot/corr_mat_cut%s.jpg'%(cut))
        #close()
        
        #figure(figsize=(6,6))
        #imshow(prob,origin='lower',interpolation='nearest',extent=[m_arr[0],m_arr[-1],b_arr[0],b_arr[-1]])
        #colorbar()
        #xlabel('m')
        #ylabel('b')
        #title('i<%s'%(cut))
        #show()
        #savefig(main_dir+'plot/probability_cut%s_ludomask.jpg'%(cut))
        #close()
        
if plot_omori_comp:
    ######### omori and holder data points #####
    a=genfromtxt('/Users/jia/Desktop/omoriholder_data.txt').T
    j = 0
    xgalnGen = lambda Wx, cut:  load('/Users/jia/Desktop/galn_test/galn_W%i_cut%i.npy'%(Wx,cut))
    xmaskGen = lambda Wx: load('/Users/jia/weaklensing/multiplicative/mask_ludo/ludomask_weight0_manu_W%i.npy'%Wx)
    weightGen = lambda Wx: load('/Users/jia/weaklensing/multiplicative/mask_ludo/ludoweight_weight0_W%i.npy'%Wx)
    def galnGen(Wx, cut):
        igaln=xgalnGen(Wx, cut)
        mask=xmaskGen(Wx)
        igaln=igaln/weightGen(Wx)
        igaln=igaln/mean(igaln[mask>0])-1
        igaln[mask<1]=0
        return igaln#WLanalysis.smooth(igaln,1.0)
    #def galnGen(Wx, cut):
        #igaln0=xgalnGen(Wx, cut)
        #mask=xmaskGen(Wx)
        #weight=weightGen(Wx)
        #igaln1=igaln0#/weight
        #igaln=igaln1/mean(igaln1[mask>0])-1
        #igaln[mask<1]=0
        #return igaln#
    def theory_CC_err(map1, map2, Wx, cut):
        mask=xmaskGen(Wx)*PmaskGen(Wx)
        map1*=mask
        map2*=mask
        fmask2=sum(mask)/sizes[Wx-1]**2
        fsky = fmask2*sizedeg_arr[Wx-1]/41253.0
        auto1 = WLanalysis.PowerSpectrum(map1, sizedeg = sizedeg_arr[Wx-1], edges=edges_arr[Wx-1],sigmaG=0.0)[-1]/fmask2/factor
        auto2 = WLanalysis.PowerSpectrum(map2, sizedeg = sizedeg_arr[Wx-1], edges=edges_arr[Wx-1],sigmaG=0.0)[-1]/fmask2/factor    
        err = sqrt(auto1*auto2/fsky/(2*ell_arr+1)/d_ell)
        CC = WLanalysis.CrossCorrelate(map1, map2, edges = edges_arr[Wx-1])[1]/fmask2/factor
        return CC, err
    ###########################################
    for cut in (22, 23, 24): ######## 3 field cross power spectrum
        ### compute C_ell, only needed once
        #planck_CC_err = array([theory_CC_err(PkappaGen(Wx), galnGen(Wx,cut), Wx, cut) for Wx in range(1,5)])
        #cfht_CC_err = array([theory_CC_err(CkappaGen(Wx), galnGen(Wx,cut), Wx, cut) for Wx in range(1,5)])
        #save(main_dir+'powspec/planck_CC_err_%s_ludo.npy'%(cut), planck_CC_err)
        #save(main_dir+'powspec/cfht_CC_err_%s_ludo.npy'%(cut), cfht_CC_err)
        
        
        planck_CC_err = load(main_dir+'powspec/planck_CC_err_%s_ludo.npy'%(cut))
        cfht_CC_err = load(main_dir+'powspec/cfht_CC_err_%s_ludo.npy'%(cut))
        
        planck_SNR = find_SNR (planck_CC_err[:,0,:], planck_CC_err[:,1,:])    
        cfht_SNR = find_SNR (cfht_CC_err[:,0,:], cfht_CC_err[:,1,:])
        
        #print 'i<%i\tSNR(planck)=%.2f\tSNR(cfht)=%.2f (using mean, or 6 bins)'%(cut,planck_SNR[0],cfht_SNR[0])
        #print 'i<%i\tSNR(planck)=%.2f\tSNR(cfht)=%.2f (using all 24 bins)'%(cut,planck_SNR[1],cfht_SNR[1])
        
        from pylab import *
        f=figure(figsize=(8,5))
        SNR_arr = ((planck_CC_err, planck_SNR), (cfht_CC_err, cfht_SNR))
        
        ##########simerr
        #errK_arr2 = array([std(load(main_dir+'powspec/CC_Plancksim_CFHTrot_cut%i_W%i.npy'%(cut, Wx)), axis=0) for Wx in range(1,5)])*1e5*ell_arr
        ##################
                
        for i in (1,):#2):
            if i == 1:
                proj='planck'
                
            if i == 2:
                proj='cfht'
                
            ell_theo, CC_theo = load(main_dir+'powspec/C%s_cut%s_arr.npy'%(proj,cut)).T
            
            ax=f.add_subplot(1,1,i)
            
            ######### omori and holder data points #########
            ell_omo, data_omo, err_omo = a[j+1], a[j+4], a[j+5]
            ax.errorbar(ell_omo, data_omo, err_omo*2,label='O&H15', fmt='o', linewidth=2)
            ################################################


            iCC, iSNR = SNR_arr[i-1]
            CC_arr = iCC[:,0,:]*ell_arr*1e5
            ax.plot(ell_theo, CC_theo*ell_theo*1e5, '--',label=' Theory (Planck 15 params)')

            errK_arr = iCC[:,1,:]*ell_arr*1e5
            SNR, SNR2, CC_mean, err_mean =iSNR
            
            ##########uncomment to use simerr
            #errK_arr = errK_arr2[:,i-1,:]
            #SNR, SNR2, CC_mean, err_mean = find_SNR(CC_arr/(1e5*ell_arr), errK_arr/(1e5*ell_arr))
            ##################
            
            print 'i<%i\tSNR(%s)=%.2f (6bins),\t%.2f (24bins)'%(cut,proj,SNR, SNR2)
            #print 'i<%i\tSNR(planck)=%.2f\tSNR(cfht)=%.2f (using all 24 bins)'%(cut,planck_SNR[1],cfht_SNR[1])
        
        
            CC_mean *= 1e5*ell_arr
            err_mean *= 1e5*ell_arr
            
            ax.bar(ell_arr, 2*err_mean, bottom=(CC_mean-err_mean), width=ones(len(ell_arr))*80, align='center',ec='brown',fc='none',linewidth=1.5, alpha=1.0)#
            
            ax.plot([0,2000], [0,0], 'k-', linewidth=1)
            seed(16)#good seeds: 6, 16, 25, 41, 53, 128, 502, 584
            for Wx in range(1,5):
                cc=rand(3)#colors[Wx-1]
                ax.errorbar(ell_arr+(Wx-2.5)*15, CC_arr[Wx-1], errK_arr[Wx-1], fmt='o',ecolor=cc,mfc=cc, mec=cc, label=r'$\rm W%i$'%(Wx), linewidth=1.2, capsize=0)
            leg=ax.legend(loc=3,fontsize=14,ncol=2)
            leg.get_frame().set_visible(False)
            ax.set_xlabel(r'$\ell$',fontsize=14)
            ax.set_xlim(0,2000)
            
            ax.set_ylabel(r'$\ell C_{\ell}^{\kappa_{%s}\Sigma}(\times10^{5})$'%(proj),fontsize=14)
            if i==1:
                ax.set_title('i<%s'%(cut), fontsize=14)# SNR(planck)=%.2f, SNR(cfht)=%.2f'%(cut, planck_SNR[0], cfht_SNR[0]),fontsize=14)
                #ax.set_ylim(-4,5)
            ax.tick_params(labelsize=14)
            ax.set_ylim(-5, 10)

        #show()
        #savefig(main_dir+'plot/ludoweight0_manu_CC_omoricomp_cut%s.jpg'%(cut))
        close()
        j+=6


############## official plot: data points ##############
if plot_cc:
    from pylab import *
    f=figure(figsize=(15,8))
    for cut in range(22,25):
        ######### referee
        #planck_CC_err = load(main_dir+'referee/planck_CC_err_%s_ludo.npy'%(cut))
        #cfht_CC_err = load(main_dir+'referee/cfht_CC_err_%s_ludo.npy'%(cut))
        #errK_arr2 = array([std(load(main_dir+'referee/CC_Plancksim_CFHTrot_ludomask_cut%i_W%i.npy'%(cut, Wx)), axis=0) for Wx in range(1,5)])
        planck_CC_err = load(main_dir+'powspec/planck_CC_err_%s_ludo.npy'%(cut))
        cfht_CC_err = load(main_dir+'powspec/cfht_CC_err_%s_ludo.npy'%(cut))
        errK_arr2 = array([std(load(main_dir+'powspec/CC_Plancksim_CFHTrot_ludomask_cut%i_W%i.npy'%(cut, Wx)), axis=0) for Wx in range(1,5)])
        
        #errK_arr2 = array([std(load(main_dir+'powspec/CC_Plancksim_CFHTrot_cut%i_W%i.npy'%(cut, Wx)), axis=0) for Wx in range(1,5)])
        
        planck_SNR = find_SNR (planck_CC_err[:,0,:], errK_arr2[:,0,:])
        cfht_SNR = find_SNR (cfht_CC_err[:,0,:], errK_arr2[:,1,:])       
        SNR_arr = ((planck_CC_err, planck_SNR), (cfht_CC_err, cfht_SNR))
        
        b_arr = linspace(0.0, 3.0, 1001)
        m_arr = linspace(0.0, 3.0, 1000)
        b_arr_bz = linspace(0.0, 3.0, 1001)
        m_arr_bz = linspace(0.0, 3.0, 1000)
        #prob = load(main_dir+'prob_cut%i.npy'%(cut))
        #prob_bz = load(main_dir+'prob_cut%i_bz.npy'%(cut))
        #prob_bz2 = load(main_dir+'prob_cut%i_bTEGMARK.npy'%(cut))
        
        prob = load(main_dir+'referee/prob%s_cut%i_Planck_40bins.npy'%(prefix,cut))
        prob_bz = load(main_dir+'referee/prob%s_cut%i_bz_40bins.npy'%(prefix,cut))
        prob_bz2 = load(main_dir+'referee/prob%s_cut%i_bTEGMARK_40bins.npy'%(prefix,cut))
        
        idx = where(prob==amax(prob))
        idx_bz = where(prob_bz==amax(prob_bz))
        idx_bz2 = where(prob_bz2==amax(prob_bz2))
        b_best, m_best = b_arr[idx[0]], m_arr[idx[1]]
        b_best_bz, m_best_bz = b_arr_bz[idx_bz[0]], m_arr_bz[idx_bz[1]]
        b_best_bz2, m_best_bz2 = b_arr_bz[idx_bz2[0]], m_arr_bz[idx_bz2[1]]
        print '(cut%i)b=%.2f, m=%.2f, b0=%.2f, m=%.2f'%(cut, b_best,m_best,b_best_bz,m_best_bz)
        
        for i in (1, 2):
            proj=['planck','cfht'][i-1]
            CC_arr = SNR_arr[i-1][0][:,0,:]
            
            ###### referee
            #ell_theo, CC_theo = load(main_dir+'referee/C%s_cut%s_arr.npy'%(proj,cut)).T
            ell_theo, CC_theo = load(main_dir+'powspec/C%s_cut%s_arr.npy'%(proj,cut)).T
            ell_theo = concatenate([[0,], ell_theo])
            CC_theo = concatenate([[0,], CC_theo])
            
            ell_theo_bz, CC_theo_bz= load(main_dir+'powspec/C%s_cut%s_arr_bz.npy'%(proj,cut)).T
            ell_theo_bz = concatenate([[0,], ell_theo_bz])
            CC_theo_bz = concatenate([[0,], CC_theo_bz])


            #ell_theo = linspace(0,2000,300)
            #ell_theo_interp = interpolate.interp1d(ell_theo0,CC_theo0,kind="cubic")
            #CC_theo = ell_theo_interp(ell_theo)
            
            SNR, SNR2, CC_mean, err_mean = SNR_arr[i-1][1]
            
            ax=f.add_subplot(2,3,(i-1)*3+cut-21)
            
            ######### theory curves ########
            
            #seed(15)
            seed(977)#good seed: 15, 66
            ax.plot(ell_theo, CC_theo*ell_theo*10**(4+i), '-',lw=2,label=r'$\rm{Planck\,} (b=1,\;m=1)$',color=rand(3))
            
            #ax.plot(ell_theo_bz, CC_theo_bz*ell_theo_bz*10**(4+i), '-',lw=2,label='Planck (b(z))',color=rand(3))
            ####### add bestfit curves ##########
            if i==1:
                ax.plot(ell_theo, CC_theo*ell_theo*10**(4+i)*b_best, '-',lw=1,label='best fit (b=%.2f)'%(b_best),color=rand(3))
                
                ax.plot(ell_theo, CC_theo_bz*ell_theo*10**(4+i)*b_best_bz, '--',lw=2,color=rand(3))
                rand(8)
                ax.plot(ell_theo, ell_theo*10**(4+i)*(b_best_bz2*CC_theo_bz-(CC_theo_bz-CC_theo)), '--',lw=1,color=rand(3))
            elif i==2:
                ax.plot(ell_theo, CC_theo*ell_theo*10**(4+i)*b_best*m_best, '-',lw=1,label=r'$\rm{best\, fit\, (const.}\,b)$',color=rand(3))
                ax.plot(ell_theo, CC_theo_bz*ell_theo*10**(4+i)*b_best_bz*m_best_bz, '--',lw=2,label=r'$\rm{best\, fit\, }(b=b_0(1+z))$',color=rand(3))
                rand(8)
                ax.plot(ell_theo, ell_theo*10**(4+i)*m_best_bz2*(b_best_bz2*CC_theo_bz-(CC_theo_bz-CC_theo)), '--',lw=1,label=r'$\rm{best\,fit}\,(b=\tilde{b}_0(1+z)-z)$',color=rand(3))
            ########################
            
            
            ax.bar(ell_arr, 2*err_mean*ell_arr*10**(4+i), bottom=(CC_mean-err_mean)*ell_arr*10**(4+i), width=ones(len(ell_arr))*80, align='center',ec='brown',fc='none',linewidth=1.5, alpha=1.0)#
            
            print 'i<%i\tSNR(%s)=%.2f (6bins),\t%.2f (24bins)'%(cut,proj,SNR, SNR2)
    
            seed(16)#good seeds: 6, 16, 25, 41, 53, 128, 502, 584
            for Wx in range(1,5):
                cc=rand(3)#colors[Wx-1]
                ax.errorbar(ell_arr+(Wx-2.5)*15, CC_arr[Wx-1]*ell_arr*10**(4+i), errK_arr2[Wx-1][i-1]*ell_arr*10**(4+i), fmt='o',ecolor=cc,mfc=cc, mec=cc, label=r'$\rm W%i$'%(Wx), linewidth=1.2, capsize=0)
            

            ax.plot([0,2300], [0,0], 'k--', linewidth=1)
            if i==2:
                handles0, labels0 = ax.get_legend_handles_labels()
                new_handles=[handles0[ii] for ii in (4,5,6,7,0,1,2,3)]
                new_labels=[labels0[ii] for ii in (4,5,6,7,0,1,2,3)]
                leg=ax.legend(new_handles, new_labels, loc=1,fontsize=12,ncol=2)
                #leg=ax.legend(loc=1,fontsize=12,ncol=2)
                leg.get_frame().set_visible(False)
                ax.set_xlabel(r'$\ell$',fontsize=20)
                ax.set_ylim(0.5, 8.8)
            ax.set_xlim(0,1990)
            
            if cut==22:
                ax.set_ylabel(r'$\ell C_{\ell}^{\kappa_{%s}\Sigma}\times 10^%i$'%(["cmb","gal"][i-1],4+i),fontsize=20)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)

            if i==1:
                ax.set_title('$18<i<%s$'%(cut),fontsize=20)
                plt.setp(ax.get_xticklabels(), visible=False) 
                ax.set_ylim(-3.5, 7.8)
                
            ax.tick_params(labelsize=16)
            #ax.set_yticks(linspace(-4,8,7))
    plt.subplots_adjust(hspace=0,wspace=0, left=0.08, right=0.98)
    show()
    #savefig(main_dir+'plot/official/CC_weight0mask%s.pdf'%(prefix))
    #savefig(main_dir+'plot/official/CC_weight0mask%s_test6bins.pdf'%(prefix))
    #savefig(main_dir+'plot/CC_weight0mask.png')
    #close()

if plot_contour:
    for bmodel in range(3):
        np.random.seed(25)
        #seed(25)
        #rand(6)
        fn=['Planck','bz','bTEGMARK'][bmodel]
        b_arr = linspace(0.0, 3.0, 1001)#[80:500]
        m_arr = linspace(0.0, 3.0, 1000)#[100:600]
        X, Y = np.meshgrid(m_arr, b_arr)
        iextent=[0,b_arr[-1],0,b_arr[-1]]
        labels = ["$18<i<%i$"%(cut) for cut in (22,23,24)]
        from pylab import *
        lines=[]
        f=figure(figsize=(8,6))
        ax=f.add_subplot(111)
        for cut in (22,23,24):
            ##### referee
            prob=load(main_dir+'referee/prob%s_cut%i_%s_40bins.npy'%(prefix,cut,fn))#[80:500,100:600]
            
            P_b = sum(prob,axis=1)        
            P_m = sum(prob,axis=0)
            V=WLanalysis.findlevel(prob)
            V_b = WLanalysis.findlevel(P_b)
            V_m = WLanalysis.findlevel(P_m)
            #string = '''cut i<%i 
#68CL: b = $%.2f\substack{+%.2f \\\\ -%.2f}$ &$%.2f\substack{+%.2f \\\\ -%.2f}$'''%(cut, 
       #b_arr[argmax(P_b)], b_arr[P_b>V_b[0]][-1]-b_arr[argmax(P_b)], b_arr[argmax(P_b)]-b_arr[P_b>V_b[0]][0], 
       #m_arr[argmax(P_m)], m_arr[P_m>V_m[0]][-1]-m_arr[argmax(P_m)], m_arr[argmax(P_m)]-m_arr[P_m>V_m[0]][0], 
       ##b_arr[argmax(P_b)], b_arr[P_b>V_b[1]][-1]-b_arr[argmax(P_b)], b_arr[argmax(P_b)]-b_arr[P_b>V_b[1]][0], 
       ##m_arr[argmax(P_m)], m_arr[P_m>V_m[1]][-1]-m_arr[argmax(P_m)], m_arr[argmax(P_m)]-m_arr[P_m>V_m[1]][0], 
       ##b_arr[argmax(P_b)], b_arr[P_b>V_b[2]][-1]-b_arr[argmax(P_b)], b_arr[argmax(P_b)]-b_arr[P_b>V_b[2]][0], 
       ##m_arr[argmax(P_m)], m_arr[P_m>V_m[2]][-1]-m_arr[argmax(P_m)], m_arr[argmax(P_m)]-m_arr[P_m>V_m[2]][0]
       #)
            string = '''cut i<%i 
    68 percent CL: b = %.2f -%.2f +%.2f, m = %.2f -%.2f +%.2f
    95 percent CL: b = %.2f -%.2f +%.2f, m = %.2f -%.2f +%.2f
    99 percent CL: b = %.2f -%.2f +%.2f, m = %.2f -%.2f +%.2f
            '''%(cut, 
        b_arr[argmax(P_b)], b_arr[argmax(P_b)]-b_arr[P_b>V_b[0]][0], b_arr[P_b>V_b[0]][-1]-b_arr[argmax(P_b)], 
        m_arr[argmax(P_m)], m_arr[argmax(P_m)]-m_arr[P_m>V_m[0]][0], m_arr[P_m>V_m[0]][-1]-m_arr[argmax(P_m)],
        b_arr[argmax(P_b)], b_arr[argmax(P_b)]-b_arr[P_b>V_b[1]][0], b_arr[P_b>V_b[1]][-1]-b_arr[argmax(P_b)], 
        m_arr[argmax(P_m)], m_arr[argmax(P_m)]-m_arr[P_m>V_m[1]][0], m_arr[P_m>V_m[1]][-1]-m_arr[argmax(P_m)],
        b_arr[argmax(P_b)], b_arr[argmax(P_b)]-b_arr[P_b>V_b[2]][0], b_arr[P_b>V_b[2]][-1]-b_arr[argmax(P_b)], 
        m_arr[argmax(P_m)], m_arr[argmax(P_m)]-m_arr[P_m>V_m[2]][0], m_arr[P_m>V_m[2]][-1]-m_arr[argmax(P_m)] )
            print ['const b','b=b0(1+z)','b=b0(1+z)=z'][bmodel]
            print string
            
            icolor=rand(3)
            #CS=ax.contourf(m_arr, b_arr, prob, [V[1], V[0], 1], origin='lower',colors=['b','g'])
            
            CS=ax.contour(X, Y, prob, levels=[V[0],], origin='lower', extent=iextent,linewidths=3.5, colors=[icolor, ])
            lines.append(CS.collections[0])
            
        
        leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':20},loc=0)
        leg.get_frame().set_visible(False)
        ax.tick_params(labelsize=16)
        ax.set_xlabel('$m$',fontsize=24)
        ax.set_ylabel([r'$b$',r'$b_0$',r'$\tilde{b}_0$'][bmodel],fontsize=24)
        ax.grid(True)
        plt.subplots_adjust(hspace=0, wspace=0, left=0.13, right=0.96,bottom=0.13, top=0.95)
        #ax.set_ylim(0.45, 1.35)#(0.2,1.0)
        if cut==24:
            #######referee
            #ax.imshow(prob[::-1,:], extent=[b_arr[0], b_arr[-1],m_arr[0], m_arr[-1]])
            #######
            ax.set_ylim(b_arr[P_b>V_b[2]][0]*0.9,b_arr[P_b>V_b[2]][-1]*1.1)
            ax.set_xlim(0.1,2.2)
        show()
        #savefig(main_dir+'plot/official/contour%s.png'%(['','_bz','_bz2'][bmodel]))
        #savefig(main_dir+'plot/official/contour%s%s.pdf'%(['','_bz','_bz2'][bmodel],prefix))
        close()
 
def steppify(arr,isX=False,interval=0):
    """
    Converts an array to double-length for step plotting
    """
    interval=0
    if isX and interval==0:
        interval = abs(arr[1]-arr[0]) / 2.0
    newarr = array(zip(arr-interval,arr+interval)).ravel()
    return newarr
    
if plot_kernel:
    from pylab import *
    z_center = arange(0.025, 3.5, 0.05)
    z_arr, Wcmb = load(main_dir+'plot/z_Wcmb.npy').T
    dndzgal_arr = array([load(main_dir+'dndz/dndz_0213_cut%s_noweight.npy'%(cut))[:,1] for cut in (22,23,24)])/0.05
    z_arr, Wcfht = load(main_dir+'plot/z_Wgal.npy').T
    f=figure(figsize=(8,6))
    ax=f.add_subplot(111)
    
    icolor=array([ 0.94101086,  0.56368138,  0.07799234])
    
    for cut in (22,23,24):
        ax.plot(z_center-0.025, dndzgal_arr[cut-22], lw=1, color='k',alpha=0.7**(abs(cut-25.5)), label="$dn/dz(18<i<%i)$"%(cut))
        ax.fill_between(z_center-0.025,dndzgal_arr[cut-22],facecolor=icolor, edgecolor='k',interpolate=False,alpha=0.7**(25-cut))
    seed(405)
    
    icolor2=rand(3)
    ax.plot(z_arr, Wcfht/amax(Wcfht)*1.8, "--", label="$W^{\kappa_{gal}}$",lw=2.5, color=icolor2)
    
    icolor3=rand(3)
    ax.plot(z_arr, Wcmb/amax(Wcmb)*1.5,"--", label="$W^{\kappa_{cmb}}$",lw=4, color=icolor3)

    leg=ax.legend(loc=1,fontsize=16,ncol=2)
    leg.get_frame().set_visible(False)
    
    ax.tick_params(labelsize=16)
    ax.set_xlabel('$z$',fontsize=20)
    ax.set_ylabel('$dn/dz$',fontsize=20)
    ax2 = ax.twinx()
    ax2.set_ylabel('$W^\kappa$',fontsize=20)
    plt.setp(ax2.get_yticklabels(), visible=False) 
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2.5)
    ax.grid(True)
    #show()
    savefig(main_dir+'plot/official/kernel.pdf')
    savefig(main_dir+'plot/kernel.jpg')
    close()
