##########################################################
### This code is for Jia's project B - 2nd try (4/18/2016)
### try to find
### observational evidence of over density at peak location
### as discovered by Yang 2011.
### It does the following:
### 1) create kappa_proj map, using L12 M_star->M_halo relation
### 2) relation between kappa_proj and kappa_lens
### 3) for each peak in kappa_proj, find all contributing halos, 
### relation N_halo (kappa_peak) for # of halos contributing to half kappa
### 4) same, but for random direction

import numpy as np
from scipy import *
from pylab import *
import os
import WLanalysis
from scipy import interpolate,stats
from scipy.integrate import quad
import scipy.optimize as op
import sys, os

make_kappa_predict = 0
if make_kappa_predict:
    ######## for stampede #####
    from emcee.utils import MPIPool
    obsPK_dir = '/home1/02977/jialiu/obsPK/'
else:
    ######## for laptop #####
    obsPK_dir = '/Users/jia/weaklensing/obsPK/'
    plot_dir = obsPK_dir+'plot/'

########### constants ######################

sizes = (1330, 800, 1120, 950)
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
PPR512=8468.416479647716
PPA512=2.4633625
c = 299792.458#km/s
Gnewton = 6.674e-8#cgs cm^3/g/s^2
H0 = 70.0#67.74
h = H0/100.0
OmegaM = 0.30#1#Planck15 TT,TE,EE+lowP+lensing+ext
OmegaV = 1.0-OmegaM
#rho_c0 = 9.9e-30#g/cm^3
M_sun = 1.989e33#gram
sigmaG_arr = (0.5, 1.0, 1.8, 3.5, 5.3, 8.9)

############################################
############ functions #####################
############################################

########### generate maps ##################

maskGen = lambda Wx: load(obsPK_dir+'mask/Mask_W%i_0.5_sigmaG10.npy'%(Wx))

kmapGen = lambda Wx, sigmaG: WLanalysis.readFits(obsPK_dir+'kappa_lens/W%i_KS_1.3_lo_sigmaG%02d.fit'%(Wx,sigmaG*10))*maskGen(Wx)

cat_gen = lambda Wx: np.load(obsPK_dir+'CFHTLens_2016-03-16T13-41-52_W%i.npy'%(Wx))
# columns: (0)ALPHA_J2000 (1)DELTA_J2000     
# (2)e1 (3)e2 (4)weight (5)MASK (6)Z_B (7)m (8)c2 
# (9)LP_Mr (10)LP_Mi (11)LP_Mz (12)LP_log10_SM_MED 
# (13)MAG_r (14)MAG_i (15)MAG_y (16)MAG_z

##############################################
########## cosmology #########################
##############################################

# growth factor
Hcgs = lambda z: H0*sqrt(OmegaM*(1+z)**3+OmegaV)*3.24e-20
H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV))
# luminosity distance Mpc
DC_integral = lambda z: c*quad(H_inv, 0, z)[0]
z_arr = linspace(0.1, 1.4, 1000)
DC_arr0 = array([DC_integral(z) for z in z_arr])
DC = interpolate.interp1d(z_arr, DC_arr0)
DA = lambda z: DC(z)/(1.0+z)
DL = lambda z: DC(z)*(1.0+z)
##rho_cz = lambda z: rho_c0*(OmegaM*(1+z)**3+(1-OmegaM))
rho_cz = lambda z: 0.375*Hcgs(z)**2/pi/Gnewton#critical density

########## update halo mass using L12
Mhalo_params_arr = [[12.520, 10.916, 0.457, 0.566, 1.53],
            [12.725, 11.038, 0.466, 0.610, 1.95],
            [12.722, 11.100, 0.470, 0.393, 2.51]]
##log10M1, log10M0, beta, sig, gamma
redshift_edges=[[0, 0.48], [0.48,0.74], [0.74, 1.30]]

def Mstar2Mhalo (Mstar_arr, redshift_arr):
    Mhalo_arr = zeros(len(Mstar_arr))
    for i in range(3):
        z0,z1 = redshift_edges[i]
        log10M1, log10M0, beta, sig, gamma = Mhalo_params_arr[i]
        print log10M1, log10M0, beta, sig, gamma 
        Mhalo_fcn = lambda log10Mstar: log10M1+beta*(log10Mstar-log10M0)+10.0**(sig*(log10Mstar-log10M0))/(1+10.0**(-gamma*(log10Mstar-log10M0)))-0.5
        idx = where((redshift_arr>z0)&(redshift_arr<=z1))[0]
        Mhalo_arr[idx] = Mhalo_fcn(Mstar_arr[idx])
    return Mhalo_arr
## Mhalo_L12 = 10**Mstar2Mhalo(iM_star, redshift)
## Note: need to check the definition of M_halo vs. M_vir, also R_vir

##############################################
######### create kappa_proj map ##############
##############################################

c_w, c_m, c_alpha, c_beta, c_gamma = 0.029, 0.097, -110.001, 2469.720, 16.885
cNFW_fcn = lambda z, Mvir: (Mvir/h) ** (c_w*z-c_m) * 10**(c_alpha/(z+c_gamma)+c_beta/(z+c_gamma)**2)

dd = lambda z: OmegaM*(1+z)**3/(OmegaM*(1+z)**3+OmegaV)
Delta_vir = lambda z: 18.0*pi**2+82.0*dd(z)-39.0*dd(z)**2

Rvir_fcn = lambda Mvir, z: (0.75/pi * Mvir*M_sun/(Delta_vir(z)*rho_cz(z)))**0.3333

def Gx_fcn (x, cNFW):#=5.0):
    '''projection function for a halo with cNFW, at location x=theta/theta_s.
    '''
    if x < 1:
        out = 1.0/(x**2-1.0)*sqrt(cNFW**2-x**2)/(cNFW+1.0)+1.0/(1.0-x**2)**1.5*arccosh((x**2+cNFW)/x/(cNFW+1.0))
    elif x == 1:
        out = sqrt(cNFW**2-1.0)/(cNFW+1.0)**2*(cNFW+2.0)/3.0
    elif 1 < x <= cNFW:
        out = 1.0/(x**2-1.0)*sqrt(cNFW**2-x**2)/(cNFW+1.0)-1.0/(x**2-1.0)**1.5*arccos((x**2+cNFW)/x/(cNFW+1.0))
    elif x > cNFW:
        out = 0
    return out

def kappa_proj (Mvir, Rvir, z_fore, x_fore, y_fore, z_back, x_back, y_back, DC_fore, DC_back):#, cNFW=5.0):
    '''return a function, for certain foreground halo, 
    calculate the projected mass between a foreground halo and a background galaxy pair. x, y(_fore, _back) are in radians.
    '''
    ######## updated next 2 lines to have a variable cNFW
    ### c0, beta = 11, 0.13 # lin & kilbinger2014 version
    ### cNFW = c0/(1+z_fore)*(Mvir/1e13)**(-beta)
    ### note 4/9/16, it was z_back before
    #cNFW=5.0
    cNFW = cNFW_fcn(z_fore, Mvir)
    f=1.0/(log(1.0+cNFW)-cNFW/(1.0+cNFW))# = 1.043 with cNFW=5.0
    two_rhos_rs = Mvir*M_sun*f*cNFW**2/(2*pi*Rvir**2)#cgs, see LK2014 footnote
    
    Dl_cm = 3.08567758e24*DC_fore/(1.0+z_fore)
    ## note: 3.08567758e24cm = 1Mpc###    
    SIGMAc = 347.29163*DC_back*(1+z_fore)/(DC_fore*(DC_back-DC_fore))
    ## note: SIGMAc = 1.07163e+27/DlDlsDs
    ## (c*1e5)**2/4.0/pi/Gnewton = 1.0716311756473212e+27
    ## 347.2916311625792 = 1.07163e+27/3.08567758e24
    theta = sqrt((x_fore-x_back)**2+(y_fore-y_back)**2)
    x = cNFW*theta*Dl_cm/Rvir 
    ## note: x=theta/theta_s, theta_s = theta_vir/c_NFW
    ## theta_vir=Rvir/Dl_cm
    Gx = Gx_fcn(x, cNFW)
    kappa_p = two_rhos_rs/SIGMAc*Gx
    return kappa_p
    
if make_kappa_predict:
    from scipy.spatial import cKDTree
    zcut = 0.2      #this is the lowest redshift for backgorund galaxies. use 0.2 to count for all galaxies.
    r = 0.006       # 0.002 rad = 7arcmin, 
            #within which I search for contributing halos

    Wx = int(sys.argv[1])
    center = centers[Wx-1]
    icat = cat_gen(Wx).T

    ra, dec, redshift, weight, MAGi, Mhalo, Rvir, DC_arr = icat
    ## varying DL
    #Mhalo[Mhalo>2e15] = 2e15#prevent halos to get crazy mass
    f_Wx = WLanalysis.gnom_fun(center)#turns to radians
    xy = array(f_Wx(icat[:2])).T

    idx_back = where(redshift>zcut)[0]
    xy_back = xy[idx_back]

    kdt = cKDTree(xy)
#nearestneighbors = kdt.query_ball_point(xy_back[:100], 0.002)
    def kappa_individual_gal (i):
        '''for individual background galaxies, find foreground galaxies within 20 arcmin and sum up the kappa contribution
        '''
        print i
        iidx_fore = array(kdt.query_ball_point(xy_back[i], r))  
        x_back, y_back = xy_back[i]
        z_back, DC_back = redshift[idx_back][i], DC_arr[idx_back][i]
        ikappa = 0
        for jj in iidx_fore:
            x_fore, y_fore = xy[jj]
            jMvir, jRvir, z_fore, DC_fore = Mhalo[jj], Rvir[jj], redshift[jj], DC_arr[jj]
            if z_fore >= z_back:
                kappa_temp = 0
            else:
                kappa_temp = kappa_proj (jMvir, jRvir, z_fore, x_fore, y_fore, z_back, x_back, y_back, DC_fore, DC_back, cNFW=5.0)
                if isnan(kappa_temp):
                    kappa_temp = 0
            ikappa += kappa_temp
            
            if kappa_temp>0:
                theta = sqrt((x_fore-x_back)**2+(y_fore-y_back)**2)
                print '%i\t%s\t%.2f\t%.3f\t%.3f\t%.4f\t%.6f'%(i, jj,log10(jMvir), z_fore, z_back, rad2arcmin(theta), kappa_temp)    
        return ikappa

    #a=map(kappa_individual_gal, randint(0,len(idx_back)-1,5))
    step=2e3
    
    def temp (ix):
        print ix
        temp_fn = obsPK_dir+'temp/kappa_proj%i_%07d.npy'%(Wx, ix)
        if not os.path.isfile(temp_fn):
            kappa_all = map(kappa_individual_gal, arange(ix, amin([len(idx_back), ix+step])))
            np.save(temp_fn,kappa_all)
    pool = MPIPool()
    ix_arr = arange(0, len(idx_back), step)
    pool.map(temp, ix_arr)
    
    all_kappa_proj = concatenate([np.load(obsPK_dir+'temp/kappa_proj%i_%07d.npy'%(Wx, ix)) for ix in ix_arr])
    np.save(obsPK_dir+'kappa_predict_W%i.npy'%(Wx), all_kappa_proj)
