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

make_kappaProj_cat = 0
make_kappaProj_map = 0
plot_maps = 0
xcorr_kappaProj_kappaLens = 0
plot_overlapping_peaks = 0
find_foreground_halos = 1

if make_kappaProj_cat or find_foreground_halos:
    ######## for stampede #####
    from emcee.utils import MPIPool
    obsPK_dir = '/work/02977/jialiu/obsPK/'
else:
    ######## for laptop #####
    obsPK_dir = '/Users/jia/weaklensing/obsPK/'
    plot_dir = obsPK_dir+'plot/'

########### constants ######################

sizes = (1330, 800, 1120, 950)
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
RA1 =(30.0, 39.0)#starting RA for W1
DEC1=(-11.5,-3.5)
RA2 =(132.0, 137.0)
DEC2=(-6.0,-0.5)
RA3 =(208.0, 221.0)
DEC3=(51.0, 58.0)
RA4 =(329.5, 336.0)
DEC4=(-1.2, 5.0)
RAs=(RA1,RA2,RA3,RA4)
DECs=(DEC1,DEC2,DEC3,DEC4)
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

maskGen = lambda Wx, sigmaG: load(obsPK_dir+'mask/Mask_W%i_0.5_sigmaG%02d.npy'%(Wx,sigmaG*10))

#klensGen = lambda Wx, sigmaG: WLanalysis.readFits(obsPK_dir+'kappa_lens/W%i_KS_1.3_lo_sigmaG%02d.fit'%(Wx,sigmaG*10))
klensGen = lambda Wx, sigmaG: WLanalysis.readFits(obsPK_dir+'kappa_lens/W%i_KS_0.4_hi_sigmaG%02d.fit'%(Wx,sigmaG*10))
blensGen = lambda Wx, sigmaG: WLanalysis.readFits(obsPK_dir+'kappa_lens/W%i_Bmode_0.4_hi_sigmaG%02d.fit'%(Wx,sigmaG*10))
kprojGen = lambda Wx, sigmaG: load(obsPK_dir+'kappa_proj/kproj_W%i_sigmaG%02d.npy'%(Wx, sigmaG*10))

cat_gen = lambda Wx: np.load(obsPK_dir+'cat/CFHTLens_2016-03-16T13-41-52_W%i.npy'%(Wx))
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

########## update halo mass using L12, table 5 SIG_MOD1
Mhalo_params_arr = [[12.520, 10.916, 0.457, 0.566, 1.53],
            [12.725, 11.038, 0.466, 0.610, 1.95],
            [12.722, 11.100, 0.470, 0.393, 2.51]]
##log10M1, log10M0, beta, sig, gamma
redshift_edges=[[0, 0.48], [0.48,0.74], [0.74, 1.30]]

def Mstar2Mhalo (Mstar_arr, redshift_arr):
    '''input log10 (Mstar_arr/Msun), return Mhalo with unit: log10(Mhalo/Mstar)
    '''
    Mhalo_arr = zeros(len(Mstar_arr))
    for i in range(3):
        z0,z1 = redshift_edges[i]
        log10M1, log10M0, beta, sig, gamma = Mhalo_params_arr[i]
        #print log10M1, log10M0, beta, sig, gamma 
        Mhalo_fcn = lambda log10Mstar: log10M1+beta*(log10Mstar-log10M0)+10.0**(sig*(log10Mstar-log10M0))/(1+10.0**(-gamma*(log10Mstar-log10M0)))-0.5
        idx = where((redshift_arr>z0)&(redshift_arr<=z1))[0]
        Mhalo_arr[idx] = Mhalo_fcn(Mstar_arr[idx])
    return Mhalo_arr

##############################################
######### create kappa_proj map ##############
##############################################

c_w, c_m, c_alpha, c_beta, c_gamma = 0.029, 0.097, -110.001, 2469.720, 16.885
cNFW_fcn = lambda z, Mvir: (Mvir/h) ** (c_w*z-c_m) * 10**(c_alpha/(z+c_gamma)+c_beta/(z+c_gamma)**2)

dd = lambda z: OmegaM*(1+z)**3/(OmegaM*(1+z)**3+OmegaV)
Delta_vir = lambda z: 18.0*pi**2+82.0*dd(z)-39.0*dd(z)**2

Rvir_fcn = lambda Mvir, z: (0.75/pi * Mvir*M_sun/(Delta_vir(z)*rho_cz(z)))**0.3333
## Rvir in unit of cm

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
    
    if z_fore>=z_back or Mvir==0:
        return 0.0
    else:
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
    
if make_kappaProj_cat:
    from scipy.spatial import cKDTree
    zcut = 0.4      #this is the lowest redshift for backgorund galaxies. use 0.2 to count for all galaxies.
    r = radians(20.0/60.0) #within which radius I search for contributing halos

    Wx = int(sys.argv[1])
    center = centers[Wx-1]

    ra, dec, redshift, weight, log10Mstar = cat_gen(Wx).T[[0,1,6,4,12]]
    Mhalo = 10**Mstar2Mhalo (log10Mstar, redshift)## unit of Msun
    Mhalo[log10Mstar==-99]=0
    Rvir = Rvir_fcn(Mhalo, redshift)
    DC_arr = DC(redshift)
    ## varying DL
    #Mhalo[Mhalo>2e15] = 2e15#prevent halos to get crazy mass
    f_Wx = WLanalysis.gnom_fun(center)#turns to radians
    xy = array(f_Wx(array([ra,dec]).T)).T

    idx_back = where((redshift>zcut)&(weight>0.001))[0]
    xy_back = xy[idx_back]

    kdt = cKDTree(xy)
    ## nearestneighbors = kdt.query_ball_point(xy_back[:100], 0.002)
    def kappa_individual_gal (i):
        '''for individual background galaxies, find foreground galaxies within 10 arcmin and sum up the kappa contribution
        '''
        
        iidx_fore = array(kdt.query_ball_point(xy_back[i], r))  
        x_back, y_back = xy_back[i]
        z_back, DC_back = redshift[idx_back][i], DC_arr[idx_back][i]
        ikappa = 0
        for jj in iidx_fore:
            x_fore, y_fore = xy[jj]
            jMvir, jRvir, z_fore, DC_fore = Mhalo[jj], Rvir[jj], redshift[jj], DC_arr[jj]
            if z_fore >= z_back or jMvir==0:
                kappa_temp = 0
            else:
                #print jMvir, jRvir, z_fore, x_fore, y_fore, z_back, x_back, y_back, DC_fore, DC_back
                kappa_temp = kappa_proj (jMvir, jRvir, z_fore, x_fore, y_fore, z_back, x_back, y_back, DC_fore, DC_back)
                if isnan(kappa_temp):
                    kappa_temp = 0
            ikappa += kappa_temp
            
            if kappa_temp>0:
                theta = sqrt((x_fore-x_back)**2+(y_fore-y_back)**2)
                print '%i\t%s\t%.2f\t%.3f\t%.3f\t%.4f\t%.6f'%(i, jj,log10(jMvir), z_fore, z_back, degrees(theta)*60, kappa_temp)  
        print i, ikappa
        return ikappa

    #a=map(kappa_individual_gal, randint(0,len(idx_back)-1,5))
    step=2e3
    
    def temp (ix):
        print ix
        temp_fn = obsPK_dir+'temp/cat_kappa_proj%i_%07d.npy'%(Wx, ix)
        if not os.path.isfile(temp_fn):
            kappa_all = map(kappa_individual_gal, arange(ix, amin([len(idx_back), ix+step])))
            np.save(temp_fn,kappa_all)
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    ix_arr = arange(0, len(idx_back), step)
    pool.map(temp, ix_arr)
    
    all_kappa_proj = concatenate([np.load(obsPK_dir+'temp/cat_kappa_proj%i_%07d.npy'%(Wx, ix)) for ix in ix_arr])
    np.save(obsPK_dir+'cat_kappa_proj_W%i.npy'%(Wx), all_kappa_proj)

    print 'DONE-DONE-DONE'

if make_kappaProj_map:
    zcut=0.4
    for Wx in (1,):#range(2,5):
        ik = load(obsPK_dir+'kappa_proj/cat_kappa_proj_W%i.npy'%(Wx))
        ra, dec, redshift, weight = cat_gen(Wx).T[[0,1,6,4]]
        idx_back = where((redshift>zcut)&(weight>0.001))[0]
        #xy_back = xy[idx_back]
        f_Wx = WLanalysis.gnom_fun(centers[Wx-1])
        y, x = array(f_Wx(array([ra,dec]).T[idx_back]))
        A, galn = WLanalysis.coords2grid(x, y, array([ik*weight[idx_back], weight[idx_back]]), size=sizes[Wx-1])
        Mkw, Mw = A
        for sigmaG in  (0.5, 1.0, 1.8, 3.5, 5.3, 8.9):
            print Wx, sigmaG
            #mask0 = maskGen(Wx, 0.5, sigmaG)
            #mask = WLanalysis.smooth(mask0, 5.0)
            ################ make maps ######################
            kmap_proj = WLanalysis.weighted_smooth(Mkw, Mw, sigmaG=sigmaG)
            #kmap_predict*=mask
            np.save(obsPK_dir+'kappa_proj/kproj_W%i_sigmaG%02d.npy'%(Wx, sigmaG*10), kmap_proj)

if plot_maps:
    #from zscale import zscale
    for Wx in range(1,5):#(1,):# 
        for sigmaG in (1.0, 1.8, 3.5, 5.3, 8.9):#(1.0,):# 
            ikmap_lens = klensGen(Wx, sigmaG)
            ikmap_proj = kprojGen(Wx, sigmaG)
            imask = maskGen(Wx, sigmaG)
            f=figure(figsize=(12,5))
            subplot(121)
            imean,istd=mean(ikmap_lens[imask>0]),std(ikmap_lens[imask>0])
            ikmap_lens[imask==0]=nan
            imshow(ikmap_lens*imask-imean,vmin=imean-3*istd,vmax=imean+3*istd,origin='lower')
            colorbar()
            title("W%i-lens, %.1f', std=%.3f"%(Wx,sigmaG,std(ikmap_lens[imask>0])))
            subplot(122)
            imean,istd=mean(ikmap_proj[imask>0]),std(ikmap_proj[imask>0])
            ikmap_proj[imask==0]=nan
            imshow(ikmap_proj*imask-imean,vmin=imean-3*istd,vmax=imean+3*istd,origin='lower')
            colorbar()
            title("W%i-proj, %.1f', std=%.3f"%(Wx,sigmaG,istd))
            
            plt.subplots_adjust(hspace=0,wspace=0.05, left=0.03, right=0.98)
            savefig(plot_dir+'kmap_W%i_sigmaG%02d.png'%(Wx, sigmaG*10))
            close()

if xcorr_kappaProj_kappaLens:
    #edgesGen = lambda Wx: logspace(0,log10(400),16)*sizes[Wx-1]/1330.0
    edgesGen = lambda Wx: linspace(1,250,11)*sizes[Wx-1]/1330.0
    ell_edges = edgesGen(1)*40
    delta_ell = ell_edges[1:]-ell_edges[:-1]
    sigmaG = 0.5
    #for sigmaG in sigmaG_arr:
    f=figure()
    seed(16)
    ax=f.add_subplot(111)
    for Wx in range(1,5):#(1,):# 
        #cc=rand(3)
        iedge=edgesGen(Wx)
        imask = maskGen(Wx, sigmaG)
        fmask=sum(imask)/float(sizes[Wx-1]**2)
        sizedeg = (sizes[Wx-1]/512.0)**2*12.25
        fsky=fmask*sizedeg/41253.0
        #for sigmaG in (, 1.8, 3.5, 5.3, 8.9):#(1.0,):# 
        ikmap_lens = klensGen(Wx, sigmaG)
        ikmap_proj = kprojGen(Wx, sigmaG)
        ibmap = blensGen(Wx, sigmaG)
        
        #ell_arr = WLanalysis.edge2center(iedge)*360./sqrt(sizedeg)
        #cc_proj_lens = WLanalysis.CrossCorrelate(ikmap_lens*imask, ikmap_proj*imask,edges=iedge)[1]/fmask
        #cc_proj_bmode = WLanalysis.CrossCorrelate(ibmap*imask, ikmap_proj*imask,edges=iedge)[1]/fmask
        #auto_lens = WLanalysis.CrossCorrelate(ikmap_lens*imask, ikmap_lens*imask,edges=iedge)[1]/fmask
        #auto_proj = WLanalysis.CrossCorrelate(ikmap_proj*imask, ikmap_proj*imask,edges=iedge)[1]/fmask
        
        #save(obsPK_dir+'PS/PS_W%i_sigmaG%02d.npy'%(Wx, sigmaG*10),[ell_arr, cc_proj_lens, cc_proj_bmode, auto_lens, auto_proj])
        
        ell_arr, cc_proj_lens, cc_proj_bmode, auto_lens, auto_proj = load(obsPK_dir+'PS/PS_W%i_sigmaG%02d.npy'%(Wx, sigmaG*10))
        
        delta_CC = sqrt((auto_lens*auto_proj+cc_proj_lens**2)/((2*ell_arr+1)*delta_ell*fsky))
        
        if Wx==1:
            
            ax.errorbar(ell_arr+(Wx-2.5)*40, cc_proj_lens, delta_CC,fmt='ko',linewidth=1, capsize=0, label=r'$\kappa_{\rm proj}\times \kappa_{\rm lens}$')   
            ax.errorbar(ell_arr+(Wx-2.5)*40, cc_proj_bmode, delta_CC, fmt='rd',linewidth=1, capsize=0, mec='r', label=r'$\kappa_{\rm proj}\times \kappa_{\rm B-mode}$')  
        else:
            ax.errorbar(ell_arr+(Wx-2.5)*60, cc_proj_lens, delta_CC,fmt='ko',linewidth=1, capsize=0) #ecolor=cc,mfc=cc, mec=cc,  
            ax.errorbar(ell_arr+(Wx-2.5)*60, cc_proj_bmode, delta_CC, fmt='rd',linewidth=1, capsize=0, mec='r') # label=r'$\rm W%i$'%(Wx)
        #ax.plot(ell_arr+(Wx-2.5)*50, auto_proj,'k-')
        #ax.plot(ell_arr+(Wx-2.5)*50, auto_lens,'k--')
    ax.plot([0,1e4],[0,0],'k--')
    ax.legend(frameon=0,fontsize=16)
    ax.set_ylabel(r'$\ell(\ell+1)/2\pi \times C_\ell$',fontsize=20)
    ax.set_xlabel(r'$\ell$',fontsize=20)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.rc('font', size=14)
    plt.subplots_adjust(hspace=0,wspace=0, left=0.16, right=0.9)
    ax.tick_params(labelsize=14)
    #show()
    #savefig(plot_dir+'CC_W%i_sigmaG%02d.pdf'%(Wx, sigmaG*10))
    savefig(plot_dir+'CC_W%i_sigmaG%02d.png'%(Wx, sigmaG*10))
    close()
    
if plot_overlapping_peaks:
    isigma = 5.3#3.5
    f=figure(figsize=(10,8))
    
    for Wx in range(1,5):
        ax=f.add_subplot(2,2,Wx)
        kmap_lens = klensGen(Wx, 8.9)
        kmap_proj = kprojGen(Wx, isigma)
        imask = maskGen(Wx, isigma)
        imask_nan = imask.copy()
        imask_nan[imask_nan==0]=nan
        #### find peaks in proj map
        kproj_peak_mat = WLanalysis.peaks_mat(kmap_proj)
        idx=where((kproj_peak_mat>0)&(imask>0))
        kappa_arr = kproj_peak_mat[idx]
        y0, x0 = idx
        y=y0/float(sizes[Wx-1])*(DECs[Wx-1][1]-DECs[Wx-1][0])+DECs[Wx-1][0]
        x=RAs[Wx-1][1]-x0/float(sizes[Wx-1])*(RAs[Wx-1][1]-RAs[Wx-1][0])
        if Wx==1:
            istd=std(kmap_lens[imask>0])
            k1,k0=amax(kappa_arr),amin(kappa_arr)
            dk=k1-k0
            
        #imshow(kmap_proj,origin='lower')
        s_arr = (kappa_arr-k0)*100/dk#400**(kappa_arr-amin(kappa_arr))
        im = ax.imshow(kmap_lens*imask_nan,origin='lower',vmin=-2*istd, vmax=2.5*istd,cmap='coolwarm',extent=[RAs[Wx-1][1],RAs[Wx-1][0],DECs[Wx-1][0],DECs[Wx-1][1]],aspect='auto')
        #colorbar()
        ax.scatter(x,y,s=s_arr,edgecolors='k',linewidths=1,facecolors='k')#'none')#
        ax.tick_params(labelsize=14)
    cbar_ax = f.add_axes([0.87, 0.1, 0.025, 0.85])#x0, y0, width, length
    f.colorbar(im, cax=cbar_ax)
    f.text(0.5, 0.02, r'$\rm {RA\,[deg]}$', ha='center', va='center',fontsize=20)
    f.text(0.06, 0.5, r'$\rm {DEC\,[deg]}$', ha='center', va='center', rotation='vertical',fontsize=20)

    plt.subplots_adjust(hspace=0.15,wspace=0.15, left=0.1, right=0.85,bottom=0.1,top=0.95)
    #show()
    savefig(plot_dir+'matching_peaks.png')
    #savefig(plot_dir+'matching_peaks.pdf')
    close()

if find_foreground_halos:
    ###### (1) identify peaks in the kappa_proj maps
    ###### (2) identify background halos that're within double the smoothing scale
    ###### (3) calculate contribution from each ith source to the total weight
    ###### (4) for jth foreground halo's contribution to k_ij 
    from scipy.spatial import cKDTree

        
    sigmaG = 1.0
    r = radians(10.0/60.0)
    
    zcut=0.4
    Wx = int(sys.argv[1])
    #for Wx in (4,):#range(1,5):#
    center = centers[Wx-1]
    #### convert from pixel to radians
    #rad2pix=lambda x: around(sizes[Wx-1]/2.0-0.5 + x*PPR512).astype(int)
    pix2rad = lambda xpix: (xpix-sizes[Wx-1]/2.0+0.5)/PPR512
    
    ###### (1) identify peaks in the kappa_proj maps
    ikmap_proj = kprojGen(Wx, sigmaG)
    imask = maskGen(Wx, sigmaG)
    kproj_peak_mat = WLanalysis.peaks_mat(ikmap_proj)
    idx_peaks=where((kproj_peak_mat>0)&(imask>0))
    ikappa_arr = kproj_peak_mat[idx_peaks]
    yx_peaks = pix2rad(array(idx_peaks)).T ## or xy..
    
    ##### (2) identify background halos that're within double the smoothing scale
    ra, dec, redshift, weight, log10Mstar = cat_gen(Wx).T[[0,1,6,4,12]]
    Mhalo = 10**Mstar2Mhalo (log10Mstar, redshift)## unit of Msun
    Mhalo[log10Mstar==-99]=0
    Rvir_arr = Rvir_fcn(Mhalo, redshift)
    DC_arr = DC(redshift)
    f_Wx = WLanalysis.gnom_fun(center)#turns to radians
    xy = array(f_Wx(array([ra,dec]).T)).T
    
    kdt = cKDTree(xy)
    
    ###### (3) calculate contribution from each ith source to the total weight
    def loop_over_peaks(mm):
        iyx = yx_peaks[mm]
    #for iyx in (yx_peaks[mm],):#yx_peaks[5:7]:#yx_peaks:#
        
        idx_all = array(kdt.query_ball_point(iyx[::-1], r/2))
        idx_back = idx_all[(redshift[idx_all]>0.4) & (weight[idx_all]>0.0001)]
        source_contribute = weight[idx_back]*exp(-0.5*sum((xy[idx_back]-iyx[::-1])**2,axis=1)/(radians(sigmaG/60.0))**2)
        #iy, ix= iyx
        ### make a matrix of ixj size, for ikappa at source i from lens j 
        ikappa_mat_ij = zeros((len(idx_back), len(idx_all)+1))
        ikappa_mat_ij[:,0]=source_contribute
        icounter=0
        for i in idx_back:
            jcounter=1
            for j in idx_all:
                #print icounter#,jcounter
                ikappa_mat_ij[icounter,jcounter] = kappa_proj (Mhalo[j], Rvir_arr[j], z_fore=redshift[j], x_fore=xy[j,0], y_fore=xy[j,1], z_back=redshift[i], x_back=xy[i,0], y_back=xy[i,1], DC_fore=DC_arr[j], DC_back=DC_arr[i])
                jcounter+=1
            icounter+=1
        print Wx, len(yx_peaks), mm, '%.4f %.4f'%(ikappa_arr[mm], sum(sum(ikappa_mat_ij[:,1:],axis=1)*source_contribute)/sum(source_contribute))
        #halos.append(ikappa_mat_ij)
        #kk+=1
        return ikappa_mat_ij
    #ikappa_mat_ij = loop_over_peaks(5)
    #source_contribute=ikappa_mat_ij.T[0]
    #print ikappa_arr[5], sum(sum(ikappa_mat_ij[:,1:],axis=1)*source_contribute)/sum(source_contribute)
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    halos = pool.map(loop_over_peaks, arange(len(yx_peaks)))
    save(obsPK_dir+'cat_halos_W%i_sigmaG%02d.npy'%(Wx, sigmaG*10), halos)

print 'DONE-DONE-DONE'