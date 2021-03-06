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
find_foreground_halos, random_direction = 0, 0
plot_N_peak, ttest = 1, 0## ttest is student t-test for probability
plot_halo_properties = 0
plot_concentration, plot_peaks_c15 = 0, 0
compare_peak_noise = 0
plot_hilo_peaks = 0
plot_yang2011_fig5 = 0
xcorr_peaks, peak_nob,clipped = 0, 0, 0#0=mapmpa, 1=peakmap, 2=peakpeak
plot_xcorr_peaks = 0
plot_peak_scatter = 0
test_lo_hi_contour = 0

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
        #cNFW = cNFW_fcn(z_fore, Mvir)
        cNFW = 1.5*cNFW_fcn(z_fore, Mvir)#####!!!test concentration
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
    #np.save(obsPK_dir+'cat_kappa_proj_W%i.npy'%(Wx), all_kappa_proj)
    np.save(obsPK_dir+'cat15c_kappa_proj_W%i.npy'%(Wx), all_kappa_proj)

    print 'DONE-DONE-DONE'
    pool.close()
    sys.exit(0)
    
if make_kappaProj_map:
    zcut=0.4
    for Wx in range(1,5):#(1,):#
        #ik = load(obsPK_dir+'kappa_proj/cat_kappa_proj_W%i.npy'%(Wx))
        ik = load(obsPK_dir+'kappa_proj/cat15c_kappa_proj_W%i.npy'%(Wx))
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
            np.save(obsPK_dir+'kappa_proj/kproj15c_W%i_sigmaG%02d.npy'%(Wx, sigmaG*10), kmap_proj)

if plot_maps:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #left, width = .25, .5
    #bottom, height = .25, .5
    #right = left + width
    #top = bottom + height
    #from zscale import zscale
    for Wx in (1,):#range(1,5):# 
        for sigmaG in (8.9,):#(1.0, 1.8, 3.5, 5.3, 8.9):# 
            ikmap_lens = klensGen(Wx, sigmaG)
            ikmap_proj = kprojGen(Wx, sigmaG)
            imask = maskGen(Wx, sigmaG)
            
            f=figure(figsize=(12,5.1))
            ax=f.add_subplot(121)
            imean,istd=mean(ikmap_lens[imask>0]),std(ikmap_lens[imask>0])
            ikmap_lens[imask==0]=nan
            im=ax.imshow(ikmap_lens*imask-imean,vmin=imean-2*istd,vmax=imean+2.5*istd,origin='lower',cmap='coolwarm',extent=[RAs[Wx-1][1],RAs[Wx-1][0],DECs[Wx-1][0],DECs[Wx-1][1]],aspect='auto')
            ipad=0.4
            ax2=f.add_subplot(122)
            imean2,istd2=mean(ikmap_proj[imask>0]),std(ikmap_proj[imask>0])
            ikmap_proj[imask==0]=nan
            im2=ax2.imshow(ikmap_proj*imask-imean2,vmin=imean-2*istd,vmax=imean+2.5*istd,origin='lower',cmap='coolwarm',extent=[RAs[Wx-1][1],RAs[Wx-1][0],DECs[Wx-1][0],DECs[Wx-1][1]],aspect='auto')
            #vmin=imean-3*istd,vmax=imean+3*istd
            cbar_ax = f.add_axes([0.9, 0.15, 0.025, 0.77])#x0, y0, width, length
            cbar_ax.tick_params(labelsize=14) 
            f.colorbar(im, cax=cbar_ax)
            ax.tick_params(labelsize=14)
            ax2.tick_params(labelsize=14)
            ax.set_ylim(DECs[Wx-1][0]+ipad,DECs[Wx-1][1]-ipad)
            ax2.set_ylim(DECs[Wx-1][0]+ipad,DECs[Wx-1][1]-ipad)
            plt.subplots_adjust(hspace=0,wspace=0.15, left=0.08, right=0.88,bottom=0.15,top=0.92)
            
            ax.text(0.8, 0.85, r'$\kappa_{\rm lens}$',fontsize=22,color='k',fontweight='bold',
            transform=ax.transAxes,
            bbox={'facecolor':'lightgrey', 'alpha':0.5})
            ax2.text(0.8, 0.85, r'$\kappa_{\rm proj}$',fontsize=22,color='k',fontweight='bold',
            transform=ax2.transAxes,
            bbox={'facecolor':'lightgrey', 'alpha':0.5})
            f.text(0.47, 0.03, r'$\rm {RA\,[deg]}$', ha='center', va='center',fontsize=20)
            f.text(0.03, 0.5, r'$\rm {DEC\,[deg]}$', ha='center', va='center', rotation='vertical',fontsize=20)
            #show()
            savefig(plot_dir+'kmap_W%i_sigmaG%02d.pdf'%(Wx, sigmaG*10))
            close()

if xcorr_kappaProj_kappaLens:
    #edgesGen = lambda Wx: logspace(0,log10(400),16)*sizes[Wx-1]/1330.0
    edgesGen = lambda Wx: linspace(1,250,11)*sizes[Wx-1]/1330.0
    ell_edges = edgesGen(1)*40
    delta_ell = ell_edges[1:]-ell_edges[:-1]
    sigmaG = 0.5
    #for sigmaG in sigmaG_arr:
    f=figure(figsize=(6,4))
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
        auto_bmode = WLanalysis.CrossCorrelate(ibmap*imask, ibmap*imask,edges=iedge)[1]/fmask
        
        #save(obsPK_dir+'PS/PS_W%i_sigmaG%02d.npy'%(Wx, sigmaG*10),[ell_arr, cc_proj_lens, cc_proj_bmode, auto_lens, auto_proj])
        
        ell_arr, cc_proj_lens, cc_proj_bmode, auto_lens, auto_proj = load(obsPK_dir+'PS/PS_W%i_sigmaG%02d.npy'%(Wx, sigmaG*10))
        
        delta_CC = sqrt((auto_lens*auto_proj+cc_proj_lens**2)/((2*ell_arr+1)*delta_ell*fsky))
        
        delta_CC_b = sqrt((auto_bmode*auto_proj+cc_proj_bmode**2)/((2*ell_arr+1)*delta_ell*fsky))
        
        if Wx==1:
            
            ax.errorbar(ell_arr+(Wx-2.5)*40, cc_proj_lens, delta_CC,fmt='o',c='orangered',mec='orangered',linewidth=1, capsize=0, label=r'$\kappa_{\rm proj}\times \kappa_{\rm lens}$')   
            ax.errorbar(ell_arr+(Wx-2.5)*40, cc_proj_bmode, delta_CC_b, fmt='kd',linewidth=1, capsize=0, mec='k', label=r'$\kappa_{\rm proj}\times \kappa_{\rm B-mode}$')  
        else:
            ax.errorbar(ell_arr+(Wx-2.5)*60, cc_proj_lens, delta_CC,fmt='o',c='orangered',mec='orangered',linewidth=1, capsize=0) #ecolor=cc,mfc=cc, mec=cc,  
            ax.errorbar(ell_arr+(Wx-2.5)*60, cc_proj_bmode, delta_CC_b, fmt='kd',linewidth=1, capsize=0, mec='k') # label=r'$\rm W%i$'%(Wx)
        #ax.plot(ell_arr+(Wx-2.5)*50, auto_proj,'k-')
        #ax.plot(ell_arr+(Wx-2.5)*50, auto_lens,'k--')
    ax.plot([0,1e4],[0,0],'k--')
    ax.legend(frameon=0,fontsize=16)
    ax.set_ylabel(r'$\ell(\ell+1)/2\pi \times C_\ell$',fontsize=20)
    ax.set_xlabel(r'$\ell$',fontsize=20)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.rc('font', size=14)
    plt.subplots_adjust(hspace=0,wspace=0, left=0.18, right=0.9,bottom=0.16)
    ax.tick_params(labelsize=14)
    #ax.grid(True)
    #show()
    savefig(plot_dir+'CC_sigmaG%02d.png'%(sigmaG*10))
    savefig(plot_dir+'CC_sigmaG%02d.pdf'%(sigmaG*10))
    close()
    
if plot_overlapping_peaks:
    isigma = 8.9#3.5
    f=figure(figsize=(10,8))
    
    for Wx in range(1,5):
        ax=f.add_subplot(2,2,Wx)
        ############## swapped kmap_lens and kmap_proj as requested by zoltan
        #kmap_lens = klensGen(Wx, isigma)
        #kmap_proj = kprojGen(Wx, isigma)
        kmap_lens = kprojGen(Wx, isigma)#8.9)
        kmap_proj = klensGen(Wx, isigma)
        
        imask = maskGen(Wx, isigma)
        imask_nan = imask.copy()
        imask_nan[imask_nan==0]=nan
        
        kmap_lens -= mean(kmap_lens[imask>0])
        kmap_proj *= WLanalysis.smooth(imask, 5)
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
        s_arr = (kappa_arr-k0)*50/dk/2.0
        #s_arr = (kappa_arr-k0)*100/dk#400**(kappa_arr-amin(kappa_arr))
        im = ax.imshow(kmap_lens*imask_nan,origin='lower',vmin=-2*istd, vmax=2.5*istd,cmap='coolwarm',extent=[RAs[Wx-1][1],RAs[Wx-1][0],DECs[Wx-1][0],DECs[Wx-1][1]],aspect='auto')#coolwarm
        #colorbar()
        ax.scatter(x,y,s=s_arr,edgecolors='k',linewidths=1,facecolors='k')#'none')#
        ax.tick_params(labelsize=14)
        ipad = 0.1
        ax.set_xlim(RAs[Wx-1][1]-ipad,RAs[Wx-1][0]+ipad)
        ax.set_ylim(DECs[Wx-1][0]+ipad,DECs[Wx-1][1]-ipad)
    cbar_ax = f.add_axes([0.87, 0.1, 0.025, 0.85])#x0, y0, width, length
    cbar_ax.tick_params(labelsize=14) 
    f.colorbar(im, cax=cbar_ax)
    f.text(0.5, 0.03, r'$\rm {RA\,[deg]}$', ha='center', va='center',fontsize=20)
    f.text(0.03, 0.5, r'$\rm {DEC\,[deg]}$', ha='center', va='center', rotation='vertical',fontsize=20)

    plt.subplots_adjust(hspace=0.15,wspace=0.15, left=0.1, right=0.85,bottom=0.1,top=0.95)
    #show()
    savefig(plot_dir+'matching_klenspeaks_sigmaG%02d.pdf'%(isigma*10))
    #savefig(plot_dir+'matching_peaks_sigmaG%02d.pdf'%(isigma*10))
    close()

if find_foreground_halos:
    ###### (1) identify peaks in the kappa_proj maps
    ###### (2) identify background halos that're within double the smoothing scale
    ###### (3) calculate contribution from each ith source to the total weight
    ###### (4) for jth foreground halo's contribution to k_ij 
    from scipy.spatial import cKDTree

        
    sigmaG = 1.0
    r = radians(10.0/60.0)#
    
    zcut=0.4
    Wx = int(sys.argv[1])
    #for Wx in (4,):#range(1,5):#
    center = centers[Wx-1]
    #### convert from pixel to radians
    #rad2pix=lambda x: around(sizes[Wx-1]/2.0-0.5 + x*PPR512).astype(int)
    pix2rad = lambda xpix: (xpix-sizes[Wx-1]/2.0+0.5)/PPR512
    
    ###### (1) identify peaks in the kappa_proj maps
    ## ikmap = kprojGen(Wx, sigmaG)###### using kappa_proj
    ikmap = klensGen(Wx, sigmaG) ## use kappa_lens
    imask = maskGen(Wx, sigmaG)
    kappa_peak_mat = WLanalysis.peaks_mat(ikmap)
    if not random_direction:
        idx_peaks=where((kappa_peak_mat>0)&(imask>0))
        ikappa_arr = kappa_peak_mat[idx_peaks]
        yx_peaks = pix2rad(array(idx_peaks)).T ## or xy..    
    else:###### (1b) identify non peaks
        idx_non_peaks=array(where(isnan(kappa_peak_mat)*imask>0)).T
        seed(0)
        sample_non_peaks = randint(0, len(idx_non_peaks), 1e4)
        idx_peaks = list(idx_non_peaks[sample_non_peaks].T)
        ikappa_arr = ikmap_proj[idx_peaks]
        yx_peaks = pix2rad(array(idx_peaks)).T
        
    
    
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
        idx_all = array(kdt.query_ball_point(iyx[::-1], r/2))
        idx_back = idx_all[(redshift[idx_all]>0.4) & (weight[idx_all]>0.0001)]
        source_contribute = weight[idx_back]*exp(-0.5*sum((xy[idx_back]-iyx[::-1])**2,axis=1)/(radians(sigmaG/60.0))**2)
        ### make a matrix of ixj size, for ikappa at source i from lens j 
        ikappa_mat_ij = zeros((len(idx_back), len(idx_all)))
        icounter=0
        for i in idx_back:
            jcounter=0
            for j in idx_all:
                #print icounter#,jcounter
                ikappa_mat_ij[icounter,jcounter] = kappa_proj (Mhalo[j], Rvir_arr[j], z_fore=redshift[j], x_fore=xy[j,0], y_fore=xy[j,1], z_back=redshift[i], x_back=xy[i,0], y_back=xy[i,1], DC_fore=DC_arr[j], DC_back=DC_arr[i])
                jcounter+=1
            icounter+=1
        halos=sum(ikappa_mat_ij*source_contribute.reshape(-1,1),axis=0)/sum(source_contribute)
        
        ###### actual comsum
        halos_contrib_comsum=cumsum(sort(halos)[::-1])/sum(halos)
        out = zeros(21)
        out[0]=sum(halos)
        out[1:]=halos_contrib_comsum[:20]
        
        ###### index of the top 20 halos
        out_idx=zeros(21)
        out_idx[0]=sum(halos)
        out_idx[1:] = idx_all[argsort(halos)[::-1][:20]]
        
        print Wx, len(yx_peaks),'%04d %.4f %.4f'%(mm, ikappa_arr[mm],sum(halos))
        return out,out_idx#ikappa_mat_ij
    #out = loop_over_peaks(5)
    
    
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    
    #from multiprocessing import Pool
    #pool = Pool(3)
    if not random_direction and Wx != 1:
        
        if Wx == 3:
            istep=int(len(yx_peaks)/4)
            allstep=int(len(yx_peaks))
            
            out_arr = pool.map(loop_over_peaks, arange(istep))#arange(5))#
            save(obsPK_dir+'top20lens_halosidx_W3_sigmaG%02d.npy'%(sigmaG*10), out_arr)
            
            out_arr2 = pool.map(loop_over_peaks, arange(istep,istep*2))#arange(5))#
            save(obsPK_dir+'top20lens_halosidx_W8_sigmaG%02d.npy'%(sigmaG*10), out_arr2)
            
            out_arr3 = pool.map(loop_over_peaks, arange(istep*2,istep*3))#arange(5))#
            save(obsPK_dir+'top20lens_halosidx_W9_sigmaG%02d.npy'%(sigmaG*10), out_arr3)
            
            out_arr4 = pool.map(loop_over_peaks, arange(istep*3,allstep))#arange(5))#
            save(obsPK_dir+'top20lens_halosidx_W10_sigmaG%02d.npy'%(sigmaG*10), out_arr4)
        else: 
            out_arr = pool.map(loop_over_peaks, arange(len(yx_peaks)))#arange(5))#
        save(obsPK_dir+'top20lens_halosidx_W%i_sigmaG%02d.npy'%(Wx, sigmaG*10), out_arr)

        ######## solve W1 problem, cut in half
    elif not random_direction and Wx == 1:
        istep=int(len(yx_peaks)/4)
        allstep=int(len(yx_peaks))
        
        out_arr = pool.map(loop_over_peaks, arange(istep))#arange(5))#
        save(obsPK_dir+'top20lens_halosidx_W1_sigmaG%02d.npy'%(sigmaG*10), out_arr)
        
        out_arr2 = pool.map(loop_over_peaks, arange(istep,istep*2))#arange(5))#
        save(obsPK_dir+'top20lens_halosidx_W5_sigmaG%02d.npy'%(sigmaG*10), out_arr2)
        
        out_arr3 = pool.map(loop_over_peaks, arange(istep*2,istep*3))#arange(5))#
        save(obsPK_dir+'top20lens_halosidx_W6_sigmaG%02d.npy'%(sigmaG*10), out_arr3)
        
        out_arr4 = pool.map(loop_over_peaks, arange(istep*3,allstep))#arange(5))#
        save(obsPK_dir+'top20lens_halosidx_W7_sigmaG%02d.npy'%(sigmaG*10), out_arr4)
       
    else:
        out_arr = pool.map(loop_over_peaks, arange(0,2500) )
        save(obsPK_dir+'top20_random1_sigmaG%02d.npy'%(sigmaG*10), out_arr)
        out_arr2 = pool.map(loop_over_peaks, arange(2500,5000) )
        save(obsPK_dir+'top20_random2_sigmaG%02d.npy'%(sigmaG*10), out_arr2)
        out_arr3 = pool.map(loop_over_peaks, arange(5000,7500) )
        save(obsPK_dir+'top20_random3_sigmaG%02d.npy'%(sigmaG*10), out_arr3)
        out_arr4 = pool.map(loop_over_peaks, arange(7500,10000) )
        save(obsPK_dir+'top20_random4_sigmaG%02d.npy'%(sigmaG*10), out_arr4)

    
    print 'DONE-DONE-DONE'
    pool.close()
    sys.exit(0)
def gen_peaks_list(Wx,sigmaG=1.0):
    ikmap = klensGen(Wx, sigmaG) ## use kappa_lens
    kmap_peak_mat = WLanalysis.peaks_mat(ikmap)
    imask = maskGen(Wx, sigmaG)
    ikappa_arr = ikmap[(kmap_peak_mat>0)&(imask>0)]
    return ikappa_arr

if plot_N_peak:
    do_lens = 1

    b=concatenate([load(obsPK_dir+'top20_random%i_sigmaG10.npy'%(i)) for i in range(1,5)],axis=0)

    if do_lens:### kappa_lens
        a=concatenate([load(obsPK_dir+'top20lens_halosidx_W%i_sigmaG10.npy'%(i))[:,0] for i in range(1,5)],axis=0)
        kappa_peaks = concatenate(map(gen_peaks_list,range(1,5)))### use klens bins
        
        kappa_peak_mat = WLanalysis.peaks_mat(kprojGen(1, 1.0))
        imask = maskGen(1, 1.0)
        idx_non_peaks=array(where(isnan(kappa_peak_mat)*imask>0)).T
        seed(0)
        sample_non_peaks = randint(0, len(idx_non_peaks), 1e4)
        idx_peaks = list(idx_non_peaks[sample_non_peaks].T)
        #ikappa_arr = ikmap_proj[idx_peaks]
        
        rkappa_peaks = klensGen(1,1.0)[idx_peaks]
    else:### kappa_proj
        a=concatenate([load(obsPK_dir+'top20_halos_W%i_sigmaG10.npy'%(i)) for i in range(1,6)],axis=0)
        kappa_peaks = a[:,0]### use kproj bins
        rkappa_peaks = b[:,0]
        
        

    N_halos = sum((a[:,1:]<0.5),axis=1)+1

    rN_halos = sum((b[:,1:]<0.5),axis=1)+1

    N_mean, N_std = zeros((2,20))
    rN_mean, rN_std = zeros((2,20))
    hi_kappa=0.05#0.2
    kappa_edges = linspace(0.005, hi_kappa,21)
    
    from scipy import stats
    ttest_arr = []
    for i in range(20):
        k0,k1 = kappa_edges[i:i+2]
        iN_halos = N_halos[(kappa_peaks>k0) &(kappa_peaks<k1)]
        irN_halos = rN_halos[(rkappa_peaks>k0) &(rkappa_peaks<k1)]
        
        N_mean[i]=mean(iN_halos)
        N_std[i]=std(iN_halos)
        
        rN_mean[i]=mean(irN_halos)
        rN_std[i]=std(irN_halos)

        ittest=stats.ttest_ind(iN_halos,irN_halos)
        ttest_arr.append(ittest)
        print 0.5*(k1+k0), ittest[1], N_mean[i], rN_mean[i]
        
    if not ttest:
        np.random.seed(399)
        cc,cc2=['orangered','green']#rand(2,3)
        f=figure(figsize=(6,4))
        ax=f.add_subplot(111)
        #ax.hist2d(kappa_peaks, N_halos,bins=(20,20),range=((0.005,hi_kappa),(0.5,20.5)),cmap='Greys')
        ## xx,yy,zz=histogram2d(kappa_peaks, N_halos,bins=(15,20),range=((0.005,hi_kappa),(0.5,20.5)))
        ## N_mean=sum(xx*WLanalysis.edge2center(zz).reshape(1,-1),axis=1)/sum(xx,axis=1)
        ax.errorbar(WLanalysis.edge2center(kappa_edges)-2e-4, N_mean, N_std,fmt='o',c=cc,ecolor=cc,mfc=cc, mec=cc,lw=1.5,capsize=0,label=r'${\rm peaks}$')
        ax.errorbar(WLanalysis.edge2center(kappa_edges)+2e-4, rN_mean, rN_std,fmt='d',c=cc2,ecolor=cc2,mfc=cc2, mec=cc2,lw=1.5,capsize=0,label=r'${\rm non-peaks}$')
        ax.tick_params(labelsize=16)
        #ax.set_xlim(0.00499,0.0499)
        ax.set_ylim(-0.3,17)
        ax.set_xlabel(r'$\kappa$',fontsize=22)
        ax.set_ylabel(r'$N_{\rm halo}$',fontsize=22)
        #ax.set_yscale('log')
        ax.legend(frameon=0,fontsize=20,loc='upper right')
        plt.subplots_adjust(hspace=0,wspace=0.05, left=0.13, right=0.95,bottom=0.16,top=0.92)
        #show()
        savefig(plot_dir+'Nhalo_peak_lens0.png')
        close()

if plot_halo_properties:
    sigmaG=1.0
    do_lens=1# 1=kappa_lens, else kappa_proj
    def properties_fcn(Wx):
        print Wx
        pix2rad = lambda xpix: (xpix-sizes[Wx-1]/2.0+0.5)/PPR512
        ra, dec, redshift, weight, log10Mstar = cat_gen(Wx).T[[0,1,6,4,12]]
        Mhalo = 10**Mstar2Mhalo (log10Mstar, redshift)## unit of Msun
        Mhalo[log10Mstar==-99]=0
        Rvir_arr = Rvir_fcn(Mhalo, redshift)
        DC_arr = DC(redshift)
        Rvir_theta_arr = Rvir_arr/(DC_arr*3.08567758e24/(1+redshift))
        f_Wx = WLanalysis.gnom_fun(centers[Wx-1])#turns to radians
        xy = array(f_Wx(array([ra,dec]).T)).T
        
        ########## kappa_proj peaks
        if not do_lens:
            idx_halo=load(obsPK_dir+'top20_halosidx_W%i_sigmaG10.npy'%(Wx))
            if Wx == 1:
                idx_halo = concatenate([idx_halo,load(obsPK_dir+'top20_halosidx_W5_sigmaG10.npy')])
            ikmap_proj = kprojGen(Wx, sigmaG)

        ######### kappa_lens peaks
        else:
            idx_halo=load(obsPK_dir+'top20lens_halosidx_W%i_sigmaG10.npy'%(Wx))[:,1,1:].astype(int)
            ikmap_proj = klensGen(Wx, sigmaG)

        imask = maskGen(Wx, sigmaG)
        kproj_peak_mat = WLanalysis.peaks_mat(ikmap_proj)
        idx_peaks=where((kproj_peak_mat>0)&(imask>0))
        yx_peaks = pix2rad(array(idx_peaks)).T 
        
        ikappa_arr = kproj_peak_mat[idx_peaks]
        
        idistance_rad_arr = sqrt(sum((xy[idx_halo]-yx_peaks[:,::-1].reshape(-1,1,2))**2,axis=-1))
        
        idistance_rvir_arr = idistance_rad_arr/Rvir_theta_arr[idx_halo]
        
        imass_arr = Mhalo[idx_halo]
        
        iredshift_arr = redshift[idx_halo]
        
        return ikappa_arr, idistance_rad_arr, idistance_rvir_arr, imass_arr, iredshift_arr
    #out = map(properties_fcn, range(1,5))
    #ikappa_arr = concatenate([out[Wx][0] for Wx in range(4)])
    #idistance_rad_arr = concatenate([out[Wx][1] for Wx in range(4)],axis=0)
    #idistance_rvir_arr = concatenate([out[Wx][2] for Wx in range(4)],axis=0)
    #imass_arr = concatenate([out[Wx][3] for Wx in range(4)],axis=0)
    #iredshift_arr = concatenate([out[Wx][4] for Wx in range(4)],axis=0)
    #save(obsPK_dir+'halo_properties_lens.npy',[idistance_rad_arr, idistance_rvir_arr, imass_arr, iredshift_arr])
    
    if not do_lens:
        kappa_halo = concatenate([load(obsPK_dir+'top20_halos_W%i_sigmaG10.npy'%(i)) for i in range(1,5)],axis=0)[:,1:]
        ikappa_arr = concatenate([load(obsPK_dir+'top20_halos_W%i_sigmaG10.npy'%(i)) for i in range(1,5)],axis=0)[:,0]
        idistance_rad_arr, idistance_rvir_arr, imass_arr, iredshift_arr = load(obsPK_dir+'halo_properties.npy')
    
    else:
        kappa_halo = concatenate([load(obsPK_dir+'top20lens_halosidx_W%i_sigmaG10.npy'%(i)) for i in range(1,5)],axis=0)[:,0,1:]
        ikappa_arr = concatenate([load(obsPK_dir+'top20lens_halosidx_W%i_sigmaG10.npy'%(i)) for i in range(1,5)],axis=0)[:,0,0]
        sigmaG=1.0
        
        #ikappa_arr = concatenate(map(gen_peaks_list,range(1,5)))
        idistance_rad_arr, idistance_rvir_arr, imass_arr, iredshift_arr = load(obsPK_dir+'halo_properties_lens.npy')

    kappa_edges = linspace(0.0, 0.024*5, 21)  
    kappa_centers = WLanalysis.edge2center(kappa_edges)
    N_halos = sum((kappa_halo<0.5),axis=1)+1
    include_halo = (kappa_halo<0.5).astype(int)
    include_halo[:,1:]=include_halo[:,:-1].copy()
    include_halo[:,0]=1

    f=figure(figsize=(10,8))
    jj=0
    labels=[r'$\log_{\rm 10} (M_{\rm halo}/M_{\odot})$',r'$\sigma_\chi\,{\rm [Mpc]}$',r'$\theta\, [{\rm arcmin}]$', r'$\theta / \theta_{vir} $']
    cc = 'orangered'
    for temp_arr in (log10(imass_arr), iredshift_arr, degrees(idistance_rad_arr)*60, idistance_rvir_arr):
        ax=f.add_subplot(2,2,jj+1)
        if jj==1:
            t0=temp_arr*include_halo
            #temp_avg = array([sum(unique(t0[_k])>0) for _k in arange(len(t0))])
            #temp_avg = temp_avg/N_halos.astype(float)
            temp_avg = array([std(DC(unique(t0[_k])[1:])) for _k in arange(len(t0))])
        else:
            temp_avg = sum(temp_arr*include_halo,axis=1)/N_halos
        N_mean, N_std = zeros((2,20))
        for i in range(20):
            k0,k1 = kappa_edges[i:i+2]
            itemp_avg=temp_avg[(ikappa_arr>k0) &(ikappa_arr<k1)]
            N_mean[i]=mean(itemp_avg)
            N_std[i]=std(itemp_avg)
        ax.errorbar(kappa_centers, N_mean, N_std,fmt='o',c=cc,ecolor=cc,mfc=cc, mec=cc,lw=1.5,capsize=0)
        
        ax.text(0.85, 0.85, r'${\rm (%s)}$'%(['a','b','c','d'][jj]),fontsize=20,color='k',fontweight='bold',transform=ax.transAxes)
        if jj==0:
            ax.set_ylim(10.9, 16.5)
        if jj==2 or jj==3:
            ax.set_ylim(0.1, 3.7)
        ax.set_ylabel(labels[jj],fontsize=20)#, labelpad=-10*int(jj==1))
        ax.tick_params(labelsize=16)
        ax.locator_params(tight=1, nbins=6)
        ax.set_xlim(kappa_edges[0]-0.002, kappa_edges[-1]+0.002)
        jj+=1
    plt.subplots_adjust(hspace=0.15,wspace=0.25, left=0.1, right=0.97,bottom=0.11,top=0.95)
    f.text(0.5, 0.03, r'$\kappa$', ha='center', va='center',fontsize=22)
    #show()
    #savefig(plot_dir+'halo_properties_lens0.png')
    savefig(plot_dir+'halo_properties_lens.pdf')
    close()      

if plot_concentration:
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    sigmaG = 1.0
    kappa_arr=array([])
    kappa15c_arr=array([])
    for Wx in range(1,5):
        imask = maskGen(Wx, sigmaG)
        ikmap_proj = kprojGen(Wx, sigmaG)
        ikmap_proj15c=load(obsPK_dir+'kappa_proj/kproj15c_W%i_sigmaG%02d.npy'%(Wx, sigmaG*10))
        kproj_peak_mat = WLanalysis.peaks_mat(ikmap_proj)
        idx = where((kproj_peak_mat>0)&(imask>0))
        kappa_arr = concatenate([kappa_arr,ikmap_proj[idx]])
        if plot_peaks_c15:
            kappa15c_arr = concatenate([kappa15c_arr,WLanalysis.peaks_list(ikmap_proj15c)])
        else:
            kappa15c_arr = concatenate([kappa15c_arr,ikmap_proj15c[idx]])
    
    k15c_mean, k15c_std = zeros((2,20))
    kappa_edges = linspace(0.00, 0.05,21)
    
    if not plot_peaks_c15:
        diff_arr = kappa15c_arr/kappa_arr-1#kappa15c_arr-kappa_arr#
        for i in range(20):
            k0,k1 = kappa_edges[i:i+2]
            ik15c = diff_arr[(kappa_arr>k0) &(kappa_arr<k1)]
            
            k15c_mean[i]=mean(ik15c)
            k15c_std[i]=std(ik15c)
        
    np.random.seed(399)
    kappa_centers=WLanalysis.edge2center(kappa_edges)
    cc,cc2=['orangered','green']#rand(2,3)
    f=figure(figsize=(6,4))
    ax=f.add_subplot(111)
    if not plot_peaks_c15:
        ax.errorbar(kappa_centers-2e-4, k15c_mean, k15c_std,fmt='o',c=cc,ecolor=cc,mfc=cc, mec=cc,lw=1.5,capsize=0)
        ax.set_ylabel(r"$\Delta\kappa/\kappa$",fontsize=22,labelpad=-5)#
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    else:
        ax.hist(kappa_arr, bins=kappa_edges,histtype='step',label='fiducial c')
        _temp1,_temp2,_temp3=ax.hist(kappa15c_arr, bins=kappa_edges,histtype='step',label='1.5*c')
        ax.set_ylabel(r"$N_{\rm peaks}$",fontsize=22,labelpad=-5)
        ax.set_yscale('log')
        ax.set_ylim(1e2,1e4)
    ax.tick_params(labelsize=16)
    ax.set_xlim(0.00499,0.0499)

    ax.set_xlabel(r'$\kappa$',fontsize=22)
    ax.legend(frameon=0,fontsize=12,loc=0)
    plt.subplots_adjust(hspace=0,wspace=0.05, left=0.15, right=0.97,bottom=0.16,top=0.95)
    
    show()
    #savefig(plot_dir+'concentration15c.pdf')
    #close()

if compare_peak_noise:
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2,1,height_ratios=[2.5,1])
    kappa_edges=linspace(-.04,.12,26)
    kappa_center0=WLanalysis.edge2center(kappa_edges)
    kappa_center=0.5*(kappa_center0[1::2]+kappa_center0[2::2])
###### below is operate on stampede
#mask_arr = [load('/home1/02977/jialiu/work/multiplicative/cfht_mask/Mask_W%i_0.7_sigmaG10.npy'%(Wx)) for Wx in range(1,5)]
#def peaks_gen(i):
    #print Wx,i
    #Ckmap = load('/work/02977/jialiu/kSZ/CFHT/Noise/W%i_Noise_sigmaG10_%04d.npy'%(Wx, i))
    #peak_hist = WLanalysis.peaks_mask_hist(Ckmap,mask_arr[Wx-1],bins=25)
    #return peak_hist
#for Wx in range(1,5):
    #all_peaks=map(peaks_gen,range(500))
    #save('/work/02977/jialiu/obsPK/peakse_noise_W%i.npy'%(Wx),all_peaks)
    
    #N_peaks=[WLanalysis.peaks_mask_hist(WLanalysis.readFits('/work/02977/jialiu/kSZ/CFHT/conv/W%i_KS_1.3_lo_sigmaG10.fit'%(Wx)), mask_arr[Wx-1], bins=25) for Wx in range(1,5)]
    #save('/work/02977/jialiu/obsPK/peakse_signal.npy',N_peaks)
    
    N_peak_noise0 = array(sum([load(obsPK_dir+'peaks/peakse_noise_W%i.npy'%(Wx)) for Wx in range(1,5)],axis=0))
    N_peak0 = sum(load(obsPK_dir+'peaks/peakse_signal.npy'),axis=0)
    N_peak_noise = N_peak_noise0[:,1::2]+N_peak_noise0[:,2::2]
    N_peak = N_peak0[1::2]+N_peak0[2::2]
    N_noise_mean = mean(N_peak_noise,axis=0)
    N_noise_std = std(N_peak_noise,axis=0)
    
    cc='green'
    f=figure(figsize=(6,6))
    #f=figure(figsize=(10,8))
    ax=f.add_subplot(gs[0])
    ax2=f.add_subplot(gs[1],sharex=ax)
    
    ax.errorbar(kappa_center,N_peak,N_noise_std,fmt='o',c=cc,mec=cc,linewidth=1.5, label=r'${\rm convergence}$',capsize=0)
    ax.plot(kappa_center,N_noise_mean,'k--', label=r'${\rm random}$')
    ax2.errorbar(kappa_center,N_peak-N_noise_mean,N_noise_std,fmt='o',c=cc,mec=cc,linewidth=1.5, capsize=0, label=r'${\rm convergence}$')
    ax2.plot(kappa_center, zeros(len(kappa_center)),'k--')
    ax2.set_xlim(kappa_center[0],kappa_center[-1])
    #ax2.set_ylim(-110,99)
    ax2.set_xlabel(r'$\kappa$',fontsize=20)
    ax.set_ylabel(r'$N_{\rm peaks}$',fontsize=20)
    ax2.set_ylabel(r'$\Delta N_{\rm peaks}$',fontsize=20,labelpad=-3)
    plt.setp(ax.get_xticklabels(), visible=False) 
    ax.legend(frameon=0,fontsize=16,loc=8)
    ax.tick_params(labelsize=16)
    ax2.tick_params(labelsize=16)
    ax2.locator_params('y',tight=True, nbins=5)
    ax2.locator_params('x',tight=True, nbins=6)
    plt.subplots_adjust(hspace=0.05,left=0.18, right=0.96,bottom=0.1,top=0.96)
    show()
    #savefig(plot_dir+'N_peaks.pdf')
    #savefig(plot_dir+'N_peaks.png')
    #close()
    
       
if plot_hilo_peaks:
    #kmap=kprojGen(1,1.0)
    kmap=klensGen(1,1.0)
    mask=maskGen(1,1.0)
    peak_mat = WLanalysis.peaks_mat(kmap)
    idx_hi=array(where((peak_mat>5*std(kmap[mask>0]))&(mask>0))).T
    idx_lo=array(where((peak_mat<0.02)&(mask>0))).T
    seed(14)#10,4
    pos_arr = [idx_hi[randint(len(idx_hi))], idx_lo[randint(len(idx_lo))], idx_lo[randint(len(idx_lo))], idx_lo[randint(len(idx_lo))]]
    f=figure(figsize=(12,3.5))
    istamp=6
    for jj in range(1,5):
        x,y=pos_arr[jj-1]
        ax=f.add_subplot(1,4,jj)
        ax.imshow(kmap[x-istamp:x+1+istamp, y-istamp:y+1+istamp],cmap='coolwarm',interpolation='nearest')
        ax.text(0.32, 1.05, r'$\kappa_{\rm peak}=%.3f$'%(kmap[x,y]), fontsize=12,color='k',fontweight='bold',transform=ax.transAxes)
        plt.setp(ax.get_xticklabels(), visible=False) 
        plt.setp(ax.get_yticklabels(), visible=False) 
    plt.subplots_adjust(hspace=0.02,wspace=0.04,left=0.02, right=0.98,bottom=0.02,top=0.97)
    #show()
    savefig(plot_dir+'sample_peaks.pdf')
    close()
    
if plot_yang2011_fig5:
    skn=0.036## sigma_kappa halos_noisy
    sknl=0.024## sigma_kappa noiseless
    halos_noisy = concatenate([load(obsPK_dir+'top20lens_halosidx_W%i_sigmaG10.npy'%(i))[:,0] for i in range(1,5)],axis=0)
    kappa_noisy = concatenate(map(gen_peaks_list,range(1,5)))
    
    halos_noiseless = concatenate([load(obsPK_dir+'top20_halos_W%i_sigmaG10.npy'%(i)) for i in range(1,6)],axis=0)
    kappa_noiseless = halos_noiseless[:,0]#-0.019
    
    N_halos_noisy = sum((halos_noisy[:,1:]<0.5),axis=1)+1
    N_halos_noiseless = sum((halos_noiseless[:,1:]<0.5),axis=1)+1
    N_halos_arr = [N_halos_noisy[kappa_noisy>=3.5*skn], 
                   N_halos_noisy[(kappa_noisy<3.5*skn)&(kappa_noisy>=skn)],
                   N_halos_noisy[kappa_noisy<skn],
                   N_halos_noiseless[kappa_noiseless>=3.5*sknl],
                   N_halos_noiseless[(kappa_noiseless<3.5*sknl)&(kappa_noiseless>=sknl)],
                   N_halos_noiseless[kappa_noiseless<sknl]]
    f=figure(figsize=(10,6))
    for jj in range(6):
        ax=f.add_subplot(2,3,jj+1)
        ax.hist(N_halos_arr[jj],bins=arange(0.5,21.5))
        if jj>2:
            ax.set_xlabel(r'$N_{\rm halo}$',fontsize=22)
        if not jj%3:
            ax.set_ylabel(r'$N_{\rm peaks}$',fontsize=22)
        ax.set_xlim(0,20)
        ax.text(0.3, 0.85, r'$\kappa_{\rm %s}\, ({\rm%s})}$'%(['lens','proj'][int(jj>2)],['high','medium','low'][jj%3]),fontsize=14,color='k',transform=ax.transAxes)#fontweight='bold'
    #plt.subplots_adjust(hspace=0,wspace=0.05, left=0.13, right=0.95,bottom=0.16,top=0.92)
    show()   

def find_SNR (CC_arr, errK_arr):
    weightK = 1/errK_arr**2/sum(1/errK_arr**2, axis=0)
    CC_mean = sum(CC_arr*weightK,axis=0)
    err_mean = sqrt(1.0/sum(1/errK_arr**2, axis=0))    
    return CC_mean, err_mean

if xcorr_peaks:
    fn = ['mapmap','peakmap','peakpeak']## type of cross correlation
    labels = [['\kappa','\kappa'],['\kappa','peak'],['peak','peak']]
    l1,l2=labels[peak_nob]
    #edgesGen = lambda Wx: logspace(0,log10(400),16)*sizes[Wx-1]/1330.0
    edgesGen = lambda Wx: linspace(1,250,11)*sizes[Wx-1]/1330.0
    ell_edges = edgesGen(1)*40
    delta_ell = ell_edges[1:]-ell_edges[:-1]
    sigmaG = 1.0
    if not peak_nob:
        sigmaG = 0.5
    f=figure(figsize=(6,4))
    seed(16)
    ax=f.add_subplot(111)
    SNR_sig=0
    SNR_b=0
    CC_arr = zeros(shape=(4,10))
    err_arr = zeros(shape=(4,10))
    CC_arrB = zeros(shape=(4,10))
    err_arrB = zeros(shape=(4,10))
    
    for Wx in range(1,5):#(1,):# 
        #cc=rand(3)
        iedge=edgesGen(Wx)
        imask = maskGen(Wx, sigmaG)
        fmask=sum(imask)/float(sizes[Wx-1]**2)
        sizedeg = (sizes[Wx-1]/512.0)**2*12.25
        fsky=fmask*sizedeg/41253.0
        print fsky
       
        #ikmap_lens = klensGen(Wx, sigmaG)*imask
        #ikmap_proj = kprojGen(Wx, sigmaG)*imask
        #ikmap_proj -= mean(ikmap_proj[imask>0])
        #ikmap_proj *= imask
        #ibmap = blensGen(Wx, sigmaG)*imask
        
        ell_arr = WLanalysis.edge2center(iedge)*360./sqrt(sizedeg)
        ########### maps
        #if peak_nob==0:
            #cc_proj_lens = WLanalysis.CrossCorrelate(ikmap_lens, ikmap_proj,edges=iedge)[1]/fmask
            #cc_proj_bmode = WLanalysis.CrossCorrelate(ikmap_proj,ibmap,edges=iedge)[1]/fmask
            #auto_lens = WLanalysis.CrossCorrelate(ikmap_lens,ikmap_lens,edges=iedge)[1]/fmask
            #auto_proj = WLanalysis.CrossCorrelate(ikmap_proj,ikmap_proj,edges=iedge)[1]/fmask
            #auto_bmode = WLanalysis.CrossCorrelate(ibmap,ibmap,edges=iedge)[1]/fmask

        ########### peaks
        #if peak_nob:#peakpeak or peakmap
            #ipeakmat_lens = WLanalysis.peaks_mat(ikmap_lens)
            #ipeakmat_proj = WLanalysis.peaks_mat(ikmap_proj)
            #ipeakmat_bmode = WLanalysis.peaks_mat(ibmap)
            #ipeakmat_lens[isnan(ipeakmat_lens)]=0
            #ipeakmat_proj[isnan(ipeakmat_proj)]=0
            #ipeakmat_bmode[isnan(ipeakmat_bmode)]=0
            #ipeakmat_bmode*=imask
            #ipeakmat_lens*=imask
            #if peak_nob == 1:#peak map
                #ipeakmat_proj = ikmap_proj
            #ipeakmat_proj*=imask
            #if clipped:
                #ipeakmat_bmode[ipeakmat_bmode>3.5*std(ikmap_lens[imask>0])]=0
                #ipeakmat_lens[ipeakmat_lens>3.5*std(ibmap[imask>0])]=0
       
            #cc_proj_lens = WLanalysis.CrossCorrelate(ipeakmat_lens, ipeakmat_proj,edges=iedge)[1]/fmask
            #cc_proj_bmode = WLanalysis.CrossCorrelate(ipeakmat_bmode,ipeakmat_proj,edges=iedge)[1]/fmask
            #auto_lens = WLanalysis.CrossCorrelate(ipeakmat_lens,ipeakmat_lens,edges=iedge)[1]/fmask
            #auto_proj = WLanalysis.CrossCorrelate(ipeakmat_proj, ipeakmat_proj,edges=iedge)[1]/fmask
            #auto_bmode = WLanalysis.CrossCorrelate(ipeakmat_bmode, ipeakmat_bmode,edges=iedge)[1]/fmask
            
        #save(obsPK_dir+'PS/PS_%s_W%i_sigmaG%02d%s.npy'%(fn[peak_nob], Wx, sigmaG*10, ['','_clipped'][clipped]),[ell_arr, cc_proj_lens, cc_proj_bmode, auto_lens, auto_proj, auto_bmode])
        ####################################
     
        ell_arr, cc_proj_lens, cc_proj_bmode, auto_lens, auto_proj,auto_bmode = load(obsPK_dir+'PS/PS_%s_W%i_sigmaG%02d%s.npy'%(fn[peak_nob], Wx, sigmaG*10,['','_clipped'][clipped]))
        
        delta_CC = sqrt((auto_lens*auto_proj+cc_proj_lens**2)/((2*ell_arr+1)*delta_ell*fsky))
        
        delta_CC_b = sqrt((auto_bmode*auto_proj+cc_proj_bmode**2)/((2*ell_arr+1)*delta_ell*fsky))
        
        SNR_sig+=sum( ((cc_proj_lens/delta_CC)**2) )
        SNR_b+=sum( ((cc_proj_bmode/delta_CC_b)**2) )
        #if Wx==1:
            
            #ax.errorbar(ell_arr+(Wx-2.5)*40, cc_proj_lens, delta_CC,fmt='o',c='orangered',mec='orangered',linewidth=1, capsize=0, label=r'${\rm %s}_{\rm proj}\times\, {\rm %s}_{\rm lens}$'%(l1,l2))   
            #ax.errorbar(ell_arr+(Wx-2.5)*40, cc_proj_bmode, delta_CC_b, fmt='kd',linewidth=1, capsize=0, mec='k', label=r'${\rm %s}_{\rm proj}\times \, {\rm %s}_{\rm B-mode}$'%(l1,l2))  
        #else:
            #ax.errorbar(ell_arr+(Wx-2.5)*60, cc_proj_lens, delta_CC,fmt='o',c='orangered',mec='orangered',linewidth=1, capsize=0) #ecolor=cc,mfc=cc, mec=cc,  
            #ax.errorbar(ell_arr+(Wx-2.5)*60, cc_proj_bmode, delta_CC_b, fmt='kd',linewidth=1, capsize=0, mec='k') # label=r'$\rm W%i$'%(Wx)
        
        CC_arr[Wx-1]=cc_proj_lens
        CC_arrB[Wx-1]=cc_proj_bmode
        err_arr[Wx-1]=delta_CC
        err_arrB[Wx-1]=delta_CC_b
        
    ########## add a bar for averaged points
    
    CC_avg, err_avg=find_SNR(CC_arr, err_arr)
    CC_avgB, err_avgB=find_SNR(CC_arrB, err_arrB)
    
    #ax.bar(ell_arr, 2*err_avg, bottom=CC_avg-err_avg, width=ones(10)*980, align='center',ec='r',fc='none',linewidth=1.5,alpha=1.0)
    #ax.bar(ell_arr, 2*err_avgB, bottom=CC_avgB-err_avgB, width=ones(10)*980, align='center',ec='k',fc='none',linewidth=1.5,alpha=1.0)
    
    ax.errorbar(ell_arr, CC_avg, err_avg,fmt='o',c='orangered',mec='orangered',linewidth=1.5, capsize=0, label=r'${\rm %s}_{\rm proj}\times\, {\rm %s}_{\rm lens}$'%(l1,l2))   
    
    ax.errorbar(ell_arr, CC_avgB, err_avgB, fmt='kd',linewidth=1.5, capsize=0, mec='k', label=r'${\rm %s}_{\rm proj}\times \, {\rm %s}_{\rm B-mode}$'%(l1,l2)) 
    
    ax.plot([0,1e4],[0,0],'k--')
    ax.legend(frameon=1,fontsize=14,loc=0)#'upper left')
    ax.set_ylabel(r'$\ell(\ell+1)/2\pi \times C_\ell$',fontsize=18)
    ax.set_xlabel(r'$\ell$',fontsize=18)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.rc('font', size=14)
    plt.subplots_adjust(hspace=0,wspace=0, left=0.15, right=0.94,bottom=0.16,top=0.92)
    ax.tick_params(labelsize=14)
    #ax.grid(True)
    #show()
    savefig(plot_dir+'CC%s_sigmaG%02d%s.pdf'%(fn[peak_nob],sigmaG*10,['','_clipped'][clipped]))
    close()
    #print 'SNR (%s, clipped=%i) = %.2f(lens) %.2f(bmode)'%(fn[peak_nob],clipped,sqrt(SNR_sig),sqrt(SNR_b))

if plot_xcorr_peaks:
    peak_nob = 2
    
    fn = ['mapmap','peakmap','peakpeak']## type of cross correlation
    labels = [['\kappa','\kappa'],['\kappa','peak'],['peak','peak']]
    l1,l2=labels[peak_nob]
    
    CC_arr = zeros(shape=(4,10))
    err_arr = zeros(shape=(4,10))
    CC_arrB = zeros(shape=(4,10))
    err_arrB = zeros(shape=(4,10))
    
    CC_arr_clipped = zeros(shape=(4,10))
    err_arr_clipped = zeros(shape=(4,10))
    CC_arrB_clipped = zeros(shape=(4,10))
    err_arrB_clipped = zeros(shape=(4,10)) 
    
    edgesGen = lambda Wx: linspace(1,250,11)*sizes[Wx-1]/1330.0
    ell_edges = edgesGen(1)*40
    delta_ell = ell_edges[1:]-ell_edges[:-1]
    sigmaG = 1.0
    
    
    f=figure(figsize=(6,4))
    ax=f.add_subplot(111)
    for Wx in range(1,5):#(1,):# 
        #cc=rand(3)
        iedge=edgesGen(Wx)
        imask = maskGen(Wx, sigmaG)
        fmask=sum(imask)/float(sizes[Wx-1]**2)
        sizedeg = (sizes[Wx-1]/512.0)**2*12.25
        fsky=fmask*sizedeg/41253.0
        ell_arr = WLanalysis.edge2center(iedge)*360./sqrt(sizedeg)
        
        clipped=0
        ell_arr, cc_proj_lens, cc_proj_bmode, auto_lens, auto_proj,auto_bmode = load(obsPK_dir+'PS/PS_%s_W%i_sigmaG%02d%s.npy'%(fn[peak_nob], Wx, sigmaG*10,['','_clipped'][clipped]))
        
        delta_CC = sqrt((auto_lens*auto_proj+cc_proj_lens**2)/((2*ell_arr+1)*delta_ell*fsky))
        
        delta_CC_b = sqrt((auto_bmode*auto_proj+cc_proj_bmode**2)/((2*ell_arr+1)*delta_ell*fsky))
        
        CC_arr[Wx-1]  = cc_proj_lens
        err_arr[Wx-1] = delta_CC
        CC_arrB[Wx-1] = cc_proj_bmode
        err_arrB[Wx-1]= delta_CC_b
        
        clipped=1
        ell_arr, cc_proj_lens, cc_proj_bmode, auto_lens, auto_proj,auto_bmode = load(obsPK_dir+'PS/PS_%s_W%i_sigmaG%02d%s.npy'%(fn[peak_nob], Wx, sigmaG*10,['','_clipped'][clipped]))
        
        delta_CC = sqrt((auto_lens*auto_proj+cc_proj_lens**2)/((2*ell_arr+1)*delta_ell*fsky))
        
        delta_CC_b = sqrt((auto_bmode*auto_proj+cc_proj_bmode**2)/((2*ell_arr+1)*delta_ell*fsky))
 
 
        CC_arr_clipped        = cc_proj_lens
        err_arr_clipped[Wx-1] = delta_CC
        CC_arrB_clipped[Wx-1] = cc_proj_bmode
        err_arrB_clipped[Wx-1]= delta_CC_b
    
    CC_avg, err_avg=find_SNR(CC_arr, err_arr)
    CC_avgB, err_avgB=find_SNR(CC_arrB, err_arrB)
    CC_avgclip, err_avgclip=find_SNR(CC_arr_clipped, err_arr_clipped)
    CC_avgBclip, err_avgBclip=find_SNR(CC_arrB_clipped, err_arrB_clipped)
        
    ax.errorbar(ell_arr, CC_avgB, err_avgB, fmt='kd',linewidth=1.5, capsize=0, mec='k')#, label=r'${\rm %s}_{\rm proj}\times \, {\rm %s}_{\rm B-mode}$'%(l1,l2)) 

    ax.errorbar(ell_arr+100, CC_avgBclip, err_avgBclip, fmt='kd',mfc='w',linewidth=1.5, capsize=0, mec='k')#, label=r'${\rm %s}_{\rm proj}\times \, {\rm %s}^{\rm lo}_{\rm B-mode}$'%(l1,l2)) 

    ax.errorbar(ell_arr, CC_avg, err_avg,fmt='o',c='orangered',mec='orangered',linewidth=1.5, capsize=0, label=r'${\rm %s}_{\rm proj}\times\, {\rm %s}_{\rm lens}$'%(l1,l2))   
    

    ax.errorbar(ell_arr+100, CC_avgclip, err_avgclip,fmt='o',c='orangered',mec='orangered',mfc='w',linewidth=1.5, capsize=0, label=r'${\rm %s}_{\rm proj}\times\, {\rm %s}^{\rm low}_{\rm lens}$'%(l1,l2))   
    
    
    ax.plot([0,1e4],[0,0],'k--')
    ax.legend(frameon=1,framealpha=1.0,fontsize=14,loc=0)#'upper left')
    ax.set_ylabel(r'$\ell(\ell+1)/2\pi \times C_\ell$',fontsize=18)
    ax.set_xlabel(r'$\ell$',fontsize=18)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.rc('font', size=14)
    plt.subplots_adjust(hspace=0,wspace=0, left=0.15, right=0.94,bottom=0.16,top=0.92)
    ax.tick_params(labelsize=14)
    ax.set_ylim(-3.6e-8, 8e-8)
    #ax.grid(True)
    #show()
    
    savefig(plot_dir+'CC%s_sigmaG%02d.pdf'%(fn[peak_nob],sigmaG*10))
    close()
        

    
if plot_peak_scatter:
    sigmaG=1.0
    mask_arr = [maskGen(Wx, sigmaG) for Wx in range(1,5)]
    #klens_arr =[klensGen(Wx, sigmaG) for Wx in range(1,5)]
    #kproj_arr =[kprojGen(Wx, sigmaG) for Wx in range(1,5)]
    
    klens_arr =[klensGen(Wx, sigmaG) - mean(klensGen(Wx, sigmaG)[mask_arr[Wx-1]>0]) for Wx in range(1,5)]
    kproj_arr = [kprojGen(Wx, sigmaG) - mean(kprojGen(Wx, sigmaG)[mask_arr[Wx-1]>0]) for Wx in range(1,5)]
    bmap_arr = [blensGen(Wx, sigmaG) - mean(blensGen(Wx, sigmaG)[mask_arr[Wx-1]>0]) for Wx in range(1,5)]
    def gen_peak_loc (ikmap, Wx):
        kmap_peak_mat = WLanalysis.peaks_mat(ikmap*mask_arr[Wx-1])
        return where((~isnan(kmap_peak_mat))&(mask_arr[Wx-1]>0))
    peaklens_lens_arr = concatenate([klens_arr[Wx-1][gen_peak_loc(klens_arr[Wx-1], Wx)] for Wx in range(1,5)]) 
    peaklens_proj_arr = concatenate([kproj_arr[Wx-1][gen_peak_loc(klens_arr[Wx-1], Wx)] for Wx in range(1,5)]) 
    peaklens_bmap_arr = concatenate([bmap_arr[Wx-1][gen_peak_loc(klens_arr[Wx-1], Wx)] for Wx in range(1,5)]) 
    peakproj_proj_arr = concatenate([kproj_arr[Wx-1][gen_peak_loc(kproj_arr[Wx-1], Wx)] for Wx in range(1,5)]) 
    peakproj_lens_arr = concatenate([klens_arr[Wx-1][gen_peak_loc(kproj_arr[Wx-1], Wx)] for Wx in range(1,5)])
    peakproj_bmap_arr = concatenate([bmap_arr[Wx-1][gen_peak_loc(kproj_arr[Wx-1], Wx)] for Wx in range(1,5)])
    
    f=figure(figsize=(10,4))    
    ax=f.add_subplot(121)
    NN,xx,yy,zz=ax.hist2d(peaklens_proj_arr,peaklens_lens_arr, bins=(20,24),range=((-0.02,0.15),(-0.05,0.15)),cmap='Greys')
    ybined=sum(NN*WLanalysis.edge2center(yy).reshape(1,-1),axis=1)/sum(NN,axis=1)
    ax.plot(WLanalysis.edge2center(xx),ybined)
    ax.set_ylabel(r'$\kappa_{\rm lens}$')
    ax.set_xlabel(r'$\kappa_{\rm proj}$')
    ax.set_title (r'${\rm peak}_{\rm lens}$')
    ax2=f.add_subplot(122)
    NN,xx,yy,zz=ax2.hist2d(peakproj_proj_arr, peakproj_lens_arr, bins=(20,24),range=((-0.02,0.15),(-0.05,0.15)),cmap='Greys')
    
    NNb,xxb,yyb=histogram2d(peakproj_proj_arr,peakproj_bmap_arr, bins=(20,24),range=((-0.02,0.15),(-0.05,0.15)))
    
    ybined=sum(NN*WLanalysis.edge2center(yy).reshape(1,-1),axis=1)/sum(NN,axis=1)
    ybinedB=sum(NNb*WLanalysis.edge2center(yy).reshape(1,-1),axis=1)/sum(NNb,axis=1)
    ax2.plot(WLanalysis.edge2center(xx),ybined,label='lens')
    ax2.plot(WLanalysis.edge2center(xx),ybinedB,'r',label='B-mode')
    ax2.legend(frameon=0,fontsize=12,loc=0)
    ax2.set_ylabel(r'$\kappa_{\rm lens}$')
    ax2.set_xlabel(r'$\kappa_{\rm proj}$')
    ax2.set_title (r'${\rm peak}_{\rm proj}$')
    ax.locator_params(tight=True, nbins=5)
    ax2.locator_params(tight=True, nbins=5)
    plt.subplots_adjust(hspace=0,wspace=0.25, left=0.12, right=0.95,bottom=0.15,top=0.9)
    show()

#if test_lo_hi_contour:
    #kappa_center=WLanalysis.edge2center(linspace(-.04,.12,26))
    #test_dir='/Users/jia/Documents/weaklensing/CFHTLenS/emulator/test_ps_bug/'
    #pk_avg = load(test_dir+'ALL_pk_avg_sigmaG10.npy')
    #pk_fidu = load(test_dir+'SIM_peaks_sigma18_emu1-512b240_Om0.305_Ol0.695_w-0.879_ns0.960_si0.765_025bins_ALL.npy')
    #pk_CFHT = load(test_dir+'ALL_pk_CFHT_sigmaG10.npy')
    #cosmo_params = genfromtxt(test_dir+'cosmo_params.txt')
    #im, iw, s = cosmo_params.T
    
    #w_arr = linspace(0,-3, 10)
    #om_arr = linspace(0, 1.2, 50)
    #si8_arr = linspace(0, 1.6, 51)
    
    #cov_inv = mat(cov(pk_fidu,rowvar=0)).I
    #cov_inv_hi=mat(cov(pk_fidu[:,22:],rowvar=0)).I
    #cov_inv_lo=mat(cov(pk_fidu[:,:22],rowvar=0)).I
    
