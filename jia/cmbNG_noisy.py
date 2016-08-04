import WLanalysis
import glob, os, sys
import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack
from scipy import interpolate
import random
import pickle
from pylab import *

######## knobs ########
filtered = 0
compute_noisy_stats = 0
load_noiseless_stats,optimizeit = 0,0
load_nooisy_stats, sigmaG = 1, 8.0
compute_noisy_contour = 0
fsky_deg=2e4#1000.0
compute_interp = 0
make_noisy_maps = 0

#### stats77 comes from the fact noisy maps are 77 x 77 in size
#### constants
CMBlensing_dir ='/Users/jia/weaklensing/CMBnonGaussian/'
bins=25
all_points = genfromtxt(CMBlensing_dir+'model_point.txt')
idx46=where(all_points.T[0]>0.14)
all_points46 = all_points[idx46]
cosmo_arr = array(['Om%.3f_Ol%.3f_w-1.000_si%.3f'%(cosmo[0],1-cosmo[0], cosmo[1]) for cosmo in all_points])
cosmo_noisy_arr0 = os.listdir(CMBlensing_dir+'colin_noisy')#[3:-1]
cosmo_noisy_arr0 =[cosmo[10:] for cosmo in cosmo_noisy_arr0]
idx_noisy = where([cosmo in cosmo_noisy_arr0 for cosmo in cosmo_arr])[0]
cosmo_noisy_arr = cosmo_arr[idx_noisy]
cosmo_params_noisy = all_points[idx_noisy]
sigmaG_arr = array([0.5, 1.0, 2.0, 5.0, 8.0])
PDFbins = linspace(-0.12, 0.12, 101)
peak_bins = linspace(-0.06,0.14,26)#linspace(-3*0.02,6*0.02,26)
sizedeg = 3.5**2
#ALratio=genfromtxt('ALratio.txt')
ALratio_mat = ifftshift(load(CMBlensing_dir+'ALratio_mat.npy'))
filter_mat =  ifftshift(load(CMBlensing_dir+'filter_mat.npy'))
idx_not_nan = delete(range(25),(1,3,6))#where(~isnan(ps_all77[0,0]))[0]    
ell_centers77 = WLanalysis.PowerSpectrum(rand(77,77), bins=bins)[0]


ell_arr2048=WLanalysis.PowerSpectrum(zeros(shape=(2048,2048)), bins=50)[0]
ell_arr77 = WLanalysis.PowerSpectrum(zeros(shape=(77,77)), bins=bins)[0]

######### functions

FTmapGen_fidu = lambda r: pickle.load(open(CMBlensing_dir+'colin_noisy/kappaMapTT_10000sims/kappaMap%03dTT_3.pkl'%(r)))
FTmapGen_Gaus = lambda r: pickle.load(open(CMBlensing_dir+'colin_noisy/kappaMapTT_Gauss_10000sims/kappaMap%03dTT_3.pkl'%(r)))
FTmapGen = lambda cosmo, r: pickle.load(open(CMBlensing_dir+'colin_noisy/reconMaps_%s/kappaMap%04dTT_3.pkl'%(cosmo, r)))#cosmo=Om0.406_Ol0.594_w-1.000_si0.847

def FT_PowerSpectrum (cosmo, r, bins=10, return_ell_arr=0, Gaus=0):
    if Gaus:
        a = FTmapGen_Gaus(r)
    else:
        a = FTmapGen(cosmo, r)
    PS2D=np.abs(fftpack.fftshift(a))**2
    ell_arr,psd1D=WLanalysis.azimuthalAverage(PS2D, bins=bins)    
    if return_ell_arr:
        ell_arr = WLanalysis.edge2center(ell_arr)* 360./3.5
        return ell_arr
    else:
        return psd1D
    
def FT2real (cosmo, r, Gaus=0):    
    if Gaus:
        a = FTmapGen_Gaus(r)
    else:
        a = FTmapGen(cosmo, r)
    a/=ALratio_mat##### to correct the difference between Colin and Vanessa
    if filtered:
        a*=filter_mat
    areal = real(fftpack.ifft2(a))
    inorm = (2*pi*3.5/360.0)/(77.0**2)
    areal /= inorm
    return areal
    
def PDFGen (kmap, PDF_bins):
    all_kappa = kmap[~isnan(kmap)]
    PDF = histogram(all_kappa, bins=PDF_bins)[0]
    PDF_normed = PDF/float(len(all_kappa))
    return PDF_normed

def peaksGen (kmap, peak_bins):
    peaks = WLanalysis.peaks_list(kmap)
    peaks_hist = histogram(peaks, bins=peak_bins)[0]
    return peaks_hist

def compute_GRF_PDF_ps_pk (cosmo, r, Gaus=0,sigmaG=8.0):
    kmap = FT2real(cosmo, r, Gaus=Gaus)
    ps = WLanalysis.PowerSpectrum(WLanalysis.smooth(kmap, 0.18), bins=bins)[1]#*2.0*pi/ell_arr**2
    if not filtered:
        kmap = WLanalysis.smooth(kmap, 2.93*sigmaG/8.0)
    PDF = PDFGen(kmap, PDFbins)
    peaks = peaksGen(kmap, peak_bins)
    return concatenate([ps, PDF, peaks])


if load_noiseless_stats:
 
    fidu_stats = load(CMBlensing_dir+'Pkappa_gadget/noiseless/kappa_Om0.296_Ol0.704_w-1.000_si0.786_ps_PDF_pk_z1100_10240.npy')
    ps_fidu_noiseless = array([fidu_stats[j][0] for j in range(1024,10240)]).squeeze() 
    idx_cut2000 = where( (ell_arr2048<2100) & (~isnan(ps_fidu_noiseless[0])))[0]
    
    ps_fidu_noiseless = ps_fidu_noiseless[:,idx_cut2000]
    PDF_fidu_noiseless = array([fidu_stats[j][1] for j in range(1024,10240)])
    peaks_fidu_noiseless = array([fidu_stats[j][2] for j in range(1024,10240)])
    
    #### noiseless TT
    all_stats46 = array([ load(CMBlensing_dir+'Pkappa_gadget/noiseless/kappa_%s_ps_PDF_pk_z1100.npy'%(cosmo))    for cosmo in cosmo_arr[idx46]])
    
    ps_noiseless46 = mean(array([ [all_stats46[icosmo][j][0] for icosmo in range(len(all_stats46))] for j in range(1000)])[:,:,idx_cut2000],axis=0)
    
    PDF_noiseless46 = mean(array([ [all_stats46[icosmo][j][1] for icosmo in range(len(all_stats46))] for j in range(1000)]),axis=0)
    
    peaks_noiseless46 = mean(array([ [all_stats46[icosmo][j][2] for icosmo in range(len(all_stats46))] for j in range(1000)]),axis=0)

    
    #### noiseless GRF
    #mat_GRF=load(CMBlensing_dir+'plot/PDF_pk_600b_GRF.npy')
    #iPDF_GRF = array([mat_GRF[x][0][4] for x in range(1024)])
    #ipeak_GRF = array([mat_GRF[x][1][4] for x in range(1024)])

### create stats for noisy TT
if compute_noisy_stats:
    for cosmo in cosmo_noisy_arr:
        if filtered:
            all_stats77 = array([compute_GRF_PDF_ps_pk(cosmo,r,Gaus=0) for r in range(1000)])#1024 
            save(CMBlensing_dir+'Pkappa_gadget/noisy/filtered_noisy_z1100_stats77_kappa_%s.npy'%(cosmo), all_stats77)
        else:    
            for sigmaG in [1.0, 5.0, 8.0]:
                print cosmo
                #cosmo='Om0.394_Ol0.606_w-1.000_si0.776'#cosmo_noisy_arr[1]
                all_stats77 = array([compute_GRF_PDF_ps_pk(cosmo,r,Gaus=0,sigmaG=sigmaG) for r in range(1000)])#1024 
                save(CMBlensing_dir+'Pkappa_gadget/noisy/%snoisy_z1100_stats77_kappa_%s_sigmaG%02d.npy'%(['','filtered_'][filtered],cosmo,sigmaG*10), all_stats77)
    #Gaus=0
    #morebins = 0
    #if morebins:
        #PDFbins = linspace(-0.24, 0.24, 201)#(-0.12, 0.12, 101)
        #peak_bins = linspace(-0.1,0.18,36)#(-0.06,0.14,26)
    #def compute_GRF_PDF_ps_pk_fidu (r,Gaus=Gaus,filtered=filtered):
        #if Gaus:
            #a = FTmapGen_Gaus(r)
        #else:
            #a = FTmapGen_fidu(r)
        #if filtered:
            ##print 'filtered'
            #a*=filter_mat
        #areal = real(fftpack.ifft2(a))
        #inorm = (2*pi*3.5/360.0)/(77.0**2)
        #areal /= inorm    
        #kmap = areal
        #ps = WLanalysis.PowerSpectrum(WLanalysis.smooth(kmap, 0.18), bins=bins)[1]#*2.0*pi/ell_arr**2
        #if not filtered:
            #kmapsmooth8 = WLanalysis.smooth(kmap, 2.93)
        #else:
            #kmapsmooth8 = areal
        #PDF = PDFGen(kmapsmooth8, PDFbins)
        #peaks = peaksGen(kmapsmooth8, peak_bins)
        #return concatenate([ps, PDF, peaks])
    #fidu_stats77 = array([compute_GRF_PDF_ps_pk_fidu(r,Gaus=Gaus) for r in range(10000)])
    #save(CMBlensing_dir+'Pkappa_gadget/noisy/%snoisy_z1100_stats77_fidu_%s%s.npy'%(['','filtered_'][filtered],['kappa','gaus'][int(Gaus)], ['','_morebins'][morebins]), fidu_stats77)
    
if load_nooisy_stats:
    ######## filtered
    all_stats77_filtered = array([load (CMBlensing_dir+'Pkappa_gadget/noisy/filtered_noisy_z1100_stats77_kappa_%s.npy'%(cosmo))[:1000,:] for cosmo in cosmo_noisy_arr])
    
    all_stats77 = array([load (CMBlensing_dir+'Pkappa_gadget/noisy/noisy_z1100_stats77_kappa_%s_sigmaG%02d.npy'%(cosmo, sigmaG*10))[:1000,:] for cosmo in cosmo_noisy_arr])
    
    all_stats77_filtered_mean = mean(all_stats77_filtered,axis=1)
    all_stats77_mean = mean(all_stats77,axis=1)

    ps_all77 = all_stats77_mean[:,idx_not_nan]#
    PDF_all77 = all_stats77_mean[:,bins:bins+len(PDFbins)-1]
    peaks_all77 = all_stats77_mean[:,bins+len(PDFbins)-1:]
    
    ps_fidu = all_stats77[1,:][:,idx_not_nan]
    PDF_fidu = all_stats77[1,:,bins:bins+len(PDFbins)-1]
    peaks_fidu = all_stats77[1,:,bins+len(PDFbins)-1:]
    
    if filtered:
        PDF_all77 = all_stats77_filtered_mean[:,bins:bins+len(PDFbins)-1]
        PDF_fidu = all_stats77_filtered[1,:,bins:bins+len(PDFbins)-1]
        
        if not optimizeit:
            peaks_all77 = all_stats77_filtered_mean[:,bins+len(PDFbins)-1:]
            peaks_fidu = all_stats77_filtered[1,:,bins+len(PDFbins)-1:]

    PDF_fidu=sum(PDF_fidu.reshape(PDF_fidu.shape[0],50,-1),axis=-1)
    PDF_all77=sum(PDF_all77.reshape(PDF_all77.shape[0],50,-1),axis=-1)

    idx_peaks=where(mean(peaks_fidu,axis=0)>1e-6)[0]
    peaks_fidu=peaks_fidu[:,idx_peaks]
    peaks_all77=peaks_all77[:,idx_peaks]
    
    idx_PDF=where(mean(PDF_fidu,axis=0)>1e-4)[0]
    PDF_fidu=PDF_fidu[:,idx_PDF]
    PDF_all77=PDF_all77[:,idx_PDF]
        
        
############ contour
def plane_gen (fidu_mat, ips_avg, obs_arr, cosmo_params, om_arr, si8_arr, method='clough'):
    
    interp_cosmo = WLanalysis.buildInterpolator2D(ips_avg, cosmo_params, method=method)
    cov_mat = cov(fidu_mat,rowvar=0)/(fsky_deg/12.5)# (2e4/12.5)#area factor for AdvACT
    cov_inv = mat(cov_mat).I

    def chisq_fcn(param1, param2):
        model = interp_cosmo((param1,param2))
        del_N = np.mat(model - obs_arr)
        chisq = float(del_N*cov_inv*del_N.T)
        return chisq
    prob_plane = WLanalysis.prob_plane(chisq_fcn, om_arr, si8_arr)[1]
    return prob_plane

if compute_noisy_contour:
    noise = 'noisy'#'noiseless'#
    om_fidu, si8_fidu=all_points[18]
    del_om, del_si8 = 0.05,0.05#0.15, 0.15
    om0,om1,si80,si81=om_fidu-del_om, om_fidu+del_om, si8_fidu-del_si8, si8_fidu+del_si8
    jjj=100
    om_arr= linspace(om0,om1,jjj)
    si8_arr=linspace(si80,si81, jjj+1)

    for ismooth in (-1,):#(1,3,4):
    #-1 is for noisy maps
    #1-4 for 5 smoothing scales
        for jj in (4,):#range(1,4):#
            istat=['ps','PDF','peaks','comb','pkPDF'][jj]
            for imethod in ('clough',):#('linear','clough','Rbf'):#
                                
                if noise == 'noisy': 
                    print sigmaG,istat, imethod
                    ips_fidu = [ps_fidu, PDF_fidu, peaks_fidu, concatenate([ps_fidu, PDF_fidu, peaks_fidu],axis=1),
                                concatenate([PDF_fidu, peaks_fidu],axis=1)][jj]
                    ips = [ps_all77, PDF_all77, peaks_all77, concatenate([ps_all77, PDF_all77, peaks_all77],axis=1), 
                           concatenate([PDF_all77, peaks_all77],axis=1)][jj]
                    obs_arr=ips[1]
                    #print ips_fidu.shape, obs_arr.shape, ips.shape
                    prob_plane = plane_gen(ips_fidu, ips, obs_arr, cosmo_params_noisy, om_arr, si8_arr,method=imethod)
                elif noise == 'noiseless':
                    print sigmaG_arr[ismooth],istat, imethod
                    sigmaG=sigmaG_arr[ismooth]
                    ips_fidu = [ps_fidu_noiseless, 
                                PDF_fidu_noiseless[:,ismooth,:], 
                                peaks_fidu_noiseless[:,ismooth,:], concatenate([ps_fidu_noiseless, PDF_fidu_noiseless[:,ismooth,:], peaks_fidu_noiseless[:,ismooth,:]],axis=1)][jj]
                    ips = [ps_noiseless46, PDF_noiseless46[:,ismooth,:],
                        peaks_noiseless46[:,ismooth,:], concatenate([ps_noiseless46, PDF_noiseless46[:,ismooth,:],
                        peaks_noiseless46[:,ismooth,:]],axis=1)][jj]
                    obs_arr=ips[16]
                    if istat=='ps' and ismooth!=1:
                        continue
                    prob_plane = plane_gen(ips_fidu, ips, obs_arr, all_points46, om_arr, si8_arr,method=imethod)
                
                if filtered:
                    save(CMBlensing_dir+'mat/%sfiltered_Prob_fsky%i_%s_%s_%s_del%s.npy'%(['','optimize_'][optimizeit],fsky_deg, noise,istat, imethod, del_om),prob_plane)
                else:
                    save(CMBlensing_dir+'mat/Prob_fsky%i_%s_%s_%s_sigmaG%02d_del%s.npy'%(fsky_deg, noise,istat, imethod, sigmaG*10, del_om),prob_plane)
                
                imshow(prob_plane,origin='lower',interpolation='nearest',extent=[si80,si81,om0,om1]);show()
                #title('%sProb_fsky%i_%s_%s_%s_sigmaG%02d'%(['','filtered_'][filtered], fsky_deg, noise,istat, imethod, sigmaG_arr[ismooth]*10),fontsize=10)
                #savefig(CMBlensing_dir+'plot/optimize_%sProb_fsky%i_%s_%s_%s_sigmaG%02d.png'%(['','filtered_'][filtered], fsky_deg, noise,istat, imethod, sigmaG_arr[ismooth]*10))
                #close()
                


if compute_interp:
    ismooth=3
    om_fidu, si8_fidu=all_points[18]
    ips = concatenate([ps_noiseless46, PDF_noiseless46[:,ismooth,:],
                        peaks_noiseless46[:,ismooth,:]],axis=1)
    ips_fidu = concatenate([ps_fidu_noiseless, PDF_fidu_noiseless[:,ismooth,:], peaks_fidu_noiseless[:,ismooth,:]],axis=1)
    obs_arr=ips[1]
    
    interp_cosmo = WLanalysis.buildInterpolator2D(delete(ips,16,axis=0), delete(all_points46,16,axis=0), method='clough')
    
    save(CMBlensing_dir+'interp.npy',[mean(ips_fidu,axis=0),std(ips_fidu,axis=0),interp_cosmo(all_points46[16])])
    

if make_noisy_maps:
    #for cosmo in cosmo_noisy_arr:
        #print cosmo
        #[save(CMBlensing_dir+'colin_noisy/maps4andrea/reconMaps_%s/recon_%s%s_r%04d.npy'%(cosmo,['','filtered_'][filtered],cosmo, r),FT2real(cosmo, r, Gaus=0)) for r in range(1000)]
    def FT2real_fidu(r, Gaus=0):
        if Gaus:
            a = FTmapGen_Gaus(r)
        else:
            a = FTmapGen_fidu(r)
        if filtered:
            #print 'filtered'
            a*=filter_mat
        areal = real(fftpack.ifft2(a))
        inorm = (2*pi*3.5/360.0)/(77.0**2)
        areal /= inorm    
        return areal
    
    cosmo=cosmo_noisy_arr[1]
    ifilter = ['','_filtered'][filtered]
    [save(CMBlensing_dir+'colin_noisy/maps4andrea/fidu_kappa%s/fidukappa%s_%s_r%04d.npy'%(ifilter, ifilter, cosmo, r),FT2real_fidu(r, Gaus=0)) for r in range(10000)]
    print 'boo'
    [save(CMBlensing_dir+'colin_noisy/maps4andrea/fidu_gauss%s/fidugauss%s_%s_r%04d.npy'%(ifilter, ifilter, cosmo, r),FT2real_fidu(r, Gaus=1)) for r in range(10000)]
