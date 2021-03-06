import WLanalysis
import glob, os, sys
import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack

from scipy import interpolate
import random
from pylab import *

ends = [0.5, 0.22, 0.18, 0.1, 0.08]
PDFbin_arr = [linspace(-end, end, 101) for end in ends]
kmap_stds = [0.06, 0.05, 0.04, 0.03, 0.02] #[0.014, 0.011, 0.009, 0.006, 0.005]
peak_bins_arr = [linspace(-3*istd, 6*istd, 26) for istd in kmap_stds]

sizedeg = 3.5**2
PPA = 2048.0/(sqrt(sizedeg)*60.0) #pixels per arcmin
sigmaG_arr = array([0.5, 1.0, 2.0, 5.0, 8.0])
sigmaP_arr = sigmaG_arr*PPA #smoothing scale in pixels

CMBlensing_dir = '/work/02977/jialiu/CMBnonGaussian/'
CMBNG_dir ='/Users/jia/weaklensing/CMBnonGaussian/'
CMBlensing_dir = CMBNG_dir
cosmo_arr = genfromtxt('/Users/jia/weaklensing/CMBnonGaussian/cosmo_arr.txt',dtype='string')
fidu_cosmo=cosmo_arr[12]

######### official plot konbs ###########
plot_design = 0
plot_comp_nicaea = 0
plot_noiseless_peaks_PDF = 0
plot_sample_noiseless_noisy_map = 0
plot_noisy_peaks_PDF, filtered = 0, 1
plot_reconstruction_noise = 0
plot_corr_mat, do_noisy = 1, 0
plot_contour_PDF_pk, area_scaling, fsky_deg = 0, 0, 1000.0
plot_contour_noisy_old, area_scaling_noisy, fsky_deg_noisy = 0, 0, 1000.0
plot_contour_theory = 0
plot_contour_PDF_pk_noisy = 0
plot_contour_comb = 0
plot_Cell_om_si = 0
plot_interp = 0
plot_skewness = 0

if plot_design:
    all_points = genfromtxt(CMBlensing_dir+'model_point.txt')
    idx46=where(all_points.T[0]>0.14)
    all_points46 = all_points[idx46]
    cosmo_arr = array(['Om%.3f_Ol%.3f_w-1.000_si%.3f'%(cosmo[0],1-cosmo[0], cosmo[1]) for cosmo in all_points])
    cosmo_noisy_arr0 = os.listdir(CMBlensing_dir+'colin_noisy')#[3:-1]
    cosmo_noisy_arr0 =[cosmo[10:] for cosmo in cosmo_noisy_arr0]
    idx_noisy = where([cosmo in cosmo_noisy_arr0 for cosmo in cosmo_arr])[0]
    cosmo_noisy_arr = cosmo_arr[idx_noisy]
    cosmo_params_noisy = all_points[idx_noisy]
    om, si8=all_points46.T
    om0,si80=all_points46[16]
    
    f=figure(figsize=(6,6))
    ax=f.add_subplot(111)
    ax.scatter(om,si8,marker='D',color='k')
    ax.scatter(om0,si80, s=300, marker='o',color='orangered', facecolors='none', edgecolors='r',lw=2)
    for iom, isi8 in cosmo_params_noisy:
        if not iom==om0:
            ax.scatter(iom,isi8, s=300, marker='o',color='deepskyblue', facecolors='none', edgecolors='deepskyblue',lw=2)
    ax.set_xlabel(r'$\Omega_m$',fontsize=22)
    ax.set_ylabel(r'$\sigma_8$',fontsize=22)
    ax.set_xlim(0.12,0.75)
    ax.set_ylim(0.46,1.05)
    ax.grid(True)
    #show()
    savefig(CMBNG_dir+'plot_official/plot_design.pdf')
    savefig(CMBNG_dir+'plot/plot_design.png')
    close()
    
if plot_comp_nicaea:
    cosmo='Om0.296_Ol0.704_w-1.000_si0.786'
    ell_gadget = (WLanalysis.edge2center(logspace(log10(1.0),log10(1024),51))*360./sqrt(12.25))[:34]

    ell_nicaea, ps_nicaea=genfromtxt('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_nicaea/Pkappa_nicaea25_{0}_1100'.format(cosmo))[33:-5].T
    
    ell_nicaea2, ps_nicaea_linear=genfromtxt('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_nicaea/Pkappa_nicaea25_{0}_1100_linear'.format(cosmo))[33:-5].T
    
    def get_1024(j):
        pspkPDFgadget=load('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_gadget/noiseless/kappa_{0}_ps_PDF_pk_z1100_{1}.npy'.format(fidu_cosmo, j))
        ps_gadget=array([pspkPDFgadget[i][0] for i in range(len(pspkPDFgadget))])
        ps_gadget=ps_gadget.squeeze()
        return ps_gadget
    ps_gadget=concatenate(map(get_1024, range(10)),axis=0)[:,:34]
    
    idx=where(~isnan(mean(ps_gadget,axis=0)))[0]
    ps_gadget=ps_gadget[:,idx]
    ell_gadget=ell_gadget[idx]
    
    f=figure(figsize=(8,6))
    ax=f.add_subplot(111)
    ax.errorbar(ell_gadget, mean(ps_gadget,axis=0),std(ps_gadget,axis=0),label=r'$\rm{simulation}$',lw=2.0,color='k',capsize=0)
    ax.plot(ell_nicaea, ps_nicaea,ls='-',lw=3,color='deepskyblue',label=r'$\rm{Smith03+Takahashi12}$')
    ax.plot(ell_nicaea2, ps_nicaea_linear,ls='--',color='orangered',lw=4,label=r'$\rm{linear}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.set_title(cosmo)
    ax.set_xlabel(r'$\ell$',fontsize=22)
    ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}/2\pi$',fontsize=22)
    ax.set_xlim(100,3e3)
    leg=ax.legend(loc=2,fontsize=20,ncol=1)
    leg.get_frame().set_visible(False)
    ax.set_ylim(6e-5,1e-2)
    #ax.set_xlim(6e-5,1e-2)
    ax.tick_params(labelsize=16)
    plt.tight_layout()
    #show()
    savefig(CMBNG_dir+'plot_official/plot_theory_comparison.pdf')
    #savefig(CMBNG_dir+'plot/plot_theory_comparison.jpg')
    close()
    

if plot_noiseless_peaks_PDF:
    mat_kappa=load('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_gadget/noiseless/kappa_{0}_ps_PDF_pk_z1100.npy'.format(fidu_cosmo))
    mat_GRF=load('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_gadget/noiseless/GRF_{0}_ps_PDF_pk_z1100.npy'.format(fidu_cosmo))
    N=len(mat_kappa)
    
    ############ plot peaks and PDF ##########
    #for j in (1,2):## 1 is PDF, 2 is kappa
        #f=figure(figsize=(6,10))
        #f.text(0.02, 0.5, [r'$\rm{PDF}$', r'$\rm{N_{peaks}\, (deg^{-2})}$'][j-1], va='center', rotation='vertical',fontsize=22)
        
        #for i in arange(len(sigmaG_arr)):
            #ax=f.add_subplot(5, 1, i+1)
            
            #iPDF_kappa = array([mat_kappa[x][j][i] for x in range(N)])
            #iPDF_GRF = array([mat_GRF[x][j][i] for x in range(N)])
            
            #x_arr = [PDFbin_arr, peak_bins_arr][j-1]
            #if j==1:#PDF, normalize by dk
                #iPDF_kappa = iPDF_kappa/(x_arr[i][1]-x_arr[i][0])
                #iPDF_GRF = iPDF_GRF/(x_arr[i][1]-x_arr[i][0])
            #elif j==2:#peak, normalized by deg^2
                #iPDF_kappa = iPDF_kappa/12.25
                #iPDF_GRF = iPDF_GRF/12.25
            #plot(x_arr[i][1:], mean(iPDF_kappa, axis=0),'k-',lw=1,label=r'$\kappa\,\rm{maps}$')
            #plot(x_arr[i][1:], mean(iPDF_GRF, axis=0),'k--',lw=2,label=r'$\rm{GRF}$')
            ##ax.errorbar(x_arr[i][1:], mean(iPDF_kappa, axis=0), std(iPDF_kappa, axis=0)/sqrt(3e4/12), label='kappa',capsize=0,)
            ##ax.errorbar(x_arr[i][1:], mean(iPDF_GRF, axis=0), std(iPDF_GRF, axis=0)/sqrt(3e4/12), fmt='--',label='GRF',capsize=0,)

            #ax.locator_params(axis = 'x', nbins = 6)
            #ax.locator_params(axis = 'y', nbins = 2)
            #ax.annotate(r"$\theta_G=%.1f'$"%(sigmaG_arr[i]), xy=(0.025, 0.8), xycoords='axes fraction',fontsize=16)
            #ax.set_yscale('log')
            #meanPDF = mean(iPDF_kappa, axis=0)
            #if j==2:#peaks
                #ax.set_ylim(amin(meanPDF), amax(meanPDF)*2)
            #else:
                #ax.set_ylim(amax([amin(meanPDF), 1e-6])*1.5, amax(meanPDF)*2.5)
                #x0=abs(x_arr[i][1:][where(meanPDF==WLanalysis.findlevel(meanPDF)[-1])[0]])
                #if i==0:
                    #x0=2.0*x0
                    #ax.set_yticks([1e-5,1e-3,1e-1])
                #else:
                    #x0=1.5*x0
                #ax.set_xlim(-x0,x0)

            #if i ==0:
                    #leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=8)
                    #leg.get_frame().set_visible(False)
            #if i == 4:
                    #ax.set_xlabel('$\kappa$',fontsize=22)
            #ax.tick_params(labelsize=16)
        #plt.tight_layout()
        #plt.subplots_adjust(hspace=0.2,wspace=0, left=0.18, right=0.95)
        ##show()
        ##savefig(CMBNG_dir+'plot/plot_noiseless_%s.jpg'%(['PDF','peaks'][j-1]))
        #savefig(CMBNG_dir+'plot_official/plot_noiseless_%s.pdf'%(['PDF','peaks'][j-1]))
        #close()

    ############# plot frac diff ##########
    for j in (1,2):## 1 is PDF, 2 is kappa
        f=figure(figsize=(6,10))
        f.text(0.02, 0.5, [r'$\rm{PDF^{\kappa}/PDF^{GRF}}-1$', r'$\rm{N_{peaks}^\kappa / N_{peaks}^{GRF}}-1$'][j-1], va='center', rotation='vertical',fontsize=22)
        
        for i in arange(len(sigmaG_arr)):
            ax=f.add_subplot(5, 1, i+1)
            iPDF_kappa = array([mat_kappa[x][j][i] for x in range(N)])
            iPDF_GRF = array([mat_GRF[x][j][i] for x in range(N)])
            x_arr = [PDFbin_arr, peak_bins_arr][j-1]
            
            #plot(x_arr[i][1:], mean(iPDF_kappa, axis=0),'k-',lw=1,label=r'$\kappa\,\rm{maps}$')
            plot([-0.5,0.5],[0,0],'k--',lw=2)
            ax.errorbar(x_arr[i][1:], mean(iPDF_kappa, axis=0)/mean(iPDF_GRF, axis=0)-1, std(iPDF_kappa, axis=0)/sqrt(2e4/12)/mean(iPDF_GRF, axis=0), capsize=0,color='k')

            ax.locator_params(axis = 'x', nbins = 6)
            locator_params(axis = 'y', nbins = 5)
            ax.annotate(r"$\theta_G=%.1f'$"%(sigmaG_arr[i]), xy=(0.025, 0.8), xycoords='axes fraction',fontsize=16)
            meanPDF = mean(iPDF_kappa, axis=0)
            if j==2:#peaks
                ax.set_ylim(amin(meanPDF), amax(meanPDF)*2)
                idx=where(meanPDF>0)[0]
                #ax.set_xlim(x_arr[i][1:][amin(idx)], x_arr[i][1:][amax(idx)])
                ax.set_xlim(x_arr[i][1:][0],x_arr[i][1:][-1])
            else:
                #ax.set_ylim(amax([amin(meanPDF), 1e-6])*1.5, amax(meanPDF)*2.5)
                x0=1.5*abs(x_arr[i][1:][where(meanPDF==WLanalysis.findlevel(meanPDF)[-1])[0]])
                ax.set_xlim(-x0,x0)#(-0.1,0.1)#
                

            if i == 4:
                    ax.set_xlabel('$\kappa$',fontsize=22)
            ax.tick_params(labelsize=16)
            ax.set_ylim(-0.6,0.6)
            
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2,wspace=0, left=0.18, right=0.95)
        show()
        #savefig(CMBNG_dir+'plot/plot_noiseless_%s.jpg'%(['PDF','peaks'][j-1]))
        #savefig(CMBNG_dir+'plot_official/plot_noiseless_%s_diff.pdf'%(['PDF','peaks'][j-1]))
        #close()

if plot_sample_noiseless_noisy_map:
    import pickle
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #kmap_noiseless=WLanalysis.readFits(CMBNG_dir+'test_maps/WLconv_z1100.00_0001r.fits')
    #kmap_noiseless_8arcmin=WLanalysis.smooth(kmap_noiseless,78.01904762)
    kmap_noiseless_8arcmin=load(CMBNG_dir+'test_maps/kmap_noiseless_8arcmin_1r.npy')
    #### get noisy map ########
    FTmap = pickle.load(open(CMBNG_dir+'colin_noisy/kappaMapTT_10000sims/kappaMap0000TT_3.pkl'))
    areal = real(fftpack.ifft2(FTmap))
    inorm = (2*pi*3.5/360.0)/(77.0**2)
    areal /= inorm
    kmap_noisy=areal
    kmap_noisy_8arcmin = WLanalysis.smooth(kmap_noisy, 2.93)
    
    f=figure(figsize=(12,5))
    
    i=1
    for kmap in [kmap_noiseless_8arcmin, kmap_noisy_8arcmin]:
        ax=subplot(1,2,i)
        #imshow(kmap_noiseless_8arcmin,vmin=-0.1,vmax=0.1,extent=(0,3.5,0,3.5))
        im=ax.imshow(kmap,vmin=-3*std(kmap_noiseless_8arcmin),vmax=3*std(kmap_noiseless_8arcmin),extent=(0,3.5,0,3.5),cmap='PuOr')
        ax.annotate(r"$\rm{%s}$"%(['noiseless','noisy'][i-1]), 
                    xy=(0.05, 0.9), xycoords='axes fraction',fontsize=24,
                    bbox={'facecolor':'thistle', 'alpha':0.5})
        ax.set_xlabel(r"$\rm{deg}$",fontsize=22)
        ax.set_ylabel(r"$\rm{deg}$",fontsize=22)
        ax.tick_params(labelsize=16)
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="5%", pad=0.1)
        cbar=colorbar(im,format="%.2f",cax=cax)
        cbar.ax.tick_params(labelsize=16)
        i+=1
    plt.subplots_adjust(hspace=0.0, wspace=0.2, left=0.05, right=0.93, bottom=0.15, top=0.95)
    #show()
    savefig(CMBNG_dir+'plot_official/plot_maps.pdf')
    close()

if plot_noisy_peaks_PDF:
    import matplotlib.gridspec as gridspec 
    bins=25
    morebins=1
    #filtered=1
    if morebins:
        PDFbins = linspace(-0.24, 0.24, 201)    
        peak_bins = linspace(-0.1,0.18,36)
    else:
        PDFbins = linspace(-0.12, 0.12, 101)
        peak_bins = linspace(-0.06,0.14,26)
    PDFbins = WLanalysis.edge2center(PDFbins)
    peak_bins = WLanalysis.edge2center(peak_bins)
    
    ell_gadget = (WLanalysis.edge2center(logspace(log10(1.0),log10(77/2.0),26))*360./sqrt(12.25))

    all_stats77 = load (CMBNG_dir+'Pkappa_gadget/noisy/%snoisy_z1100_stats77_fidu_kappa%s.npy'%(['','filtered_'][filtered],['','_morebins'][morebins]))#_morebins
    ps_all77 = all_stats77[:,:bins]
    PDF_all77 = all_stats77[:, bins:bins+len(PDFbins)]
    peaks_all77 = all_stats77[:, bins+len(PDFbins):]

    all_stats77_GRF = load(CMBNG_dir+'Pkappa_gadget/noisy/%snoisy_z1100_stats77_fidu_gaus%s.npy'%(['','filtered_'][filtered],['','_morebins'][morebins]))#_morebins
    ps_all77_GRF = all_stats77_GRF[:,:bins]
    PDF_all77_GRF = all_stats77_GRF[:, bins:bins+len(PDFbins)]
    peaks_all77_GRF = all_stats77_GRF[:, bins+len(PDFbins):]
    
    PDFbins = PDFbins[0::2]
    PDF_all77 = sum(PDF_all77.reshape(all_stats77.shape[0],-1,2),axis=-1)
    PDF_all77_GRF = sum(PDF_all77_GRF.reshape(all_stats77.shape[0],-1,2),axis=-1)
    
    all_stats=[[ell_gadget, ps_all77, ps_all77_GRF],[PDFbins, PDF_all77, PDF_all77_GRF],[peak_bins, peaks_all77, peaks_all77_GRF]]
    
    gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
    for i in (1,2):#range(3):## 1 is PDF, 2 is peaks
        print i
        stats = all_stats[i]
        x,y,z=stats
        if i==1:
            y = y/(x[1]-x[0])
            z = z/(x[1]-x[0])
        else:
            y = y/12.5
            z = z/12.5
        f=figure(figsize=(8,6))
        ax=f.add_subplot(gs[0])
        ax2=f.add_subplot(gs[1],sharex=ax)
        
        #ax=f.add_subplot(111)
        ax.errorbar(x,mean(y,axis=0),std(y,axis=0)*sqrt(12/2e4),color='k',fmt='-',capsize=0,label=r'$\kappa\,\rm{maps}$',lw=1)
        ax.errorbar(x,mean(z,axis=0),std(z,axis=0)*sqrt(12/2e4),color='k',fmt='--',capsize=0,label=r'$\rm{GRF}$',lw=2)
        
        ######## SNR ########
        inside=((mean(y,axis=0)-mean(z,axis=0))/std(z,axis=0)/sqrt(12/2e4))**2
        inside=inside[~isnan(inside) & isfinite(inside)]
        print sqrt(sum(inside))
        
        ax.set_yscale('log')
        handles0, labels0 = ax.get_legend_handles_labels()
        handles=[h[0] for h in handles0]
        leg=ax.legend(handles,labels0, title=[r"$\rm{Gaussian\, (8')}$",r"$\rm{Wiener\, filter}$"][filtered],ncol=1, prop={'size':20}, loc=8, frameon=0)
        leg.get_title().set_fontsize('18')
        #plt.setp(leg.get_title(),fontsize='xx-small')
        ax.set_ylabel([r'$\rm{PDF}$', r'$\rm{N_{peaks}\, (deg^{-2})}$'][i-1],fontsize=18)
        
        ax.tick_params(labelsize=16)
        
        ax2.errorbar(x,mean(y,axis=0)/mean(z,axis=0)-1,std(y,axis=0)*sqrt(12/3e4)/mean(z,axis=0), color='k',fmt='-',capsize=0,label=r'$\kappa\,\rm{maps}$',lw=1)
        ax2.plot((-0.3,0.3),(0,0),'k--')
        ax2.set_xlabel('$\kappa$',fontsize=22)
        ax2.set_ylabel([r'$\rm{\Delta PDF/PDF}$', r'$\rm{\Delta N_{peaks}/ N_{peaks}}$'][i-1],fontsize=20)
        ax2.tick_params(labelsize=16)
        ax2.locator_params(axis = 'y', nbins = 5)
        
        
        if i==1:# and not filtered:
            ax2.set_xlim(-0.25, 0.18)
        if i==2:# and not filtered:
            ax2.set_xlim(-0.11, 0.16)
        ax.set_ylim(amax([ax.get_ylim()[0],2e-4]), amax(mean(z,axis=0))*2)
        ax2.set_ylim([-0.25,0.25])
        ax2.locator_params(axis = 'x', nbins = 5)
            #else:
                #ax2.set_ylim([[-0.23,0.13],[-0.25,0.25]][i-1])
                #ax2.set_xlim(x[0],[.18,x[-1]][i-1])
        #else:
            #ax.set_ylim(1.2e-6, 2e-4)
        plt.subplots_adjust(hspace=0.05,left=0.15)
        plt.setp(ax.get_xticklabels(), visible=False)
        #show()
        #savefig(CMBNG_dir+'plot_official/plot_noisy_%s%s%s.pdf'%(['ps','PDF','peaks'][i],['','_filtered'][filtered],['','_morebins'][morebins]))
        #savefig(CMBNG_dir+'plot_official/png/plot_noisy_%s%s%s.png'%(['ps','PDF','peaks'][i],['','_filtered'][filtered],['','_morebins'][morebins]))
        close()

if plot_reconstruction_noise: 
    mat_GRF=load('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_gadget/noiseless/GRF_{0}_ps_PDF_pk_z1100.npy'.format(fidu_cosmo))
    mat_kappa=load('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_gadget/noiseless/kappa_{0}_ps_PDF_pk_z1100.npy'.format(fidu_cosmo))
    N=len(mat_GRF)
    
    
    PDFbins = linspace(-0.12, 0.12, 101)
    PDFbins = WLanalysis.edge2center(PDFbins)
    peak_bins = linspace(-0.06, 0.14, 26)
    peak_bins = WLanalysis.edge2center(peak_bins)
    ell_77 = (WLanalysis.edge2center(logspace(log10(1.0),log10(77/2.0),26))*360./sqrt(12.25))


    ips_GRF = array([mat_kappa[x][0] for x in range(N)])
    iPDF_GRF = array([mat_GRF[x][1][-1] for x in range(N)])
    ipeak_GRF = array([mat_GRF[x][2][-1] for x in range(N)])
    ell_2048 = (WLanalysis.edge2center(logspace(log10(1.0),log10(1024.0),51))*360./sqrt(12.25))
    
    bins=25
    all_stats77_GRF = load(CMBNG_dir+'Pkappa_gadget/noisy/noisy_z1100_stats77_fidu_gaus.npy')
    ps_all77_GRF = all_stats77_GRF[:,:bins]
    PDF_all77_GRF = all_stats77_GRF[:, bins:bins+len(PDFbins)]
    peaks_all77_GRF = all_stats77_GRF[:, bins+len(PDFbins):]
    
    idx_noiseless = where(~isnan(ips_GRF[0]))[0]
    idx_noisy = where(~isnan(ps_all77_GRF[0]))[0]
    xbins=[[ell_2048[idx_noiseless], ell_77[idx_noisy]],[WLanalysis.edge2center(PDFbin_arr[-1]), PDFbins],[WLanalysis.edge2center(peak_bins_arr[-1]), peak_bins]]
    pss=[[ips_GRF[:,idx_noiseless],ps_all77_GRF[:,idx_noisy]], [iPDF_GRF/(0.08-0.0784), PDF_all77_GRF/0.0024], [ipeak_GRF/12.25, peaks_all77_GRF/12.25]]
    ############ plot, ps, peaks and PDF ##########
    f=figure(figsize=(6,8))
    #f.text(0.02, 0.5, [r'$\rm{PDF}$', r'$\rm{N_{peaks}}$'][j-1], va='center', rotation='vertical',fontsize=22)
    
    for i in arange(3):
        xnoiseless, xnoisy = xbins[i]
        ynoiseless, ynoisy = pss[i]
        
        ax=f.add_subplot(3, 1, i+1)
        
        ax.plot(xnoiseless, mean(ynoiseless, axis=0), 'k',label=r'$\rm{noiseless \,GRF}$',lw=1)
        ax.plot(xnoisy, mean(ynoisy, axis=0),'k--',label=r'$\rm{noisy \,GRF}$',lw=2)

        ax.locator_params(axis = 'x', nbins = 4)
        ax.locator_params(axis = 'y', nbins = 2)
        ax.set_yscale('log')

        if i ==0:
            ax.set_ylim(1.2e-4,0.35)
            ax.set_xscale('log')
            ax.set_xlabel('$\ell$',fontsize=22)
            ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}/2\pi$',fontsize=20)
            ax.set_xlim(90,3e3)
        elif i==1:
            leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=8)
            leg.get_frame().set_visible(False)
            ax.set_xlabel('$\kappa$',fontsize=22)
            ax.set_ylabel(r'$\rm{PDF}$',fontsize=20)
            ax.set_ylim(0.3,25)
            ax.set_xlim(xnoisy[0],xnoisy[-1])
        else:
            ax.set_xlabel(r'$\kappa$',fontsize=22)
            ax.set_ylabel(r'$\rm{N_{peaks}\,(deg^{-2})}$',fontsize=20)
            ax.set_xlim(xnoisy[0],xnoisy[-1])
            ax.set_ylim(5e-3,0.5)
        ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4,bottom=0.1,right=0.96)#hspace=0.2,wspace=0, left=0.18, right=0.95)
    #show()
    savefig(CMBNG_dir+'plot_official/plot_reconstruction.pdf')
    close()


#if plot_contour_peaks:
    
########### cosmology constraints #############
# first get average of everything
# covariance matrix
cosmo_params = array([[float(cosmo[2:7]), float(cosmo[-5:])] for cosmo in cosmo_arr])

def getmat(cosmo, psPDFpk='ps', sigmaG_idx=0, avg=1):
    '''return the matrix of 'ps','PDF',or'pk', for sigmaG, if avg=1, then return average, instead of all 1024 realizations'''
    print cosmo
    mat_kappa=load('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_gadget/kappa_{0}_ps_PDF_pk_z1100.npy'.format(cosmo))
    N=len(mat_kappa)
    #Om = float(cosmo[2:7])
    #si8 = float(cosmo[-5:])
    if psPDFpk=='ps':
        ips = array([mat_kappa[x][0] for x in range(N)])
    elif psPDFpk=='PDF':
        ips = array([mat_kappa[x][1][sigmaG_idx] for x in range(N)])
    elif psPDFpk=='pk':
        ips = array([mat_kappa[x][2][sigmaG_idx] for x in range(N)])
    if avg:
        ips=mean(ips,axis=0)
    return ips

#save(CMBNG_dir+'mat_ps_avg.npy', [getmat(cosmo) for cosmo in cosmo_arr])
#save(CMBNG_dir+'mat_ps_fidu.npy', getmat(fidu_cosmo,avg=0))

#[save(CMBNG_dir+'mat/mat_%s_sigmaG%i_avg.npy'%(pk, isigmaG), [getmat(cosmo, psPDFpk=pk, sigmaG_idx=isigmaG) for cosmo in cosmo_arr]) for pk in ['PDF','pk'] for isigmaG in range(len(sigmaG_arr))] 

#[save(CMBNG_dir+'mat/mat_%s_sigmaG%i_fidu.npy'%(pk, isigmaG), getmat(fidu_cosmo, psPDFpk=pk, sigmaG_idx=isigmaG, avg=0)) for pk in ['PDF','pk'] for isigmaG in range(len(sigmaG_arr))] 


ell_gadget = (WLanalysis.edge2center(logspace(log10(1.0),log10(1024),51))*360./sqrt(12.25))
def create_prob_plane(psPDFpk='pk', sigmaG_idx=0):
    if psPDFpk =='ps':
        idx2000=where(ell_gadget<10000)[0]
        obs_arr = load(CMBNG_dir+'mat/mat_ps_avg.npy')[:,idx2000]
        fidu_mat = load(CMBNG_dir+'mat/mat_ps_fidu.npy')[:,idx2000]
    else:
        obs_arr = load(CMBNG_dir+'mat/mat_%s_sigmaG%i_avg.npy'%(psPDFpk, sigmaG_idx))
        
        fidu_mat = load(CMBNG_dir+'mat/mat_%s_sigmaG%i_fidu.npy'%(psPDFpk, sigmaG_idx))
    
    idx = where(~isnan(mean(fidu_mat,axis=0))&(mean(fidu_mat,axis=0)!=0))[0]
    fidu_mat=fidu_mat[:,idx]
    interp_cosmo=WLanalysis.buildInterpolator2D(obs_arr[:,idx], cosmo_params)
    
    #cov_mat = 
    cov_mat = cov(fidu_mat,rowvar=0)/(2e4/12.5)
    cov_inv = mat(cov_mat).I
    
    def chisq_fcn(param1, param2):
        model = interp_cosmo((param1,param2))
        del_N = np.mat(model - mean(fidu_mat,axis=0))
        chisq = float(del_N*cov_inv*del_N.T)
        return chisq
        
    prob_plane = WLanalysis.prob_plane(chisq_fcn, om_arr, si8_arr)
    return prob_plane[1]


############ create probability planes #######
#for sigmaG_idx in range(5):
    #print sigmaG_idx
    #if sigmaG_idx==0:
        #iP = create_prob_plane(psPDFpk='ps')
        #save(CMBNG_dir+'mat/Prob_ps_ell2000.npy',iP)
    #for psPDFpk in ['pk','PDF']:
        #iP = create_prob_plane(psPDFpk=psPDFpk, sigmaG_idx=sigmaG_idx)
        #save(CMBNG_dir+'mat/Prob_%s_sigmaG%i.npy'%(psPDFpk, sigmaG_idx),iP)

#iP = create_prob_plane(psPDFpk='ps')
#save(CMBNG_dir+'mat/Prob_ps_ell10000.npy',iP)
        
#imshow(iP,origin='lower',extent=[si80,si81,om0,om1],interpolation='nearest')
#xlabel('si8')
#ylabel('om')
#show()
###########################################



if plot_contour_PDF_pk:
    om_fidu, si8_fidu=cosmo_params[12]
    #0.05, 0.05 # 0.01, 0.01
    if area_scaling:
        del_om, del_si8 =0.15, 0.15
        del_om /= sqrt(2e4/fsky_deg)
        del_si8/= sqrt(2e4/fsky_deg)
    else:
        del_om, del_si8 =0.05, 0.05
    om0,om1,si80,si81=om_fidu-del_om, om_fidu+del_om, si8_fidu-del_si8, si8_fidu+del_si8
    jjj=250#100#
    om_arr= linspace(om0,om1,jjj)
    si8_arr=linspace(si80,si81, jjj+1)
    colors=[['darkorchid','plum','mediumvioletred','limegreen'],
            ['steelblue','deepskyblue','dodgerblue','limegreen']]
    for imethod in ('clough',):#'Rbf','linear'):#imethod='Rbf'#'clough'#'linear'
        for j in range(2):
            istat=['PDF','Peaks'][j]
            
            seed(55)
            X, Y = np.meshgrid(om_arr,si8_arr)
            labels = [r"$\rm{%s\,(%s')}$"%(istat,sigmaG) for sigmaG in sigmaG_arr[[1,3,4]] ]
            labels.append(r"$\rm{PS}\,(\ell<2,000)$")
            #labels.append(r"$\rm{PS}(\ell<10,000)$")
            lines=[]
            f=figure(figsize=(8,6))
            ax=f.add_subplot(111)
            iextent=[si80,si81,om0,om1]
            jjj=-1
            for sigmaG_idx in (1,3,4,5):#range(6):
                jjj+=1
                if area_scaling:
                    if sigmaG_idx==5:
                        prob=load(CMBNG_dir+'mat/Prob_fsky%i_noiseless_ps_%s_sigmaG10_del0.15.npy'%(fsky_deg,imethod))
                    else:
                        prob=load(CMBNG_dir+'mat/Prob_fsky%i_noiseless_%s_%s_sigmaG%02d_del0.15.npy'%(fsky_deg,['PDF','peaks'][j], imethod, sigmaG_arr[sigmaG_idx]*10))
                else:
                    if sigmaG_idx==5:
                        prob=load(CMBNG_dir+'mat/Prob_noiseless_ps_%s_sigmaG10.npy'%(imethod))
                    else:
                        prob=load(CMBNG_dir+'mat/Prob_noiseless_%s_%s_sigmaG%02d.npy'%(['PDF','peaks'][j], imethod, sigmaG_arr[sigmaG_idx]*10))
                prob[isnan(prob)]=0
                V=WLanalysis.findlevel(prob)
                icolor=colors[j][jjj]
                CS=ax.contour( X, Y, prob.T, levels=[V[0],], origin='lower', extent=iextent,linewidths=2, colors=[icolor, ])
                lines.append(CS.collections[0])

            leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':18},loc=0)#,title=r'${\rm noisy}$')
            leg.get_frame().set_visible(False)
            ax.tick_params(labelsize=16)
            ax.set_title(r'${\rm noiseless}$',fontsize=22)
            ax.locator_params(axis = 'both', nbins = 5)
            ax.set_ylabel('$\sigma_8$',fontsize=22)
            ax.set_xlabel('$\Omega_m$',fontsize=22)
            #ax.grid(True)
            #ax.set_xlim(0.276, 0.324)
            #ax.set_ylim(0.777, 0.797)
            ax.set_xlim(0.263, 0.343)
            ax.set_ylim(0.748, 0.829)
            ax.plot(0.296, 0.786,'xk',markersize=5,mew=2)

            plt.subplots_adjust(hspace=0.0,bottom=0.13,right=0.96,left=0.15)
            #show()
        
            savefig(CMBNG_dir+'plot_official/plot_contour_noiseless_%s_%s.pdf'%(istat,imethod));close()
            #savefig(CMBNG_dir+'plot_official/png/contour%s_noiseless_%s_%s.png'%(['','_areascale'][area_scaling],istat,imethod))
            #close()


if plot_contour_noisy_old:
    filtered = 0
    noise = 'noiseless'#'noisy'#
    om_fidu, si8_fidu=cosmo_params[12]
    #0.05, 0.05 # 0.01, 0.01
    if area_scaling_noisy:
        del_om, del_si8 =0.15,0.15
        del_om /= sqrt(2e4/fsky_deg)
        del_si8/= sqrt(2e4/fsky_deg)
    else:
        del_om, del_si8 =0.05,0.05
    om0,om1,si80,si81=om_fidu-del_om, om_fidu+del_om, si8_fidu-del_si8, si8_fidu+del_si8
    jjj=250#100#
    om_arr= linspace(om0,om1,jjj)
    si8_arr=linspace(si80,si81, jjj+1)
    iextent=[si80,si81,om0,om1]
    
    X, Y = np.meshgrid(si8_arr, om_arr)
    #del_om_noisy, del_si8_noisy = 0.05, 0.05
    #om0_noisy,om1_noisy,si80_noisy,si81_noisy=om_fidu-del_om_noisy, om_fidu+del_om_noisy, si8_fidu-del_si8_noisy, si8_fidu+del_si8_noisy
    #jjj=250
    #om_arr_noisy= linspace(om0_noisy,om1_noisy,jjj)
    #si8_arr_noisy=linspace(si80_noisy,si81_noisy, jjj+1)
    #iextent=[si80_noisy,si81_noisy,om0_noisy,om1_noisy]
    
    
    for imethod in ('clough',):#'Rbf','linear'):#imethod='clough'#'Rbf'#'linear'#
        seed(55)
        
        labels = [r"$\rm{%s}$"%(istat) for istat in ['Power\; Spectrum','PDF','Peaks','PS+PDF+Peaks'] ]
        lines=[]
        f=figure(figsize=(8,6))
        ax=f.add_subplot(111)
        
        jjj=-1
        for istat in ['ps','PDF','peaks','all3']:#['ps','peaks']:#
            jjj+=1
            print istat
            
            
            if area_scaling_noisy:
                if noise == 'noisy':
                                    
                    if istat != 'ps':
                        prob=load(CMBlensing_dir+'mat/%sProb_fsky%i_noisy_%s_%s_sigmaG80_del0.15.npy'%(['','filtered_'][filtered],fsky_deg_noisy,istat, imethod))
                        if istat=='peaks':
                            prob=load(CMBlensing_dir+'mat/Prob_fsky%i_noisy_%s_%s_sigmaG80_del0.15.npy'%(fsky_deg_noisy,istat, imethod))
                        #if istat=='all3':
                            #prob=load(CMBlensing_dir+'mat/optimize_%sProb_fsky%i_noisy_%s_%s_sigmaG80_del0.15.npy'%(['','filtered_'][filtered],fsky_deg_noisy,istat, imethod))
                        
                    else:
                        prob=load(CMBlensing_dir+'mat/Prob_fsky%i_noisy_%s_%s_sigmaG05_del0.15.npy'%(fsky_deg_noisy, istat, imethod))
                        
                    
                else:
                    
                    if istat == 'ps':
                        prob=load(CMBlensing_dir+'mat/Prob_noiseless_%s_%s_sigmaG10.npy'%(istat, imethod))
                    else:
                        prob=load(CMBlensing_dir+'mat/Prob_noiseless_%s_%s_sigmaG80.npy'%(istat, imethod))
            else: ###### no area scaling
                if noise == 'noisy':
                    if istat != 'ps':
                        prob=load(CMBlensing_dir+'mat/%sProb_noisy_%s_%s_sigmaG80.npy'%(['','filtered_'][filtered],istat, imethod))
                        
                    else:
                        prob=load(CMBlensing_dir+'mat/Prob_noisy_%s_%s.npy'%(istat, imethod))
                    
                else:
                    
                    if istat == 'ps':
                        prob=load(CMBlensing_dir+'mat/Prob_noiseless_%s_%s_sigmaG10.npy'%(istat, imethod))
                    else:
                        prob=load(CMBlensing_dir+'mat/Prob_noiseless_%s_%s_sigmaG80.npy'%(istat, imethod))
                        
            prob[isnan(prob)]=0.0
            V=WLanalysis.findlevel(prob)
            print 'N_pixels about 68%:',sum(prob>V[0])
            icolor=['darkorchid','mediumvioletred','darkorange','limegreen'][jjj]
            CS=ax.contour(X, Y, prob, levels=[V[0],], origin='lower',linewidths=2, colors=[icolor, ])#extent=iextent
            lines.append(CS.collections[0])

        leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':18},loc=0)
        leg.get_frame().set_visible(False)
        ax.tick_params(labelsize=16)
        ax.locator_params(axis = 'both', nbins = 5)
        ax.set_xlabel('$\sigma_8$',fontsize=22)
        ax.set_ylabel('$\Omega_m$',fontsize=22)
        ax.grid(True)
        ax.set_xlim(0.754,0.832)#(0.78,0.797)
        ax.set_ylim(0.267,0.333)
        #ax.plot(0.786, 0.296,'ko'  )#0.29591837,  0.78571429
        plt.subplots_adjust(hspace=0.0,bottom=0.13,right=0.96,left=0.15)
        show()
        
        #savefig(CMBNG_dir+'plot_official/contour%s_%s_%s.pdf'%(['','_filtered'][filtered],noise,imethod))
        #savefig(CMBNG_dir+'plot_official/png/optimize_contour%s_fsky%i_%s_%s.png'%(['','_filtered'][filtered],fsky_deg_noisy,noise,imethod))
        #close()


if plot_corr_mat:
    
    if not do_noisy: ##noiseless
        ell_arr2048=array([   110.50448683,    126.93632224,    145.81154455,    167.49348136,
          192.39948651,    221.00897366,    253.87264448,    291.62308909,
          334.98696272,    384.79897302,    442.01794731,    507.74528896,
          583.24617818,    669.97392544,    769.59794604,    884.03589463,
         1015.49057792,   1166.49235637,   1339.94785088,   1539.19589208,
         1768.07178925,   2030.98115583,   2332.98471274,   2679.89570175,
         3078.39178417,   3536.14357851,   4061.96231167,   4665.96942547,
         5359.79140351,   6156.78356833,   7072.28715702,   8123.92462333,
         9331.93885094,  10719.58280701,  12313.56713667,  14144.57431404,
        16247.84924667,  18663.87770189,  21439.16561402,  24627.13427334,
        28289.14862808,  32495.69849334,  37327.75540378,  42878.33122805,
        49254.26854668,  56578.29725615,  64991.39698667,  74655.51080755,
        85756.6624561 ,  98508.53709335])
        #ell_arr2048=WLanalysis.PowerSpectrum(zeros(shape=(2048,2048)), bins=50)[0]

        #fidu_stats = load(CMBlensing_dir+'Pkappa_gadget/noiseless/kappa_Om0.296_Ol0.704_w-1.000_si0.786_ps_PDF_pk_z1100_10240.npy')
        
        #ps_fidu_noiseless = array([fidu_stats[j][0] for j in range(1024,10240)]).squeeze() 
        #idx_cut2000 = where( (ell_arr2048<2700) & (~isnan(ps_fidu_noiseless[0])))[0]
        
        #ps_fidu_noiseless = ps_fidu_noiseless[:,idx_cut2000]
        #PDF_fidu_noiseless = array([fidu_stats[j][1] for j in range(1024,10240)])
        #peaks_fidu_noiseless = array([fidu_stats[j][2] for j in range(1024,10240)])
        
        #ismooth=1

        #fidu_stats2 = concatenate([ps_fidu_noiseless, sum(PDF_fidu_noiseless[:,ismooth,:].reshape(-1,50,2),axis=-1), peaks_fidu_noiseless[:,ismooth,:]],axis=1)
        
        #cov_mat = cov(fidu_stats2,rowvar=0)
        #corr_mat = WLanalysis.corr_mat(cov_mat)
        
        #save(CMBNG_dir+'corr_mat.npy',corr_mat)
        corr_mat=load(CMBNG_dir+'corr_mat.npy')
    else:
        filtered=0
        fidu_stats77 = load(CMBlensing_dir+'Pkappa_gadget/noisy/%snoisy_z1100_stats77_fidu_kappa.npy'%(['','filtered_'][filtered]))
        ps = fidu_stats77[:,:25]
        ps = ps[:,~isnan(ps[0])][:,:20]
        PDF = sum(fidu_stats77[:,25:125].reshape(-1,50,2),axis=-1)
        peaks = fidu_stats77[:,125:]
        all_stats = concatenate([ps,PDF,peaks],axis=1)
        #fidu_stats77 = fidu_stats77[:,~isnan(fidu_stats77[0])]
        cov_mat = cov(all_stats,rowvar=0)
        corr_mat = WLanalysis.corr_mat(cov_mat)
    
    from matplotlib.patches import FancyBboxPatch
    fig=figure(figsize=(7,6))
    ax=fig.add_subplot(111)
    im=ax.imshow(corr_mat,origin='lower',interpolation='nearest',cmap='PuOr',vmax=0.1,vmin=-0.1)
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.subplots_adjust(top=0.88,bottom=0.12,right=0.92,left=0.07)
    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    left='off',
    right='off'
    )
    #plt.setp(ax.get_yticklabels(), visible=False)
    #plt.setp(ax.get_xticklabels(), visible=False)

    ax.text(0, 96, 'PS',  fontsize=14)
    bb=FancyBboxPatch((-1,95), 25, 10, fc='thistle', alpha=1)
    ax.set_ylim(0,100)
    ax.add_patch(bb)
    ax.text(20, 96, 'PDF', fontsize=14)
    bb=FancyBboxPatch((20,95), 100, 10, fc='powderblue', alpha=1)
    ax.add_patch(bb)
    ax.text(70, 96, 'Peaks',  fontsize=14)
    bb=FancyBboxPatch((70,95), 30, 10, fc='lightsalmon', alpha=1)
    ax.add_patch(bb)
    ax.set_yticks(ax.get_yticks()[:-1])
    ax.set_title(r'${\rm %s}$'%(['noiseless','noisy'][do_noisy]),fontsize=22)
    
    plt.subplots_adjust(hspace=0.0,wspace=0, left=0.04, right=0.97,bottom=0.08,top=0.9)
    show()
    #savefig(CMBlensing_dir+'plot_official/corr_mat%s.pdf'%(['','_noisy'][do_noisy]))
    #close()

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, lw=4, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, fill=0, color='darkviolet', lw=lw)

    CS=ax.add_artist(ellip)
    return CS
    
if plot_contour_theory:
    import pickle
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    
    Ls,NL=pickle.load(open('/Users/jia/Desktop/lensNoisePower6014_4000_SimCosmo.pkl','r'))
    Nlkk = NL['TT'] * (Ls*(Ls+1)/2.0)**2

    ell,Pfidu=genfromtxt(CMBlensing_dir+'Pkappa_nicaea/Pkappa_nicaea25_Om0.296_Ol0.704_w-1.000_si0.786_1100').T
    ell, Pom =genfromtxt(CMBlensing_dir+'Pkappa_nicaea/Pkappa_nicaea25_Om0.299_Ol0.701_w-1.000_si0.786_1100').T
    ell, Psi=genfromtxt(CMBlensing_dir+'Pkappa_nicaea/Pkappa_nicaea25_Om0.296_Ol0.704_w-1.000_si0.794_1100').T

    Pfidu /= ell*(ell+1)/2/pi
    Pom /=  ell*(ell+1)/2/pi
    Psi /=  ell*(ell+1)/2/pi

    fsky=20000./41253
    dom=0.299-0.296
    dsi=0.794-0.786
    idx= where((ell<2000)&(ell>50))[0]
    delta_ell= ell[idx]-ell[idx[0]-1:idx[-1]]
    iNlkk=interpolate.interp1d(Ls, Nlkk)(ell[idx])

    var_theory = 2*(Pfidu[idx]+iNlkk)**2/fsky/(2.0*ell[idx]+1.0)/delta_ell
    
    cov_mat = zeros((len(idx),len(idx)))
    for i in range(len(idx)):
        cov_mat[i,i]=var_theory[i]
    cov_inv = mat(cov_mat).I

    Nom=mat(((Pom-Pfidu)/dom)[idx])
    Nsi=mat(((Psi-Pfidu)/dsi)[idx])

    F=mat(zeros((2,2)))
    F[0,0]=0.5*trace(cov_inv*(2*Nom.T*Nom))
    F[1,1]=0.5*trace(cov_inv*(2*Nsi.T*Nsi))
    F[0,1]=0.5*trace(cov_inv*(Nom.T*Nsi+Nsi.T*Nom))
    F[1,0]=0.5*trace(cov_inv*(Nsi.T*Nom+Nom.T*Nsi))
    F_inv = F.I
    
    lines=[]
    fidu_point=[0.296, 0.786]
    f=figure(figsize=(8,6))
    ax=f.add_subplot(111)
    plot_cov_ellipse(F_inv, fidu_point,lw=4,ax=ax)
    ax.set_xlim(0.283, 0.3125)
    ax.set_ylim(0.772, 0.805)
    ax.set_xlabel('$\Omega_m$',fontsize=24,weight='bold')
    ax.set_ylabel('$\sigma_8$',fontsize=24,weight='bold')
    ax.plot(fidu_point[0],fidu_point[1],'xk',markersize=12,mew=4)
    
    ax.annotate('Simulation', 
                xy=(0.288, 0.7935),weight='bold', size=18,color='limegreen',
                xytext=(0.289, 0.797),
                arrowprops=dict(facecolor='limegreen', shrink=0.05,ec='none'),
            )
    ax.annotate('Analytical Theory', 
                xy=(0.303, 0.787), weight='bold',
                size=18,color='darkviolet',
                xytext=(0.30, 0.7915),
                arrowprops=dict(facecolor='darkviolet', shrink=0.05,ec='none'),
            )
    
    
    ####### sim contour
    om_fidu, si8_fidu=fidu_point
    del_om, del_si8 =0.05,0.05
    om0,om1,si80,si81=om_fidu-del_om, om_fidu+del_om, si8_fidu-del_si8, si8_fidu+del_si8
    jjj=250
    om_arr= linspace(om0,om1,jjj)
    si8_arr=linspace(si80,si81, jjj+1)
    iextent=[si80,si81,om0,om1]
    X, Y = np.meshgrid(om_arr, si8_arr)
    prob=load(CMBlensing_dir+'mat/Prob_noisy_ps_clough.npy').T
    V=WLanalysis.findlevel(prob)

    CS=ax.contour(X, Y, prob, levels=[V[0]], origin='lower',linewidths=4,colors='limegreen',label='sims')
   
    ax.tick_params(labelsize=16)
    ax.locator_params(axis = 'both', nbins = 5)
    #######
    plt.subplots_adjust(left=0.14,bottom=0.14,right=0.96,top=0.96)
    
    show()
    #savefig(CMBlensing_dir+'plot_official/plot_contour_fisher.pdf')
    #close()

if plot_contour_PDF_pk_noisy:
    om_fidu, si8_fidu=cosmo_params[12]
    del_om, del_si8 =0.05, 0.05
    om0,om1,si80,si81=om_fidu-del_om, om_fidu+del_om, si8_fidu-del_si8, si8_fidu+del_si8
    
    colors=[['darkorchid','plum','mediumvioletred','darkslategrey','limegreen'],
            ['steelblue','deepskyblue','dodgerblue','darkslategrey','limegreen']]
    for imethod in ('clough',):#'Rbf','linear'):#imethod='Rbf'#'clough'#'linear'
        for j in range(2):
            istat=['PDF','Peaks'][j]
            
            seed(55)
            
            labels = [r"$\rm{%s\,(%s')}$"%(istat,sigmaG) for sigmaG in sigmaG_arr[[1,3]] ]
            labels.append(r"$\rm{%s\,(filtered)}$"%(istat))
            labels.append(r"$\rm{PS}\,(\ell<2,000)$")
            #labels.append(r"$\rm{PS}(\ell<10,000)$")
            lines=[]
            f=figure(figsize=(8,6))
            ax=f.add_subplot(111)
            iextent=[si80,si81,om0,om1]
            jjj=-1
            for sigmaG_idx in (1,3,4, 6, 5):#range(6):
                jjj+=1
                if sigmaG_idx==6:
                    iii=100#250#
                    prob=load(CMBNG_dir+'mat/filtered_Prob_fsky20000_noisy_%s_clough_del0.05.npy'%(['PDF','peaks'][j]))#_sigmaG80
                elif sigmaG_idx==5:
                    iii=250
                    prob=load(CMBNG_dir+'mat/Prob_noisy_ps_clough.npy')
                else:
                    iii=100
                    prob=load(CMBNG_dir+'mat/Prob_fsky20000_noisy_%s_clough_sigmaG%02d_del0.05.npy'%(['PDF','peaks'][j], sigmaG_arr[sigmaG_idx]*10))
                om_arr= linspace(om0,om1,iii)
                si8_arr=linspace(si80,si81, iii+1)
                
                X, Y = np.meshgrid(om_arr,si8_arr)
                prob[isnan(prob)]=0
                V=WLanalysis.findlevel(prob)
                icolor=colors[j][jjj]
                if sigmaG_idx!=4:
                    #ax.contourf( X, Y, prob.T, levels=[V[0],V[1]], origin='lower', extent=iextent,linewidths=4, colors=[icolor, ])
                    #if sigmaG_idx==6:
                        #CS=ax.contour( X, Y, prob.T, levels=[V[0],], origin='lower', extent=iextent,linewidths=2, colors=[icolor, ],linestyles='dashed')
                        
                    #else:
                    CS=ax.contour( X, Y, prob.T, levels=[V[0],], origin='lower', extent=iextent,linewidths=4, colors=[icolor, ])
                    
                    lines.append(CS.collections[0])

            leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':18},loc=0)
            leg.get_frame().set_visible(False)
            ax.tick_params(labelsize=16)
            ax.locator_params(axis = 'both', nbins = 6)
            ax.set_ylabel('$\sigma_8$',fontsize=22)
            ax.set_xlabel('$\Omega_m$',fontsize=22)
            #ax.grid(True)
            #ax.set_xlim(0.266, 0.343)
            #ax.set_ylim(0.751, 0.819)
            ax.set_xlim(0.263, 0.343)
            ax.set_ylim(0.748, 0.829)
            ax.plot(0.296, 0.786,'xk',markersize=5,mew=2)
            ax.set_title(r'${\rm noisy}$',fontsize=22)
            plt.subplots_adjust(hspace=0.0,bottom=0.13,right=0.96,left=0.15)
            #show()
            
            savefig(CMBNG_dir+'plot_official/plot_contour_noisy_%s_%s.pdf'%(istat,imethod))
            close()

if plot_contour_comb:
    om_fidu, si8_fidu=cosmo_params[12]
    del_om, del_si8 =0.05, 0.05
    om0,om1,si80,si81=om_fidu-del_om, om_fidu+del_om, si8_fidu-del_si8, si8_fidu+del_si8
    
    colors=['limegreen','orchid','dodgerblue',]
    imethod='clough'
    labels = [r"$\rm{PS}\,(\ell<2,000)$",
              #r"$\rm{PS\,+\,PDF(5')\,+\,Peaks(5')}$",
              r"$\rm{PDF(5')\,+\,Peaks(5')}$",
              r"$\rm{PS\,+\,PDF(5')\,+\,Peaks(5')}$",
              #r"$\rm{PS\,+\,PDF(filtered)\,+\,Peaks(filtered)}$"
              ]
    f=figure(figsize=(8,6))
    ax=f.add_subplot(111)
    iextent=[si80,si81,om0,om1]
    prob_arr = [] 
    prob_arr.append(load(CMBNG_dir+'mat/Prob_noisy_ps_clough.npy'))
    prob_arr.append(load(CMBNG_dir+'mat/Prob_fsky20000_noisy_pkPDF_clough_sigmaG50_del0.05.npy'))
    prob_arr.append(load(CMBNG_dir+'mat/Prob_fsky20000_noisy_comb_clough_sigmaG50_del0.05.npy'))
    #prob_arr.append(load(CMBNG_dir+'mat/filtered_Prob_fsky20000_noisy_comb_clough_del0.05.npy'))#optimize_Prob_fsky20000_noisy_comb_clough_sigmaG50_del0.05.npy'))
    lines=[]
    lws=[4,2,3]
    lss=['solid','solid','dashed']
    for jjj in range(3):

        if jjj==0:
            iii=250
        else:
            iii=100
        prob=prob_arr[jjj]    
        om_arr= linspace(om0,om1,iii)
        si8_arr=linspace(si80,si81, iii+1)
        
        ############ marginalized error #############
        prob_om = sum(prob, axis=1)
        prob_si8 = sum(prob, axis=0)
        delta_oms = om_arr[prob_om>WLanalysis.findlevel(prob_om)[0]]
        delta_si8s = si8_arr[prob_si8>WLanalysis.findlevel(prob_si8)[0]]
        print ['PS','PDF+Peaks','PS+PDF+Peaks'][jjj], 'delta_om', delta_oms[-1]-delta_oms[0], 'delta_si8',  delta_si8s[-1]-delta_si8s[0]
        
        X, Y = np.meshgrid(om_arr,si8_arr)
        prob[isnan(prob)]=0
        V=WLanalysis.findlevel(prob)
        
        CS=ax.contour( X, Y, prob.T, levels=[V[0],], origin='lower', extent=iextent,linewidths=lws[jjj],linestyles=lss[jjj], colors=colors[jjj])
                
        lines.append(CS.collections[0])

    leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':18},loc=0)
    leg.get_frame().set_visible(False)
    ax.tick_params(labelsize=16)
    ax.locator_params(axis = 'both', nbins = 6)
    ax.set_ylabel('$\sigma_8$',fontsize=22)
    ax.set_xlabel('$\Omega_m$',fontsize=22)

    ax.set_xlim(0.282, 0.314)
    ax.set_ylim(0.768, 0.809)
    ax.plot(0.296, 0.786,'xk',markersize=10,mew=4)

    plt.subplots_adjust(hspace=0.0,bottom=0.13,right=0.96,left=0.15)
    #show()
    
    #savefig(CMBNG_dir+'plot_official/plot_contour_noisy_comb_%s.pdf'%(imethod))
    close()
    
if plot_Cell_om_si:
    ell,Pfidu=genfromtxt(CMBNG_dir+'Pkappa_nicaea/Pkappa_nicaea25_Om0.296_Ol0.704_w-1.000_si0.786_1100').T
    ell, Pom =genfromtxt(CMBNG_dir+'Pkappa_nicaea/Pkappa_nicaea25_Om0.299_Ol0.701_w-1.000_si0.786_1100').T
    ell, Psi=genfromtxt(CMBNG_dir+'Pkappa_nicaea/Pkappa_nicaea25_Om0.296_Ol0.704_w-1.000_si0.794_1100').T
    f=figure()
    ax=f.add_subplot(111)
    ax.plot(ell, zeros(len(ell)), '--', color='k',lw=2, label=r'${\rm Fiducial}$')
    ax.plot(ell, Pom/Pfidu-1, '-', color='darkorchid',lw=4, label=r'$+1\%\; \Omega_m$')
    ax.plot(ell, Psi/Pfidu-1, '-', color='mediumvioletred',lw=2, label=r'$+1\%\; \sigma_8$')
    legend(loc=2,frameon=0,fontsize=22)
    ax.set_xlabel(r'$\ell$',fontsize=22)
    ax.set_ylabel(r'$\Delta C_{\ell}/ C_{\ell}^{\rm fiducial}$',fontsize=22)
    ax.set_xlim(10,2500)
    ax.set_ylim(-0.024, 0.05)
    #ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=16)
    plt.subplots_adjust(hspace=0.0, wspace=0.2, left=0.15, right=0.95, bottom=0.15, top=0.95)
    #show()
    ax.locator_params(axis = 'y', nbins = 5)
    savefig(CMBNG_dir+'plot_official/plot_Cell_diff.pdf')
    close()

if plot_interp:
    ips_mean, ips_std, ips_interp = load(CMBlensing_dir+'interp.npy')
    ips_std *= 1/sqrt(1000)
    ell_arr2048=array([   110.50448683,    126.93632224,    145.81154455,    167.49348136,
          192.39948651,    221.00897366,    253.87264448,    291.62308909,
          334.98696272,    384.79897302,    442.01794731,    507.74528896,
          583.24617818,    669.97392544,    769.59794604,    884.03589463,
         1015.49057792,   1166.49235637,   1339.94785088,   1539.19589208,
         1768.07178925,   2030.98115583,   2332.98471274,   2679.89570175,
         3078.39178417,   3536.14357851,   4061.96231167,   4665.96942547,
         5359.79140351,   6156.78356833,   7072.28715702,   8123.92462333,
         9331.93885094,  10719.58280701,  12313.56713667,  14144.57431404,
        16247.84924667,  18663.87770189,  21439.16561402,  24627.13427334,
        28289.14862808,  32495.69849334,  37327.75540378,  42878.33122805,
        49254.26854668,  56578.29725615,  64991.39698667,  74655.51080755,
        85756.6624561 ,  98508.53709335])
    
    idx_cut2000 = array([ 0,  2,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  21])
    xbins=[ell_arr2048[idx_cut2000],PDFbin_arr[3][1:],peak_bins_arr[3][4:-4]]
    idx_arr = [arange(18), arange(18,18+100),arange(118+3,143-4)] #arange(118,143)]
    
    f=figure(figsize=(6,8))
    for i in arange(3):
        x = xbins[i]
        y = ips_mean[idx_arr[i]]
        ystd = ips_std[idx_arr[i]]/y
        yinterp = ips_interp[idx_arr[i]]
        if i==1:
            x=x[::2]
            y=mean(y.reshape(-1,2),axis=-1)
            ystd=mean(ystd.reshape(-1,2),axis=-1)
            yinterp=mean(yinterp.reshape(-1,2),axis=-1)
        
        ax=f.add_subplot(3, 1, i+1)
        
        ax.errorbar(x, zeros(len(x)),ystd,lw=1,ls='--',color='k')
        ax.plot(x, yinterp/y-1, 'k',lw=2)

        #ax.locator_params(axis = 'x', nbins = 4)
        ax.locator_params(axis = 'y', nbins = 5)
        ax.set_ylim(-0.06,0.06)
        if i ==0:
            
            ax.set_xscale('log')
            ax.set_xlabel('$\ell$',fontsize=22)
            ax.set_ylabel(r'$\Delta C_{\ell} / C_\ell$',fontsize=20)
            ax.set_xlim(100,2e3)
        elif i==1:
            ax.set_xlabel('$\kappa$',fontsize=22)
            ax.set_ylabel(r'$\Delta\rm{PDF} / {\rm{PDF}}$',fontsize=20)
            ax.set_xlim(x[0],x[-1])
        else:
            ax.set_xlabel(r'$\kappa$',fontsize=22)
            ax.set_ylabel(r'$\Delta N_{\rm peaks}/N_{\rm peaks}$',fontsize=20)
            ax.set_xlim(x[0],x[-1])
            ax.set_ylim(-0.14,0.141)
        ax.tick_params(labelsize=16)
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4,bottom=0.1,right=0.96)#hspace=0.2,wspace=0, left=0.18, right=0.95)
    #show()
    savefig(CMBNG_dir+'plot_official/plot_interp.pdf')
    close()

if plot_skewness:
    skew_GRF_mean = array([-0.29340457, -0.33032634, -0.75293509, -0.4970827 , -0.3324177 ])
    skew_GRF_std = array([ 0.0718468 ,  0.07587833,  0.11654383,  0.17339482,  0.22071048])
    labels = [r"$%s'$"%(sigmaG) for sigmaG in [0.5, 1.0, 2.0, 5.0, 8.0]]
    
    skews = load(CMBlensing_dir+'CMBL_skewness.npy')
    skews_mean = mean(skews, axis=0)
    skews_std = std(skews, axis=0)
    
    f=figure(figsize=(7,8))
    ax=f.add_subplot(211)
    ax2=f.add_subplot(212)
    
    ax.errorbar(arange(5), skews_mean[0],skews_std[0],capsize=0, fmt= '^',lw=1.5, mfc='deepskyblue',mec='deepskyblue',label=r'${\rm noiseless \; \kappa}$',ms=10,color='deepskyblue')
    ax.errorbar(arange(5)-0.03, skews_mean[1], skews_std[1],capsize=0, fmt='d',lw=1.5, mec='orangered',mfc='orangered',label=r'${\rm noisy \;\kappa}$',ms=10,color='orangered')
    ax.errorbar(arange(5)+0.03, skew_GRF_mean, skew_GRF_std,capsize=0, fmt= 'o',lw=1.5, mec='darkorchid',mfc='darkorchid',label=r'${\rm noisy \;{\rm GRF}}$',ms=10,color='darkorchid')
    ax.plot((-1,6),(0,0),'k--')
    
    
    ax.set_ylabel(r'$S$', fontsize=20)
    ax.set_ylim(-1,1.3)
    ax.set_xlim(-0.7, 4.5)
    ax.legend(frameon=1,fontsize=18,ncol=1,numpoints=1)
    
    ax2.plot(arange(5), skews_mean[1]/skew_GRF_mean-1, 'd',lw=1.5, mec='orangered',mfc='orangered',label=r'${\rm noisy \;(\kappa)}$',ms=10,color='orangered')#capsize=0,
    ax2.set_ylabel(r'$S^{\kappa}_{\rm noisy}/S^{\rm GRF}_{\rm noisy}-1$', fontsize=20)
    ax2.set_xlabel(r'${\rm smoothing\;\;scale}$', fontsize=22)
    ax2.set_xlim(-0.7, 4.5)
    ax2.locator_params(axis = 'y', nbins = 3)
    ax2.set_ylim(-0.041, 0.003)
    ax.tick_params(axis='y', labelsize=16)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.set_yticks([-0.04, -0.03,-0.02,-0.01])
    plt.xticks(range(5), labels, rotation=15, fontsize=18)
    plt.subplots_adjust(hspace=0.05, wspace=0., left=0.18, right=0.96, bottom=0.13, top=0.96)
    #show()
    
    
    savefig(CMBNG_dir+'plot_official/plot_skewness3.pdf')
    close()
 
