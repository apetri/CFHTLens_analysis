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

#CMBlensing_dir = '/work/02977/jialiu/CMBnonGaussian/'
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
plot_contour_noisy, area_scaling_noisy, fsky_deg_noisy = 0, 0, 1000.0

if plot_design:
    all_points=genfromtxt(CMBNG_dir+'model_point.txt')
    idx = argsort(all_points.T[0])[4:]
    om,si8=all_points[idx].T
    f=figure(figsize=(6,6))
    ax=f.add_subplot(111)
    ax.scatter(om,si8,marker='D',color='k')
    ax.scatter(om[12],si8[12], s=300, marker='o',color='red', facecolors='none', edgecolors='r',lw=2)
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
    ell_gadget = (WLanalysis.edge2center(logspace(log10(1.0),log10(1024),51))*360./sqrt(12.25))[:34]

    ell_nicaea, ps_nicaea=genfromtxt('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_nicaea/Pkappa_nicaea25_{0}_1100'.format(cosmo))[33:-5].T
    
    def get_1024(j):
        pspkPDFgadget=load('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_gadget/kappa_{0}_ps_PDF_pk_z1100_{1}.npy'.format(fidu_cosmo, j))
        ps_gadget=array([pspkPDFgadget[i][0] for i in range(len(pspkPDFgadget))])
        ps_gadget=ps_gadget.squeeze()
        return ps_gadget
    ps_gadget=concatenate(map(get_1024, range(10)),axis=0)[:,:34]
    
    idx=where(~isnan(mean(ps_gadget,axis=0)))[0]
    ps_gadget=ps_gadget[:,idx]
    ell_gadget=ell_gadget[idx]
    
    f=figure(figsize=(8,6))
    ax=f.add_subplot(111)
    ax.errorbar(ell_gadget, mean(ps_gadget,axis=0),std(ps_gadget,axis=0),label=r'$\rm{Simulation}$',lw=1.0,color='k')
    ax.plot(ell_nicaea, ps_nicaea,'k--',lw=2,label=r'$\rm{Smith03+Takahashi12}$')
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
    savefig(CMBNG_dir+'plot/plot_theory_comparison.jpg')
    close()
    

if plot_noiseless_peaks_PDF:
    mat_kappa=load('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_gadget/kappa_{0}_ps_PDF_pk_z1100.npy'.format(fidu_cosmo))
    mat_GRF=load('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_gadget/GRF_{0}_ps_PDF_pk_z1100.npy'.format(fidu_cosmo))
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

            #ax.locator_params(axis = 'x', nbins = 4)
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
            locator_params(axis = 'y', nbins = 3)
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
                ax.set_xlim(-x0,x0)

            if i == 4:
                    ax.set_xlabel('$\kappa$',fontsize=22)
            ax.tick_params(labelsize=16)
            ax.set_ylim(-0.5,0.5)
            
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2,wspace=0, left=0.18, right=0.95)
        #show()
        #savefig(CMBNG_dir+'plot/plot_noiseless_%s.jpg'%(['PDF','peaks'][j-1]))
        savefig(CMBNG_dir+'plot_official/plot_noiseless_%s_diff.pdf'%(['PDF','peaks'][j-1]))
        close()

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
    filtered=1
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
        ax.set_yscale('log')
        handles0, labels0 = ax.get_legend_handles_labels()
        handles=[h[0] for h in handles0]
        leg=ax.legend(handles,labels0,ncol=1, prop={'size':24}, loc=8)
        leg.get_frame().set_visible(False)
        ax.set_ylabel([r'$\rm{PDF}$', r'$\rm{N_{peaks}\, (deg^{-2})}$'][i-1],fontsize=22)
        
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
        savefig(CMBNG_dir+'plot_official/plot_noisy_%s%s%s.pdf'%(['ps','PDF','peaks'][i],['','_filtered'][filtered],['','_morebins'][morebins]))
        savefig(CMBNG_dir+'plot_official/png/plot_noisy_%s%s%s.png'%(['ps','PDF','peaks'][i],['','_filtered'][filtered],['','_morebins'][morebins]))
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
            ax.set_xlim(130,2e4)
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
    del_om, del_si8 =0.15, 0.15#0.05, 0.05 # 0.01, 0.01
    if area_scaling:
        del_om /= sqrt(2e4/fsky_deg)
        del_si8/= sqrt(2e4/fsky_deg)
    om0,om1,si80,si81=om_fidu-del_om, om_fidu+del_om, si8_fidu-del_si8, si8_fidu+del_si8
    jjj=100#250
    om_arr= linspace(om0,om1,jjj)
    si8_arr=linspace(si80,si81, jjj+1)

    for imethod in ('clough','Rbf','linear'):#imethod='Rbf'#'clough'#'linear'
        for j in range(2):
            istat=['PDF','Peaks'][j]
            
            seed(55)
            X, Y = np.meshgrid(si8_arr, om_arr)
            labels = [r"$\rm{%s\,(%s')}$"%(istat,sigmaG) for sigmaG in sigmaG_arr[[1,3,4]] ]
            labels.append(r"$\rm{PS}(\ell<2,000)$")
            #labels.append(r"$\rm{PS}(\ell<10,000)$")
            lines=[]
            f=figure(figsize=(8,6))
            ax=f.add_subplot(111)
            iextent=[si80,si81,om0,om1]
            for sigmaG_idx in (1,3,4,5):#range(6):
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
                icolor=rand(3)
                CS=ax.contour(X, Y, prob, levels=[V[0],], origin='lower', extent=iextent,linewidths=2, colors=[icolor, ])
                lines.append(CS.collections[0])

            leg=ax.legend(lines, labels, ncol=1, labelspacing=0.3, prop={'size':18},loc=0)
            leg.get_frame().set_visible(False)
            ax.tick_params(labelsize=16)
            ax.locator_params(axis = 'both', nbins = 5)
            ax.set_xlabel('$\sigma_8$',fontsize=22)
            ax.set_ylabel('$\Omega_m$',fontsize=22)
            ax.grid(True)
            #ax.set_xlim(0.78,0.797)
            #ax.set_xlim(0.782,0.791)#(0.78,0.797)
            #ax.set_ylim(0.29,0.304)
            ax.set_xlim(0.754,0.832)
            ax.set_ylim(0.267,0.333)

            plt.subplots_adjust(hspace=0.0,bottom=0.13,right=0.96,left=0.15)
            #show()
        
            #savefig(CMBNG_dir+'plot_official/contour_noiseless_%s_%s.pdf'%(istat,imethod))
            savefig(CMBNG_dir+'plot_official/png/contour%s_noiseless_%s_%s.png'%(['','_areascale'][area_scaling],istat,imethod))
            close()


if plot_contour_noisy:
    filtered = 0
    noise = 'noisy'#'noiseless'#
    om_fidu, si8_fidu=cosmo_params[12]
    del_om, del_si8 =0.15,0.15#0.05, 0.05 # 0.01, 0.01
    if area_scaling_noisy:
        del_om /= sqrt(2e4/fsky_deg)
        del_si8/= sqrt(2e4/fsky_deg)
    om0,om1,si80,si81=om_fidu-del_om, om_fidu+del_om, si8_fidu-del_si8, si8_fidu+del_si8
    jjj=100#250
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
        
        for istat in ['ps','PDF','peaks','all3']:#['ps','peaks']:#
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
            else:
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
            icolor=rand(3)
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
        #show()
        
        #savefig(CMBNG_dir+'plot_official/contour%s_%s_%s.pdf'%(['','_filtered'][filtered],noise,imethod))
        savefig(CMBNG_dir+'plot_official/png/optimize_contour%s_fsky%i_%s_%s.png'%(['','_filtered'][filtered],fsky_deg_noisy,noise,imethod))
        close()


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
        fidu_stats77 = fidu_stats77[:,~isnan(fidu_stats77[0])]
        cov_mat = cov(fidu_stats77,rowvar=0)
        corr_mat = WLanalysis.corr_mat(cov_mat)
    fig=figure(figsize=(7,6))
    ax=fig.add_subplot(111)
    im=ax.imshow(corr_mat,origin='lower',interpolation='nearest',cmap='PuOr',vmax=1,vmin=-1)
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
    if do_noisy:
        ax.text(0, 148, 'PS',  fontsize=14,
            bbox={'width':80, 'facecolor':'thistle', 'alpha':1})
        ax.text(23, 148, 'PDF', fontsize=14,
            bbox={'width':255, 'facecolor':'lightsalmon', 'alpha':1})
        ax.text(123, 148, 'Peaks',  fontsize=14,
            bbox={'width':60, 'facecolor':'powderblue', 'alpha':1})
    
    else:
        ax.text(0, 98, 'PS',  fontsize=14,
            bbox={'width':90, 'facecolor':'thistle', 'alpha':1})
        ax.text(20, 98, 'PDF', fontsize=14,
            bbox={'width':230, 'facecolor':'lightsalmon', 'alpha':1})
        ax.text(70, 98, 'Peaks',  fontsize=14,
            bbox={'width':95, 'facecolor':'powderblue', 'alpha':1})
    plt.subplots_adjust(hspace=0.0,wspace=0, left=0.04, right=0.97,bottom=0.08,top=0.9)
    #show()
    savefig(CMBlensing_dir+'plot_official/corr_mat%s.pdf'%(['','_noisy'][do_noisy]))
    close()






