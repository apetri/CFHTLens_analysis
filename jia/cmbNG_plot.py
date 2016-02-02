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
cosmo_arr = genfromtxt('/Users/jia/weaklensing/CMBnonGaussian/cosmo_arr.txt',dtype='string')
fidu_cosmo=cosmo_arr[12]

######### official plot konbs ###########
plot_design = 0
plot_comp_nicaea = 0
plot_noiseless_peaks_PDF = 0
plot_sample_noiseless_noisy_map = 1
plot_noisy_peaks_PDF = 0
plot_reconstruction_noise = 0
plot_corr_mat = 0
plot_contour_PDF_pk = 0

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

    ell_nicaea, ps_nicaea=genfromtxt('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_nicaea/Pkappa_nicaea25_{0}_1100'.format(fidu_cosmo))[33:-5].T
    
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

            ax.locator_params(axis = 'x', nbins = 4)
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
    FTmap = pickle.load(open(CMBNG_dir+'kappaMapTT_10000sims/kappaMap0000TT_3.pkl'))
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
        im=ax.imshow(kmap,vmin=-3*std(kmap_noiseless_8arcmin),vmax=3*std(kmap_noiseless_8arcmin),extent=(0,3.5,0,3.5))
        ax.annotate(r"$\rm{%s}$"%(['noiseless','noisy'][i-1]), xy=(0.05, 0.9), xycoords='axes fraction',fontsize=24)
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
    PDFbins = linspace(-0.12, 0.12, 101)
    PDFbins = WLanalysis.edge2center(PDFbins)
    peak_bins = linspace(-0.06,0.14,26)
    peak_bins = WLanalysis.edge2center(peak_bins)
    ell_gadget = (WLanalysis.edge2center(logspace(log10(1.0),log10(77/2.0),26))*360./sqrt(12.25))

    all_stats77 = load (CMBNG_dir+'Pkappa_gadget/noisy_z1100_stats77_kappa.npy')
    ps_all77 = all_stats77[:,:bins]
    PDF_all77 = all_stats77[:, bins:bins+len(PDFbins)]
    peaks_all77 = all_stats77[:, bins+len(PDFbins):]

    all_stats77_GRF = load(CMBNG_dir+'Pkappa_gadget/noisy_z1100_stats77_GRF.npy')
    ps_all77_GRF = all_stats77_GRF[:,:bins]
    PDF_all77_GRF = all_stats77_GRF[:, bins:bins+len(PDFbins)]
    peaks_all77_GRF = all_stats77_GRF[:, bins+len(PDFbins):]

    all_stats=[[ell_gadget, ps_all77, ps_all77_GRF],[PDFbins, PDF_all77, PDF_all77_GRF],[peak_bins, peaks_all77, peaks_all77_GRF]]
    
    gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
    for i in (1,2):## 1 is PDF, 2 is peaks
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
        ax.set_ylim(amax([ax.get_ylim()[0],2e-4]), amax(mean(z,axis=0))*2)
        ax.tick_params(labelsize=16)
        
        ax2.errorbar(x,mean(y,axis=0)/mean(z,axis=0)-1,std(y,axis=0)*sqrt(12/3e4)/mean(z,axis=0), color='k',fmt='-',capsize=0,label=r'$\kappa\,\rm{maps}$',lw=1)
        ax2.plot(x,zeros(len(x)),'k--')
        ax2.set_xlabel('$\kappa$',fontsize=22)
        ax2.set_ylabel([r'$\rm{\Delta PDF/PDF}$', r'$\rm{\Delta N_{peaks}/ N_{peaks}}$'][i-1],fontsize=20)
        ax2.tick_params(labelsize=16)
        ax2.locator_params(axis = 'y', nbins = 5)
        ax2.set_ylim([-0.25,0.25])
        #ax2.set_ylim([[-0.23,0.13],[-0.25,0.25]][i-1])
        ax2.set_xlim(x[0],x[-1])
        plt.subplots_adjust(hspace=0.05,left=0.15)
        plt.setp(ax.get_xticklabels(), visible=False)
        #show()
        savefig(CMBNG_dir+'plot_official/plot_noisy_%s.pdf'%(['PDF','peaks'][i-1]))
        close()

if plot_reconstruction_noise: 
    mat_GRF=load('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_gadget/GRF_{0}_ps_PDF_pk_z1100.npy'.format(fidu_cosmo))
    mat_kappa=load('/Users/jia/weaklensing/CMBnonGaussian/Pkappa_gadget/kappa_{0}_ps_PDF_pk_z1100.npy'.format(fidu_cosmo))
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
    all_stats77_GRF = load(CMBNG_dir+'Pkappa_gadget/noisy_z1100_stats77_GRF.npy')
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

om_fidu, si8_fidu=cosmo_params[12]
del_om, del_si8 = 0.01, 0.01
om0,om1,si80,si81=om_fidu-del_om, om_fidu+del_om, si8_fidu-del_si8, si8_fidu+del_si8
jjj=250
om_arr= linspace(om0,om1,jjj)
si8_arr=linspace(si80,si81, jjj+1)
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
    for j in range(2):
        PDF=('PDF','peaks')[j]
        PDF2=('PDF','pk')[j]
        
        seed(55)
        X, Y = np.meshgrid(si8_arr, om_arr)
        labels = [r"$\rm{%s\,(%s')}$"%(PDF,sigmaG) for sigmaG in sigmaG_arr[[1,3,4]] ]
        labels.append(r"$\rm{PS}(\ell<2,000)$")
        labels.append(r"$\rm{PS}(\ell<10,000)$")
        lines=[]
        f=figure(figsize=(8,6))
        ax=f.add_subplot(111)
        iextent=[si80,si81,om0,om1]
        for sigmaG_idx in (1,3,4,5):#range(6):
            if sigmaG_idx==5:
                prob=load(CMBNG_dir+'mat/Prob_ps_ell2000.npy')
            elif sigmaG_idx==6:
                prob=load(CMBNG_dir+'mat/Prob_ps_ell10000.npy')
            else:
                prob=load(CMBNG_dir+'mat/Prob_%s_sigmaG%i.npy'%(PDF2,sigmaG_idx))
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
        ax.set_xlim(0.78,0.797)
        plt.subplots_adjust(hspace=0.0,bottom=0.13,right=0.96,left=0.15)
        #show()
       
        savefig(CMBNG_dir+'plot_official/contour_%s.pdf'%(PDF))
        close()













#def cosmo_avg_calc (cosmo,return_mat=0):
        ## index: 50ps+100PDF*5+25peaks*5
        #temp = load(CMBlensing_dir+'Pkappa_gadget/%s_ps_PDF_pk_600b.npy'%(cosmo))
        #big_mat = array([concatenate([temp[i][0], concatenate(temp[i][1]), concatenate(temp[i][2])]) for i in range(len(temp))])
        #if cosmo=='Om0.369_Ol0.631_w-1.000_si0.918':
                #big_mat[331]=big_mat[-1]
        #if return_mat:
                #return big_mat
        #else:
                #return mean(big_mat,axis=0),std(big_mat,axis=0)
        
###pspkavgerr = array(map(cosmo_avg_calc, cosmo_arr))
###pspkPDF_avg, pspkPDF_err = swapaxes(pspkavgerr,0,1)
###save(CMBlensing_dir+'pspkPDF_avg.npy',pspkPDF_avg)
###save(CMBlensing_dir+'pspkPDF_err.npy',pspkPDF_err)
###pspkPDF_fidu_mat = cosmo_avg_calc('Om0.260_Ol0.740_w-1.000_si0.800',return_mat=1)
###save(CMBlensing_dir+'pspkPDF_fidu_mat.npy',pspkPDF_fidu_mat)
###pspkPDF_fidu_mat = cosmo_avg_calc(cosmo_arr[12],return_mat=1)
###save(CMBlensing_dir+'pspkPDF_fidu_mat12.npy',pspkPDF_fidu_mat)

#pspkPDF_avg=load(CMBlensing_dir+'pspkPDF_avg.npy')
#pspkPDF_err=load(CMBlensing_dir+'pspkPDF_err.npy')
#pspkPDF_fidu_mat=load(CMBlensing_dir+'pspkPDF_fidu_mat12.npy')
#pspkPDF_fidu_mean=mean(pspkPDF_fidu_mat,axis=0)
##0-49: power spectrum, 7-34 for ell<1e4, 7-22 for ell<2000
##50-549: PDF
##550-674: peaks
#stats_idx = range(650, 674)
##peaks 2arcmin: range(600, 625)
##peaks 5arcmin:range(626, 647) #range(7,22)+range(450,549)
##PDF8arcmin:range(450,549)
##ps:range(7,22)
#jjj=50
#ititle='peaks (8arcmin), 2e4deg^2'#'ps(ell<2000)+PDF(8arcmin), 2e4deg^2'
#iii=12
#del_om, del_si8 = 0.01, 0.01


#ps_avg = pspkPDF_avg[:,stats_idx]
#ps_err = pspkPDF_err[:,stats_idx]
#ps_fidu_mat = pspkPDF_fidu_mat[:,stats_idx]
#cov_mat = cov(ps_fidu_mat,rowvar=0)/(2e4/3.5**2)
#cov_inv = mat(cov_mat).I
#obs=mean(ps_fidu_mat,axis=0)
#err=std(ps_fidu_mat,axis=0)

##ps_avg_test = delete(ps_avg.copy(),iii,axis=0)
##cosmo_params_test = delete(cosmo_params.copy(),iii,axis=0)
##interp_cosmo = buildInterpolator(ps_avg_test, cosmo_params_test)
#obs = ps_avg[iii]

#from pylab import *
#plot_dir='/Users/jia/weaklensing/CMBnonGaussian/plot/'

##iii=9

#def buildInterpolator(obs_arr, cosmo_params):
        #'''Build an interpolator:
        #input:
        #obs_arr = (points, Nbin), where # of points = # of models
        #cosmo_params = (points, Nparams), currently Nparams is hard-coded
        #to be 3 (om,w,si8)
        #output:
        #spline_interps
        #Usage:
        #spline_interps[ibin](im, wm, sm)
        #'''
        #m, w, s = cosmo_params.T
        #spline_interps = list()
        #for ibin in range(obs_arr.shape[-1]):
                #model = obs_arr[:,ibin]
                ##iinterp = interpolate.LinearNDInterpolator(cosmo_params, model)
                #iinterp = interpolate.Rbf(m, s, model)
                #spline_interps.append(iinterp)
        ##return spline_interps
        #def interp_cosmo (params):
                #'''Interpolate the powspec for certain param.
                #Params: list of 3 parameters = (om, w, si8)
                #method = 'multiquadric'
                #'''
                #mm, sm = params
                #gen_ps = lambda ibin: spline_interps[ibin](mm, sm)
                #ps_interp = array(map(gen_ps, range(obs_arr.shape[-1])))
                #ps_interp = ps_interp.reshape(-1,1).squeeze()
                #return ps_interp
        #return interp_cosmo

############ begin: test interpolator ##############
##for iii in range(46):
        ##print iii
        ##ps_avg_test = delete(ps_avg.copy(),iii,axis=0)
        ##cosmo_params_test = delete(cosmo_params.copy(),iii,axis=0)
        ###interp_cosmo = WLanalysis.buildInterpolator(ps_avg_test, cosmo_params_test)
        ##interp_cosmo = buildInterpolator(ps_avg_test, cosmo_params_test)
        ##f=figure(figsize=(8,6))
        ##ax=f.add_subplot(111)
        ##ax.errorbar(ell_gadget,ps_avg[iii], ps_err[iii],label='true')
        ##ax.plot(ell_gadget,interp_cosmo(cosmo_params[iii][[0,-1]]),'-',label='interpolate')
        ###iparam=cosmo_params[iii].copy()
        ###iparam[1]=-1
        ###ax.plot(ell_gadget,interp_cosmo(iparam),'-',label='interpolate(w=-1)')
        ##leg=ax.legend()
        ##leg.get_frame().set_visible(False)
        ##ax.set_xlabel(r'$\ell$')
        ##ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$')
        ###show()
        ##ax.set_title('om, w, si8='+str(cosmo_params[iii]))
        ##ax.set_xscale('log')
        ##ax.set_yscale('log')
        ##ax.set_xlim(ell_gadget[0],ell_gadget[-1])
        ##savefig(plot_dir+'testinterp_%i.jpg'%(iii))
        ##close()
############### end: test interpolator ########

########### begin: heat map ######
#om_fidu=cosmo_params[iii][0]
#si8_fidu=cosmo_params[iii][2]
#om0,om1,si80,si81=om_fidu-del_om, om_fidu+del_om, si8_fidu-del_si8, si8_fidu+del_si8

#def chisq_grid (obs, interp_cosmo, cov_inv, Om_arr=linspace(om0,om1,jjj), si8_arr=linspace(si80,si81, jjj+1)):
        #heatmap = zeros(shape=(len(Om_arr),len(si8_arr)))
        #for i in range(len(Om_arr)):
                #for j in range(len(si8_arr)):
                        #best_fit = (Om_arr[i], si8_arr[j])
                        ##best_fit = (Om_arr[i], -1, si8_arr[j])
                        #model = interp_cosmo(best_fit)
                        #del_N = np.mat(model - obs)
                        #chisq = float(del_N*cov_inv*del_N.T)
                        #heatmap[i,j] = chisq
        #return heatmap



#interp_cosmo = buildInterpolator(ps_avg, cosmo_params)
#heatmap = chisq_grid (obs, interp_cosmo, cov_inv)
#P=exp(-0.5*heatmap)
#P/=sum(P)

##imshow(heatmap,origin='lower',interpolation='nearest',extent=[si80,si81,om0,om1])
#imshow(P,origin='lower',interpolation='nearest',extent=[si80,si81,om0,om1])
#colorbar()
#scatter(cosmo_params[iii][2],cosmo_params[iii][0])
#title(ititle)
##scatter(0.8, 0.26)
#xlabel('sigma_8')
#ylabel('omega_m')
#show()
############ end: head map ##############


#cosmo_params[9]=([ 0.286, -1.272,  1.104])

############ plot on local laptop ##############

#mat_kappa=load('PDF_pk_600b_kappa.npy')
#mat_GRF=load('PDF_pk_600b_GRF.npy')

#for i in arange(len(sigmaG_arr)):
        #ax=f.add_subplot(5,2,i*2+1)
        #ax2=f.add_subplot(5,2,i*2+2)
        #iPDF_kappa = array([mat_kappa[x][0][i] for x in range(1024)])
        #ipeak_kappa = array([mat_kappa[x][1][i] for x in range(1024)])
        #iPDF_GRF = array([mat_GRF[x][0][i] for x in range(1024)])
        #ipeak_GRF = array([mat_GRF[x][1][i] for x in range(1024)])

        #mean_pk_GRF = mean(ipeak_GRF,axis=0)
        #mean_pk_kappa = mean(ipeak_kappa,axis=0)
        ##idx_pk = nonzero(mean_pk_GRF)[0]
        #idx_pk = nonzero(mean_pk_kappa)[0]
        
        #mean_PDF_GRF = mean(iPDF_GRF,axis=0)
        #mean_PDF_kappa = mean(iPDF_kappa,axis=0)
        ##idx_PDF = where(mean_PDF_GRF>5e-4)[0]
        #idx_PDF = where(mean_PDF_kappa>5e-4)[0]
        
        ##covI_PDF = mat(cov(iPDF_GRF[:,idx_PDF]/sqrt(3e4/12),rowvar=0)).I
        ##covI_peak = mat(cov(ipeak_GRF[:,idx_pk]/sqrt(3e4/12),rowvar=0)).I
        
        #covI_PDF = mat(cov(iPDF_kappa[:,idx_PDF]/sqrt(3e4/12),rowvar=0)).I
        #covI_peak = mat(cov(ipeak_kappa[:,idx_pk]/sqrt(3e4/12),rowvar=0)).I

        #dN_PDF = mat((mean_PDF_GRF-mean_PDF_kappa)[idx_PDF])
        #dN_pk =  mat((mean_pk_GRF-mean_pk_kappa)[idx_pk])
        #chisq_PDF = dN_PDF*covI_PDF*dN_PDF.T
        #chisq_peak = dN_pk*covI_peak*dN_pk.T
        #print 'sigmaG = %.1f arcmin, SNR(PDF) = %.2f, SNR(peaks) = %.2f' % (sigmaG_arr[i], sqrt(chisq_PDF), sqrt(chisq_peak))
############# test plots ######################

##a=WLanalysis.readFits('/Users/jia/Documents/weaklensing/map_conv_shear_sample/WL-conv_m-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.798_4096xy_0001r_0029p_0100z_og.gre.fit')

##b=GRF_Gen(a)

##ell, ps_a = WLanalysis.PowerSpectrum(a)
##ell, ps_b = WLanalysis.PowerSpectrum(b)

##from pylab import *
##print 'hi'
##f=figure()
##ax=f.add_subplot(111)
##ax.plot(ell, ps_a,'b-',lw=2,label='Convergence Map')
##ax.plot(ell, ps_b,'r--',lw=2,label='Gaussian Random Field')
##ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$',fontsize=20)
##ax.set_xscale('log')
##ax.set_yscale('log')
##ax.set_xlabel(r'$\ell$',fontsize=20)
##ax.set_xlim(ell[6],ell[-5])
##ax.set_ylim(3e-6,1e-3)
##leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=0)
##leg.get_frame().set_visible(False)
##show()

##asmooth=WLanalysis.smooth(a, 2.5*4)
##bsmooth=WLanalysis.smooth(b, 2.5*4)
#f=figure()
#xxx=0.01
#subplot(121)
#imshow(asmooth, vmin=-2*xxx, vmax=8*xxx)
##title('Convergence Map',fontsize=20)
#colorbar()
#subplot(122)
#imshow(bsmooth, vmin=-2*xxx, vmax=8*xxx)
##title('Gaussian Random Field',fontsize=20)
#colorbar()
#show()

######################################################
############### J U N K ##############################
######################################################

#img = a.astype(float)
#e1, ps_kappa = WLanalysis.PowerSpectrum(a)
#GRF = GRF_Gen (ell_arr_center, psd1D0, size)
#e1, ps_GRF = WLanalysis.PowerSpectrum(GRF)

#loglog(e1, ps_kappa, label='kappa')
##loglog(e1, ps_GRF, label='GRF')
##xxxx=array([WLanalysis.PowerSpectrum(GRF_Gen (ell_arr_center, psd1D0, size))[1] for i in range(5)])
#loglog(e1,mean(xxxx,axis=0), label='GRF')
#legend()
#xlabel('ell')
#ylabel('ell**2*P')
#show()

#subplot(121)
#imshow(a,vmin=-.06,vmax=.06)
#title('kappa')
#colorbar()
#subplot(122)
#imshow(xxx,vmin=-.06,vmax=.06)
#title('GRF')
#colorbar()
#show()

#b300_dir = '/work/02918/apetri/kappaCMB/Om0.260_Ol0.740_Ob0.046_w-1.000_ns0.960_si0.800/1024b300/Maps/'
##Pixels on a side: 2048
##Pixel size: 2.98828125 arcsec
##Total angular size: 1.7 deg
##lmin=2.1e+02 ; lmax=3.1e+05


#ell600 = array([110.50448683,    126.93632224,    145.81154455,    167.49348136,
          #192.39948651,    221.00897366,    253.87264448,    291.62308909,
          #334.98696272,    384.79897302,    442.01794731,    507.74528896,
          #583.24617818,    669.97392544,    769.59794604,    884.03589463,
         #1015.49057792,   1166.49235637,   1339.94785088,   1539.19589208,
         #1768.07178925,   2030.98115583,   2332.98471274,   2679.89570175,
         #3078.39178417,   3536.14357851,   4061.96231167,   4665.96942547,
         #5359.79140351,   6156.78356833,   7072.28715702,   8123.92462333,
         #9331.93885094,  10719.58280701,  12313.56713667,  14144.57431404,
        #16247.84924667,  18663.87770189,  21439.16561402,  24627.13427334,
        #28289.14862808,  32495.69849334,  37327.75540378,  42878.33122805,
        #49254.26854668,  56578.29725615,  64991.39698667,  74655.51080755,
        #85756.6624561 ,  98508.53709335])

#ell300 = array([227.50923759,     261.33948696,     300.20023877,
           #344.83952045,     396.11658987,     455.01847518,
           #522.67897393,     600.40047754,     689.67904089,
           #792.23317975,     910.03695035,    1045.35794786,
          #1200.80095508,    1379.35808178,    1584.4663595 ,
          #1820.0739007 ,    2090.71589571,    2401.60191017,
          #2758.71616357,    3168.932719  ,    3640.14780141,
          #4181.43179142,    4803.20382034,    5517.43232714,
          #6337.86543799,    7280.29560281,    8362.86358284,
          #9606.40764068,   11034.86465428,   12675.73087598,
         #14560.59120563,   16725.72716569,   19212.81528136,
         #22069.72930855,   25351.46175197,   29121.18241125,
         #33451.45433138,   38425.63056271,   44139.45861711,
         #50702.92350393,   58242.36482251,   66902.90866275,
         #76851.26112542,   88278.91723422,  101405.84700787,
        #116484.72964502,  133805.8173255 ,  153702.52225084,
        #176557.83446844,  202811.69401573])

########## operation on stampede        
#pool = MPIPool()

#out600 = pool.map(compute_PDF_ps, [(fn, 3.5**2) for fn in glob.glob(b600_dir+'*.fits')])
#save(CMBlensing_dir+'out600.npy',out600)

#ps600 = array([out600[i][1] for i in range(len(out600))])
#save(CMBlensing_dir+'ps600.npy',ps600)
#for j in range(len(sigmaG_arr)):
        #PDF600 = array([out600[i][0][j][0] for i in range(len(out600))])
        #mean600 = array([out600[i][0][j][1] for i in range(len(out600))])
        #std600 = array([out600[i][0][j][2] for i in range(len(out600))])
        #save(CMBlensing_dir+'PDF600%02d.npy'%(sigmaG_arr[j]*10),PDF600)
        #save(CMBlensing_dir+'mean600%02d.npy'%(sigmaG_arr[j]*10),mean600)
        #save(CMBlensing_dir+'std600%02d.npy'%(sigmaG_arr[j]*10),std600)


#out300 = pool.map(compute_PDF_ps, [(fn, 1.7**2) for fn in glob.glob(b300_dir+'*.fits')])
#save(CMBlensing_dir+'out300.npy',out300)

#ps300 = array([out300[i][1] for i in range(len(out300))])
#save(CMBlensing_dir+'ps300.npy',ps300)
#for j in range(len(sigmaG_arr)):
        #PDF300 = array([out300[i][0][j][0] for i in range(len(out300))])
        #mean300 = array([out300[i][0][j][1] for i in range(len(out300))])
        #std300 = array([out300[i][0][j][2] for i in range(len(out300))])
        #save(CMBlensing_dir+'PDF300%02d.npy'%(sigmaG_arr[j]*10),PDF300)
        #save(CMBlensing_dir+'mean300%02d.npy'%(sigmaG_arr[j]*10),mean300)
        #save(CMBlensing_dir+'std300%02d.npy'%(sigmaG_arr[j]*10),std300)

#print 'DONE-DONE-DONE'

#######################################
###### local laptop plotting ##########
#######################################

#import matplotlib.pyplot as plt
#from pylab import *

#ell_arr = [ell600, ell300]
#plot_dir = '/Users/jia/Desktop/CMBnonGaussian/plot/'
#i=0
#gaussian = lambda x, mu, sig: np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))/sig/sqrt(2.0*pi)

#f=figure(figsize=(8,6))
#ax=f.add_subplot(111)
#for res in ('600','300'):
        #res_dir = '/Users/jia/Desktop/CMBnonGaussian/b%s/'%(res)
        #ps = load(res_dir+'ps%s.npy'%(res))    
        #ax.errorbar(ell_arr[i], mean(ps,axis=0),std(ps,axis=0), label='Gadget (box size = %s Mpc/h)'%(res))
        #i+=1
        
#ell_nicaea, P_kappa_smith = genfromtxt('/Users/jia/Documents/code/nicaea_2.5/Demo/P_kappa_smithrevised').T

#ell_nicaea, P_kappa_linear = genfromtxt('/Users/jia/Documents/code/nicaea_2.5/Demo/P_kappa_linear').T

#ax.plot(ell_nicaea, P_kappa_smith, label='Nicaea2.5 (smith03)')
#ax.plot(ell_nicaea, P_kappa_linear, label='Nicaea2.5 (linear)')
#ax.set_xlim(ell_arr[0][0],ell_arr[1][-1])
#ax.set_ylim(1e-4, 1e-2)
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_xlabel(r'$\ell$')
#ax.set_ylabel(r'$\ell(\ell+1)\rm{P(\ell)/2\pi}$')
#leg=ax.legend(loc=0)
#leg.get_frame().set_visible(False)
#savefig(plot_dir+'ps_nicaea.jpg')
#close()

#i=0
#for res in ('600','300'):      
        ##f=figure(figsize=(12,8))
        ##for j in range(4):
                ##ax=f.add_subplot(2,2,j+1)
                ##sigmaG = sigmaG_arr[j]
                ##iPDF = load(res_dir+'PDF%s%02d.npy'%(res,sigmaG_arr[j]*10))
                ##imean =mean(load(res_dir+'mean%s%02d.npy'%(res,sigmaG_arr[j]*10)))
                ##istd = mean(load(res_dir+'std%s%02d.npy'%(res,sigmaG_arr[j]*10)))
                ##PDF_center = WLanalysis.edge2center(PDFbin_arr[j])
                ##norm = 1.0/(PDF_center[-1]-PDF_center[-2])
                ##ax.errorbar(PDF_center, mean(iPDF,axis=0)*norm, std(iPDF,axis=0)*norm/sqrt(3e4/12))
                
                ##xbins = linspace(PDFbin_arr[j][0],PDFbin_arr[j][-1], 100)
                ##ax.plot(xbins, gaussian(xbins, imean, istd))
                ##ax.set_xlabel(r'$\kappa$')
                ##ax.set_ylabel('PDF')
                ##ax.annotate('$\sigma_G = %s$'%(sigmaG), xy=(0.05, 0.85),xycoords='axes fraction',color='k',fontsize=16)
                
                ##ax.set_yscale('log')
                ##ax.set_ylim(1e-4, 50)
                ##ax.set_xlim(-0.2, 0.3)
                
        ##savefig(plot_dir+'PDF_log_scaled_b%s.jpg'%(res))
        ##close()
        ##i+=1
        
        
###################
#for j in (0,1):## 0 is PDF, 1 is kappa
#f=figure(figsize=(8,16))
#for i in arange(len(sigmaG_arr)):
        #ax=f.add_subplot(5,2,i*2+1)
        #ax2=f.add_subplot(5,2,i*2+2)
        #iPDF_kappa = array([mat_kappa[x][0][i] for x in range(1024)])
        #ipeak_kappa = array([mat_kappa[x][1][i] for x in range(1024)])
        #iPDF_GRF = array([mat_GRF[x][0][i] for x in range(1024)])
        #ipeak_GRF = array([mat_GRF[x][1][i] for x in range(1024)])
        
        #ax.errorbar(PDFbin_arr[i][1:], mean(iPDF_kappa, axis=0), std(iPDF_kappa, axis=0)/sqrt(3e4/12), label='kappa',)
        #ax.errorbar(PDFbin_arr[i][1:], mean(iPDF_GRF, axis=0), std(iPDF_GRF, axis=0)/sqrt(3e4/12), label='GRF')
        #ax2.errorbar(peak_bins_arr[i][1:], mean(ipeak_kappa, axis=0), std(ipeak_kappa, axis=0)/sqrt(3e4/12), label='kappa')
        #ax2.errorbar(peak_bins_arr[i][1:], mean(ipeak_GRF, axis=0), std(ipeak_GRF, axis=0)/sqrt(3e4/12), label='GRF')
        #ax.set_yscale('log')
        #ax2.set_yscale('log')
        #meanPDF = mean(iPDF_kappa, axis=0)
        #meanPK = mean(ipeak_kappa, axis=0)
        #ax.set_ylim(amax([amin(meanPDF), 1e-7]), amax(meanPDF)*1.5)
        #ax2.set_ylim(amin(meanPK), amax(meanPK)*1.5)
        #ax.set_title('PDF(%.1farcmin)'%(sigmaG_arr[i]))
        #ax2.set_title('peaks(%.1farcmin)'%(sigmaG_arr[i]))
        #if i ==0:
                #leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':16},loc=0)
                #leg.get_frame().set_visible(False)
        #if i == 4:
                #ax.set_xlabel('kappa')
                #ax2.set_xlabel('kappa')
        
#savefig(CMBNG_dir+'plot/plot_noiseless_peaks.jpg')
##savefig(CMBNG_dir+'plot/plot_noiseless_PDF.jpg')
#close()