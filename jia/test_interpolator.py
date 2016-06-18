import WLanalysis
import glob, os, sys
import numpy as np
from scipy import *
from scipy import interpolate, stats, fftpack
from scipy import interpolate
import random
from pylab import *
from sklearn.gaussian_process import GaussianProcess

#### stats77 comes from the fact noisy maps are 77 x 77 in size
CMBlensing_dir ='/Users/jia/weaklensing/CMBnonGaussian/'
all_points = genfromtxt(CMBlensing_dir+'model_point.txt')
all_points46 = all_points[where(all_points.T[0]>0.14)]

cosmo_arr = array(['Om%.3f_Ol%.3f_w-1.000_si%.3f'%(cosmo[0],1-cosmo[0], cosmo[1]) for cosmo in all_points46])

all_stats46 = array([ load(CMBlensing_dir+'Pkappa_gadget/noiseless/kappa_%s_ps_PDF_pk_z1100.npy'%(cosmo))    for cosmo in cosmo_arr])

PDF_noiseless46 = array([all_stats46[icosmo][j][1][-1] for icosmo in range(len(all_stats46)) for j in range(1000)]).reshape(46,1000,-1)

peaks_noiseless46 = array([all_stats46[icosmo][j][2][-1] for icosmo in range(len(all_stats46)) for j in range(1000)]).reshape(46,1000,-1)


om_fidu, si8_fidu=all_points[18]
del_om, del_si8 = 0.02, 0.02
om0,om1,si80,si81=om_fidu-del_om, om_fidu+del_om, si8_fidu-del_si8, si8_fidu+del_si8
jjj=30#250
om_arr= linspace(om0,om1,jjj)
si8_arr=linspace(si80,si81, jjj+1)

def buildInterpolator2D(obs_arr, cosmo_params, method='Rbf'):
    m, s = cosmo_params.T
    spline_interps = list()
    for ibin in range(obs_arr.shape[-1]):
        model = obs_arr[:,ibin]
        if method == 'Rbf':
            iinterp = interpolate.Rbf(m, s, model)#
        elif method == 'linear':
            iinterp = interpolate.LinearNDInterpolator(cosmo_params,model)#
        elif method == 'clough':
            iinterp = interpolate.CloughTocher2DInterpolator(cosmo_params,model)#
        spline_interps.append(iinterp)
    def interp_cosmo (params):
        mm, sm = params
        gen_ps = lambda ibin: spline_interps[ibin](mm, sm)
        ps_interp = array(map(gen_ps, range(obs_arr.shape[-1])))
        ps_interp = ps_interp.reshape(-1,1).squeeze()
        return ps_interp
    return interp_cosmo
    
def buildInterpolator2D_GP(ps_avg, ps_std, cosmo_params):
    gp_interps = list()
    for ibin in range(ps_avg.shape[-1]):
        #print ibin
        y = ps_avg[:,ibin]
        dy = ps_std[:,ibin]
        gp = GaussianProcess(corr='squared_exponential', nugget=(dy / y) ** 2, random_start=100)
        gp.fit(cosmo_params, y)
        gp_interps.append(gp)
    def interp_cosmo(params):
        mm, sm = params
        gen_ps = lambda ibin: gp_interps[ibin].predict(params)
        ps_interp = array(map(gen_ps, range(ps_avg.shape[-1])))
        ps_interp = ps_interp.reshape(-1,1).squeeze()
        return ps_interp
    return interp_cosmo

cosmo_params=all_points46
PDF_noiseless46_binned = sum(PDF_noiseless46.reshape(46,1000,50,-1),axis=-1)
ps_avg=mean(PDF_noiseless46_binned,axis=1)
ps_std=std(PDF_noiseless46_binned,axis=1)

PDFbins = linspace(-0.12, 0.12, 50)#101
#interp_GP = buildInterpolator2D_GP(ps_avg, ps_std, cosmo_params)


def test_interp (N):
    ips_avg = delete(ps_avg, N,axis=0)
    ips_std = delete(ps_std, N,axis=0)
    icosmo_params = delete(cosmo_params, N,axis=0)
    fidu_params = cosmo_params[N]
    x,y=fidu_params
    
    ps_GP = buildInterpolator2D_GP(ips_avg, ips_std, icosmo_params)(fidu_params)
    ps_Rbf = buildInterpolator2D(ips_avg, icosmo_params, method='Rbf')(fidu_params)
    ps_linear = buildInterpolator2D(ips_avg, icosmo_params, method='linear')(fidu_params)
    ps_clough = buildInterpolator2D(ips_avg, icosmo_params, method='clough')(fidu_params)
    
    subplot(211)
    errorbar(PDFbins, ps_avg[N], ps_std[N], label='True')
    plot(PDFbins, ps_GP, label='GP')
    plot(PDFbins, ps_Rbf, label='Rbf')
    plot(PDFbins, ps_linear, label='linear')
    plot(PDFbins, ps_clough, label='Clough')
    title(cosmo_arr[N])
    legend(frameon=0)
    xlabel('kappa')
    ylabel('PDF')
    subplot(212)
    errorbar(PDFbins, zeros(50), ps_std[N]/ps_avg[N], label='True')
    plot(PDFbins, ps_GP/ps_avg[N]-1, label='GP')
    plot(PDFbins, ps_Rbf/ps_avg[N]-1, label='Rbf')
    plot(PDFbins, ps_linear/ps_avg[N]-1, label='linear')
    plot(PDFbins, ps_clough/ps_avg[N]-1, label='Clough')
    xlabel('kappa')
    ylabel(r'$\Delta {PDF}/PDF-1$')
    savefig(CMBlensing_dir+"plot/interp_test_%s.png"%(cosmo_arr[N]))
    close()
    #show()
    
#map(test_interp, range(46))
#test_interp(1)