#!/opt/astro/SL64/anaconda/bin/python
# this is a file Jia wrote on 4/17/2013, with 2 purposes: 
#1) convert txt file to fit
#2) re-organize field 11 & 13 which had wrong coordinates for the horizontal patches
#3) also some other admin stuff: get yxew, where w already took into consideration of m correction; raytrace files

from scipy import *
import numpy as np
import WLanalysis
from emcee.utils import MPIPool
import sys

zmin=0.2
zmax=1.3

pool = MPIPool()

ray_fmt=['%.8e',]*5

full_dir = '/direct/astro+astronfs03/workarea/jia/CFHT/CFHT/full_subfields/'

def organizeFit(i):
	'''Organize from txt file to fits, create full_subfield, raytrace_subfield, zcut_idx,
	for 04/18/2014
	'''
	print i
	fn_backup=full_dir+'backup/full_subfield%i'%(i)
	fn = full_dir+'full_subfield%i'%(i)
	fn_ray = full_dir+'raytrace_subfield%i'%(i)
	zcutfn = full_dir+'zcut_idx_subfield%i'%(i)
	fn_yxew = full_dir+'yxew_subfield%i'%(i)#w already included m correction
	
	#fullfile=genfromtxt(fn)
	fullfile=WLanalysis.readFits(fn_backup+'.fit')
	print 'done genfromtxt',i
	if i in (11,13):
		if i ==11:
			idx=where(fullfile[:,5]<300)[0]
		if i ==13:
			idx=where(fullfile[:,5]>40)[0]
		print 'idx',i
		y0 = (fullfile[idx,0]).copy()
		#y0 = (fullfile[idx,1]).copy()
		#y0 = y0-amin(y0)-0.03
		#fullfile[idx,0]=y0
		fullfile[idx,0]=y0-amin(y0)-0.03
	
	zs = fullfile[:, [2, 3, 4]]
	zidx = np.where((amax(zs,axis=1) <= zmax) & (amin(zs,axis=1) >= zmin))[0]
	
	WLanalysis.writeFits(fullfile,fn+'.fit')
	WLanalysis.writeFits(fullfile[zidx],fn+'_zcut0213.fit')
	
	WLanalysis.writeFits(fullfile[:, :5],fn_ray+'.fit')
	WLanalysis.writeFits(fullfile[zidx, :5],fn_ray+'_zcut0213.fit')	

	y, x, e1, e2, w, c2, m = fullfile[:, [0, 1, 9, 10, 11, 17, 16]].T
	e2 = e2-c2	
	w *= (1+m)
	k = array([y,x,e1,e2,w]).T
	WLanalysis.writeFits(k, fn_yxew+'.fit')
	WLanalysis.writeFits(k[zidx], fn_yxew+'_zcut0213.fit')
	
	savetxt(fn_ray,fullfile[:, :5],fmt=ray_fmt)
	savetxt(fn_ray+'_zcut0213',fullfile[zidx, :5],fmt=ray_fmt)
	print 'done writefits',i
	
#pool.map(organizeFit,(11,13))
#savetxt(full_dir+'done0418',zeros(5))

def create_yxewm (i):
	fn = full_dir+'full_subfield%i.fit'%(i)
	fullfile = WLanalysis.readFits(fn)
	fn_yxewm = full_dir+'yxewm_subfield%i'%(i)
	print i
	zidx = WLanalysis.readFits(full_dir+'zcut0213_idx_subfield%i.fit'%(i))
	y, x, e1, e2, w, c2, m = fullfile[:, [0, 1, 9, 10, 11, 17, 16]].T
	e2 = e2-c2
	k = array([y,x,e1,e2,w,m]).T
	WLanalysis.writeFits(k, fn_yxewm+'.fit')
	WLanalysis.writeFits(k[zidx], fn_yxewm+'_zcut0213.fit')
	
pool.map(create_yxewm,range(1,14))
print 'Done-Done-Done-Done'
savetxt(full_dir+'done0501',zeros(5))
sys.exit(0)