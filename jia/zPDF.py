#! /afs/rhic.bnl.gov/@sys/opt/astro/SL64/anaconda/bins
# /vega/astro/users/jl3509/tarball/anacondaa/bin/python

from WLanalysis import *
from emcee.utils import MPIPool
import os

x2 = SIMz
step = (x2[1:]-x2[:-1])/2
binwidth = (step[1:]+step[:-1])/2
midedges = array([x2[1:-1]-binwidth,x2[1:-1]+binwidth]).T
leftedge = [2*x2[0]-midedges[0,0],midedges[0,0]]
rightedge= [midedges[-1,-1], 3.5] # the last bin has edges [2.03957899, 3.5] to cover possible high redshifts
edges = concatenate(([leftedge],midedges,[rightedge]))

CFHTzbins = arange(0.0,3.525,.05) 
edges =  append(edges[:,0],[3.5,])
binwidth=edges[1:]-edges[:-1]
SIMbins = edges[:-1]+binwidth/2
w = sum(binwidth)/binwidth/70
full_dir='/direct/astro+astronfs01/workarea/jia/CFHT/full_subfields/'
def pdfs(i):#subfield count
	fn='/direct/astro+astronfs01/workarea/jia/CFHT/full_subfields/zcut_full_subfield%i.fit'%(i)
	if os.path.isfile(fn):
		m=WLanalysis.readFits(fn)
	else:
		print i
		idx=readFits(full_dir+'zcut_idx_subfield%i.fit'%(i)) 
		m=genfromtxt('/direct/astro+astronfs01/workarea/jia/CFHT/full_subfields/full_subfield%s'%(i))[idx]
		print 'done genfromtxt', i
		WLanalysis.writeFits(m, fn) 
		print 'done writeFits', i
	pk=m[:,2]
	rndz=m[:,3:5].flatten()
	pk=histogram(pk,bins=edges)[0]/float(len(pk))
	rndz=histogram(rndz,bins=edges)[0]/float(len(rndz))
	all_PDF=m[:,range(89-70,89)]
	all_PDF/=sum(all_PDF,axis=1)[:,newaxis]
	average_PDF=average(all_PDF,axis=0) 
	savetxt(full_dir+'avgPDF_arr%s'%(i),average_PDF)
	savetxt(full_dir+'pks_arr%s'%(i),pk)
	savetxt(full_dir+'rnds_arr%s'%(i),rndz)
	# for CFHT each bin has 3.5/70 probability, for SIM, each bin has 3.5/67 height
	return all_PDF, pk, rndz, len(all_PDF)

avgPDF_arr = zeros(shape=(13,70))
pks_arr = zeros(shape=(13,67))
rnds_arr =  zeros(shape=(13,67))
lena_arr = zeros(13)

p = MPIPool()
x = p.map(pdfs,range(1,14))

for i in range(1,14):
	aa, peaks, rnds, lena=x[i-1]
	avgPDF_arr_arr[i-1]=aa
	pks_arr[i-1]=peaks
	rnds_arr[i-1]=rnds
	lena_arr[i-1]=lena
	print i, lena

savetxt('/direct/astro+astronfs01/workarea/jia/CFHT/full_subfields/avgPDF_arr.ls',avgPDF_arr)
savetxt('/direct/astro+astronfs01/workarea/jia/CFHT/full_subfields/pks_arr.ls',pks_arr)
savetxt('/direct/astro+astronfs01/workarea/jia/CFHT/full_subfields/rnds_arr.ls',rnds_arr)
savetxt('/direct/astro+astronfs01/workarea/jia/CFHT/full_subfields/lena_arr.ls',lena_arr)
print 'done-done-done'