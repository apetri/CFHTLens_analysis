##########################################################
### This code is for Jia's thesis project B. It does the following:
### 1) organizes the CFHT catalogue to 4 Wx fields, with proper 
### format that's easy to use in the future
### 2) pick out a random redshift and 2 peak redshift
### 3) converts RA DEC to (x, y) radian, using Genomonic projection,
### centered at map center
### 4) final products: convergence maps and galcount maps split in 
### different redshift bins

import numpy as np
from scipy import *
from pylab import *
import os
import WLanalysis
#from emcee.utils import MPIPool
#from multiprocessing import Pool

cat_dir='/home1/02977/jialiu/CFHT_cat/'
#cat_dir = '/Users/jia/CFHTLenS/catalogue/'
split_dir = cat_dir+'split/'
W_dir = lambda Wx: cat_dir+'W%s/'%(Wx) #dir for W1..W4 field
splitfiles = os.listdir(split_dir)

zbins = array([0.4, 0.5, 0.6, 0.7, 0.85, 1.3])#arange(0.3,1.35,0.1)
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])
sigmaG_arr = (0.1,)#(0.5, 1, 1.8, 3.5, 5.3, 8.9)

############################################################
########## calculate map size ##############################
#RA1 =(30.0, 39.0)#starting RA for W1
#DEC1=(-11.5,-3.5)
#RA2 =(132.0, 137.0)
#DEC2=(-6.0,-0.5)
#RA3 =(208.0, 221.0)
#DEC3=(51.0, 58.0)
#RA4 =(329.5, 336.0)
#DEC4=(-1.2, 5.0)
#RAs=(RA1,RA2,RA3,RA4)
#DECs=(DEC1,DEC2,DEC3,DEC4)
###dpp=0.0016914558667664816#degrees per pixel = sqrt(12)/2048
#dpp = 0.0067658234670659265#degrees per pixel = sqrt(12)/512
#xnum = lambda RA: round((amax(RA)-amin(RA))/dpp+1)
#ynum = lambda DEC:round((amax(DEC)-amin(DEC))/dpp+1)
##sized calculated using above 3 lines:
## W1 1331, W2 814, W3 1922, W4 962
sizes = (1330, 800, 1120, 950)
############################################################

z_arr = arange(0.025,3.5,.05)
idx2z = lambda idx:z_arr[idx]
field2int = lambda str: int(str[1])

#DrawRedshifts = lambda iPz: concatenate([[z_arr[argmax(iPz)],], WLanalysis.DrawFromPDF(z_arr, iPz, 2)])
DrawRedshifts = lambda iPz: WLanalysis.DrawFromPDF(z_arr, iPz, 1)

def list2coords(radeclist, Wx): 
	'''Input: radeclist = (Wfield, ra, dec), a 3xN matrix, (ra, dec) in degrees
	Return: (subfield, x, y), a 3xN matrix, (x, y) in radians
	'''
	xy = zeros(shape = radeclist.shape) #create xy list
	center = centers[Wx-1] #find the center for Wx field
	f_Wx = WLanalysis.gnom_fun(center)
	xy = degrees(array(map(f_Wx,radeclist)))
	return xy

def OrganizeSplitFile(ifile):
	'''read in one of the split file, pick out the redshift, and sort by fields, e2 is c2 correted, with e2-=c2'''	
	
	field = genfromtxt(split_dir+ifile,usecols=0,dtype=str)
	field = array(map(field2int,field))
	print ifile

	# generate 2 random redshift and 1 peak
	Pz = genfromtxt(split_dir+ifile,usecols=arange(14,84),dtype=str)
	Pz = (np.core.defchararray.replace(Pz,',','')).astype(float)
	
	seed(99)
	z_rand1 = array(map(DrawRedshifts,Pz)).ravel()
	seed(88)
	z_rand2 = array(map(DrawRedshifts,Pz)).ravel()
	z_peak = z_arr[argmax(Pz,axis=1)]
	z_all = concatenate([[z_peak,], [z_rand1,], [z_rand2,]]).T

	sheardata = genfromtxt(split_dir+ifile,usecols=[1,2,5,6,7,8,9,10,11,12,13,84])
	ra, dec, e1, e2, w, fitclass, r, snr, mask, m, c2, mag = sheardata.T
	e2 -= c2
	
	
	i=0
	for Wx in range(1,5):
		idx=where((field==Wx)&(mask<=1.0)&(fitclass==0)&(amin(z_all,axis=-1)>=0.2)&(amax(z_all,axis=-1)<=1.3))[0]
		print ifile, Wx, len(idx)/50000.0
		if len(idx) > 0:
			#data = (np.array([ra,dec,e1,e2,w,r,snr,m,c2,mag]).T)[idx]
			data = (np.array([ra,dec,e1,e2,w,r,snr,m,c2,mag,z_peak, z_rand1, z_rand2]).T)[idx]
			radeclist = sheardata[idx][:,[0,1]]
			xylist = list2coords(radeclist, Wx)
			xy_data = concatenate([xylist,data],axis=1)
			WLanalysis.writeFits(xy_data, W_dir(Wx)+ifile+'.fit')#,fmt=['%i','%i','%s','%s','%s','%.3f'])
		i+=1

def SumSplitFile2Grid(Wx):
	'''For Wx field, read in each split file, 
	and create e1, e2 grid for mass construction.
	Input: Wx=1,2,3,4
	Output: (Me1, Me2, Mw, galn) split in each redshift bins'''
	isize = sizes[Wx-1]
	ishape = (len(zbins), isize, isize)
	ishape_hi = (len(zbins)-1, isize, isize)#no need to do hi for zcut=1.3 since it's everything
	Me1_hi = zeros(shape=ishape_hi)
	Me2_hi = zeros(shape=ishape_hi)#hi is for higher redshift bins, lo is lower redshift
	Mw_hi = zeros(shape=ishape_hi)
	#Mk_hi = zeros(shape=ishape_hi)
	galn_hi = zeros(shape=ishape_hi)

	Me1_lo = zeros(shape=ishape)
	Me2_lo = zeros(shape=ishape)
	Mw_lo = zeros(shape=ishape)
	#Mk_lo = zeros(shape=ishape)
	galn_lo = zeros(shape=ishape)
	
	Wfiles = os.listdir(W_dir(Wx))#get the list of split file for Wx
	for iW in Wfiles:
		datas = WLanalysis.readFits(W_dir(Wx)+iW)
		#cols: x, y, ra, dec, e1, e2, w, r, snr, m, c2, mag, z_peak, z_rand1, z_rand2
		z = datas.T[-3]#z_peak, -2 is z_rand1, -1 is z_rand2
		i = 0 #zbin count
		for zcut in zbins:
			idx0 = where(z<zcut)[0]
			idx1 = where(z>=zcut)[0]
			for idx in [idx0,idx1]:
				y, x, e1, e2, w, m = (datas[idx].T)[[0,1,4,5,6,9]]#note x, y is reversed in python
				k = array([e1*w, e2*w, (1+m)*w])
				x = radians(x)
				y = radians(y)
				print 'W'+str(Wx), iW, 'coords2grid, zbin =',zbins[i]
				A, galn = WLanalysis.coords2grid(x, y, k, size=isize)
				if len(idx)==0:#no need to calculate hi bin for zcut=1.3
					continue
				elif idx[0] == idx0[0]:
					Me1_lo[i] += A[0]
					Me2_lo[i] += A[1]
					Mw_lo[i] += A[2]
					galn_lo[i] += galn
				
				else:
					Me1_hi[i] += A[0]
					Me2_hi[i] += A[1]
					Mw_hi[i] += A[2]
					galn_hi[i] += galn
			i+=1
	print 'Done collecting small fields for W'+str(Wx)
	
	for i in range(len(zbins)):
		for hl in ('lo','hi'):
			Me1_fn = cat_dir+'Me_Mw_galn/W%i_Me1w_%s_%s.fit'%(Wx, zbins[i],hl)
			Me2_fn = cat_dir+'Me_Mw_galn/W%i_Me2w_%s_%s.fit'%(Wx, zbins[i],hl)
			Mw_fn = cat_dir+'Me_Mw_galn/W%i_Mwm_%s_%s.fit'%(Wx, zbins[i],hl)
			galn_fn = cat_dir+'Me_Mw_galn/W%i_galn_%s_%s.fit'%(Wx, zbins[i],hl)
			if hl=='hi' and i==len(zbins)-1:
				continue
			elif hl=='lo':
				WLanalysis.writeFits(Me1_lo[i],Me1_fn, rewrite = True)
				WLanalysis.writeFits(Me2_lo[i],Me2_fn, rewrite = True)
				WLanalysis.writeFits(Mw_lo[i],Mw_fn, rewrite = True)
				WLanalysis.writeFits(galn_lo[i],galn_fn, rewrite = True)
			else:
				WLanalysis.writeFits(Me1_hi[i],Me1_fn, rewrite = True)
				WLanalysis.writeFits(Me2_hi[i],Me2_fn, rewrite = True)
				WLanalysis.writeFits(Mw_hi[i],Mw_fn, rewrite = True)
				WLanalysis.writeFits(galn_hi[i],galn_fn, rewrite = True)


PPA512=2.4633625
def KSmap(iinput):
	'''Input:
	i = ith zbin for zcut
	hl = 'hi' or 'lo' for higher/lower z of the zcut
	sigmaG: smoothing scale
	Wx = 1..4 of the field
	Output:
	smoothed KS map and galn map.
	'''
	Wx, sigmaG, i, hl = iinput
	print 'Wx, sigmaG, i, hl:', Wx, sigmaG, i, hl
	kmap_fn = cat_dir+'KS/W%i_KS_%s_%s_sigmaG%02d.fit'%(Wx, zbins[i],hl,sigmaG*10)
	galn_smooth_fn = cat_dir+'KS/W%i_galn_%s_%s_sigmaG%02d.fit'%(Wx, zbins[i],hl,sigmaG*10)
	
	isfile_kmap, kmap = WLanalysis.TestFitsComplete(kmap_fn, return_file = True)
	if isfile_kmap == False:
		Me1_fn = cat_dir+'Me_Mw_galn/W%i_Me1w_%s_%s.fit'%(Wx, zbins[i],hl)
		Me2_fn = cat_dir+'Me_Mw_galn/W%i_Me2w_%s_%s.fit'%(Wx, zbins[i],hl)
		Mw_fn = cat_dir+'Me_Mw_galn/W%i_Mwm_%s_%s.fit'%(Wx, zbins[i],hl)
		Me1 = WLanalysis.readFits(Me1_fn)
		Me2 = WLanalysis.readFits(Me2_fn)
		Mw = WLanalysis.readFits(Mw_fn)	
		Me1_smooth = WLanalysis.weighted_smooth(Me1, Mw, PPA=PPA512, sigmaG=sigmaG)
		Me2_smooth = WLanalysis.weighted_smooth(Me2, Mw, PPA=PPA512, sigmaG=sigmaG)
		kmap = WLanalysis.KSvw(Me1_smooth, Me2_smooth)
		WLanalysis.writeFits(kmap,kmap_fn)
	isfile_galn, galn_smooth = WLanalysis.TestFitsComplete(galn_smooth_fn, return_file = True)
	if isfile_galn == False:
		galn_fn = cat_dir+'Me_Mw_galn/W%i_galn_%s_%s.fit'%(Wx, zbins[i],hl)
		galn = WLanalysis.readFits(galn_fn)
		galn_smooth = WLanalysis.smooth(galn, sigma=sigmaG*PPA512)
		WLanalysis.writeFits(galn_smooth, galn_smooth_fn)
	#return kmap, galn_smooth

def Bmode(iinput):
	'''Input:
	i = ith zbin for zcut
	hl = 'hi' or 'lo' for higher/lower z of the zcut
	sigmaG: smoothing scale
	Wx = 1..4 of the field
	Output:
	smoothed KS map and galn map.
	'''
	Wx, sigmaG, i, hl = iinput
	print 'Bmode - Wx, sigmaG, i, hl:', Wx, sigmaG, i, hl
	bmap_fn = cat_dir+'KS/W%i_Bmode_%s_%s_sigmaG%02d.fit'%(Wx, zbins[i],hl,sigmaG*10)
	#galn_smooth_fn = cat_dir+'KS/W%i_galn_%s_%s_sigmaG%02d.fit'%(Wx, zbins[i],hl,sigmaG*10)
	
	isfile_kmap, bmap = WLanalysis.TestFitsComplete(bmap_fn, return_file = True)
	if isfile_kmap == False:
		Me1_fn = cat_dir+'Me_Mw_galn/W%i_Me1w_%s_%s.fit'%(Wx, zbins[i],hl)
		Me2_fn = cat_dir+'Me_Mw_galn/W%i_Me2w_%s_%s.fit'%(Wx, zbins[i],hl)
		Mw_fn = cat_dir+'Me_Mw_galn/W%i_Mwm_%s_%s.fit'%(Wx, zbins[i],hl)
		Me1 = WLanalysis.readFits(Me1_fn)
		Me2 = WLanalysis.readFits(Me2_fn)
		Mw = WLanalysis.readFits(Mw_fn)	
		Me1_smooth = WLanalysis.weighted_smooth(Me1, Mw, PPA=PPA512, sigmaG=sigmaG)
		Me2_smooth = WLanalysis.weighted_smooth(Me2, Mw, PPA=PPA512, sigmaG=sigmaG)
		### Bmode conversion is equivalent to
		### gamma1 -> gamma1' = -gamma2
		### gamma2 -> gamma2' = gamma1
		bmap = WLanalysis.KSvw(-Me2_smooth, Me1_smooth)
		WLanalysis.writeFits(bmap,bmap_fn)
	#return bmap
def Noise(iinput):
	'''Input: (Wx, iseed)
	Return: files of noise KS map, using randomly rotated galaxy.
	'''
	Wx, iseed = iinput
	seed(iseed)
	print 'Bmode - Wx, iseed:', Wx, iseed
	bmap_fn = cat_dir+'Noise/W%i/W%i_Noise_sigmaG10_%04d.fit'%(Wx, Wx, iseed)
	
	isfile_kmap, bmap = WLanalysis.TestFitsComplete(bmap_fn, return_file = True)
	if isfile_kmap == False:
		Me1_fn = cat_dir+'Me_Mw_galn/W%i_Me1w_1.3_lo.fit'%(Wx)
		Me2_fn = cat_dir+'Me_Mw_galn/W%i_Me2w_1.3_lo.fit'%(Wx)
		Mw_fn = cat_dir+'Me_Mw_galn/W%i_Mwm_1.3_lo.fit'%(Wx)
		Me1_init = WLanalysis.readFits(Me1_fn)
		Me2_init = WLanalysis.readFits(Me2_fn)
		#### randomly rotate Me1, Me2 ###
		Me1, Me2 = WLanalysis.rndrot(Me1_init, Me2_init)
		#################################
		Mw = WLanalysis.readFits(Mw_fn)	
		Me1_smooth = WLanalysis.weighted_smooth(Me1, Mw, PPA=PPA512, sigmaG=sigmaG)
		Me2_smooth = WLanalysis.weighted_smooth(Me2, Mw, PPA=PPA512, sigmaG=sigmaG)
		bmap = WLanalysis.KSvw(Me1_smooth, Me2_smooth)
		WLanalysis.writeFits(bmap,bmap_fn)

plot_dir = '/Users/jia/CFHTLenS/plot/obsPK/'
def plotimshow(img,ititle,vmin=None,vmax=None):		 
	 #if vmin == None and vmax == None:
	imgnonzero=img[nonzero(img)]
	if vmin == None:
		std0 = std(imgnonzero)
		x0 = median(imgnonzero)
		vmin = x0-3*std0
		vmax = x0+3*std0
	im=imshow(img,interpolation='nearest',origin='lower',aspect=1,vmin=vmin,vmax=vmax)
	colorbar()
	title(ititle,fontsize=16)
	savefig(plot_dir+'%s.jpg'%(ititle))
	close()	
	
test_dir = '/Users/jia/CFHTLenS/obsPK/'
def TestCrossCorrelate (Wx, zcut, sigmaG):
	'''Input: 
	Wx - one of the W1..W4 field (= 1..4) 
	zcut - redshift cut between KS background galaxies and forground cluster probe
	sigmaG - smoothing
	Output:
	ell_arr, CCK, CCB
	'''
	galn_hi = WLanalysis.readFits(test_dir+'W%i_galn_%s_hi_sigmaG%02d.fit'%(Wx,zcut,sigmaG*10))
	galn_lo = WLanalysis.readFits(test_dir+'W%i_galn_%s_lo_sigmaG%02d.fit'%(Wx,zcut,sigmaG*10))
	galn_cut = 0.5*0.164794921875 #5gal/arcmin^2*arcmin^2/pix, arcmin/pix = 12.0*60**2/512.0**2 = 
	bmap = WLanalysis.readFits(test_dir+'W%i_Bmode_%s_hi_sigmaG%02d.fit'%(Wx,zcut,sigmaG*10))
	kmap = WLanalysis.readFits(test_dir+'W%i_KS_%s_hi_sigmaG%02d.fit'%(Wx,zcut,sigmaG*10))
	mask = where(galn_hi<galn_cut)
	bmap[mask]=0
	kmap[mask]=0
	edges=linspace(5,100,11)
	ell_arr, CCB = WLanalysis.CrossCorrelate (bmap,galn_lo,edges=edges)
	ell_arr, CCK = WLanalysis.CrossCorrelate (kmap,galn_lo,edges=edges)
	f=figure(figsize=(8,6))
	ax=f.add_subplot(111)
	ax.plot(ell_arr, CCB, 'ro',label='B-mode')
	ax.plot(ell_arr, CCK, 'bo', label='KS')
	legend()
	#ax.set_xscale('log')
	ax.set_xlabel('ell')
	ax.set_ylabel(r'$\ell(\ell+1)P_{n\kappa}(\ell)/2\pi$')
	ax.set_title('W%i_zcut%shi_sigmaG%02d'%(Wx,zcut,sigmaG*10))
	#show()
	savefig(plot_dir+'CC_edges_W%i_zcut%shi_sigmaG%02d.jpg'%(Wx,zcut,sigmaG*10))
	close()
	#plotimshow(kmap,'kmap_W%i_zcut%shi_sigmaG%02d.jpg'%(Wx,zcut,sigmaG*10))
	#plotimshow(bmap,'bmap_W%i_zcut%shi_sigmaG%02d.jpg'%(Wx,zcut,sigmaG*10))
	#plotimshow(galn_lo,'galn_W%i_zcut%shi_sigmaG%02d.jpg'%(Wx,zcut,sigmaG*10))
		
Wx_sigmaG_i_hl_arr = [[Wx, sigmaG, i, hl] for Wx in range(1,5) for sigmaG in sigmaG_arr for i in range(0,len(zbins)-1) for hl in ['hi','lo']]+[[Wx, sigmaG, -1, 'lo'] for Wx in range(1,5) for sigmaG in sigmaG_arr]

################################################
###(1) split file organizing ###################
###    uncomment next 1 line ###################
#pool.map(OrganizeSplitFile,splitfiles)
################################################
###(2) sum up the split file into 4 Wx fields###
###    uncomment next 2 line ###################
#for Wx in range(1,5):
	#SumSplitFile2Grid(Wx)
################################################
###(3) create KS maps for 6 zbins 6 sigmaG #####
###    total should have 528 files (galn, KS)###
###    uncomment next 1 line ###################
#map(KSmap, Wx_sigmaG_i_hl_arr[::-1])
################################################
###(4) B mode for picking out signals
###    use 1000 maps with galaxies randomly
###    rotated
###    uncomment the next 1 line
#map(Bmode, Wx_sigmaG_i_hl_arr)
################################################
###(5) Create Noise KS maps by randomly rotate
###    galaxies (2014/09/09)
#noise_input_arr =[[Wx, iseed] for Wx in range(1,5) for iseed in range(200,500)]
#map(Noise, noise_input_arr)

################################################
###(6) cross corrrelation
###    put mask on KS map, and cross correlate
###    for both B-mode(for compare), and true KS
### test on sigmaG=1.0, zcut=0.85	
#Wx=1
#for zcut in zbins[:-1]:
	#for sigmaG in sigmaG_arr:
		#print 'Wx, zcut, sigmaG',Wx, zcut, sigmaG
		#TestCrossCorrelate (Wx, zcut, sigmaG)
################################################
###(7) organize the Wx file, with 
concWx = lambda Wx: array([WLanalysis.readFits(W_dir(Wx)+iW) for iW in os.listdir(W_dir(Wx))])

#y, x, ra, dec, e1, e2, w, r, snr, m, c2, mag, z_peak, z_rand1, z_rand2


print 'DONE-DONE-DONE'
