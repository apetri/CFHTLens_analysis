# Jia Liu 2014/08/09
# Overview: this code use the bad/good field list provided by
# CFHT: http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/community/CFHTLens/README_catalogs_release.txt
# and mask out the bad fields

from WLanalysis import * #this includes scipy *, numpy as np
import WLanalysis
## directories, etc ####
#cat_dir = '/Users/jia/CFHTLenS/catalogue/'
#subfields=genfromtxt('/Users/jia/Documents/code/CFHTLens_analysis/jia/subfieldcenters.ls')
cat_dir='/home1/02977/jialiu/CFHT_cat/'
subfields=genfromtxt('/home1/02977/jialiu/CFHTLens_analysis/jia/subfieldcenters.ls')

goodlist = genfromtxt(cat_dir+('BadFieldsMask/goodfields.txt'),dtype=str)
split_dir = cat_dir+'split/'
sf_dir = lambda sf: cat_dir+'BadFieldsMask/subfield%s/'%(sf) #dir for 1..13 subfield
splitfiles = os.listdir(split_dir)

sort_subfs=[subfields[arange(7)],subfields[[7,8,9]],subfields[[10,11,12,13]],subfields[[14,15]]]
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])

field2int = lambda str: int(str[1]) #e.g. pick out 1 from 'W1p2'
pointings2GB = lambda a: int(a in goodlist)
j2s=lambda j:[0,1,2,3,10,11,12,4,11,12,5,6,7,8,9,10][j] #dict, from 16 patches to 13 subfields
	
def list2subfield(radeclist): 
	'''Input: radeclist = (Wfield, ra, dec), a 3xN matrix, (ra, dec) in degrees
	pointings, an array with N entries, representing one of the 171 pointings.
	Return: 
	1) (subfield, x, y), a 3xN matrix, (x, y) in radians, 
	2) GB list, array of N entries GB=1/0 for good/bad field.
	'''
	xylist = zeros(shape = radeclist.shape)
	
	j = 0 #subfield count
	for i in range(4): #W field count
		idx = where(radeclist[:,0] == i+1)[0] #find W1 enries
		if len(idx) > 0:
			print 'Found entry for W',i+1
			sublist = radeclist[idx] #pick out W1 entries
			#print 'idx',idx
			sort_subf = sort_subfs[i] #get W1 configurations
			center = centers[i] #prepare for x,y calc
			f_Wx = gnom_fun(center)
			xy = degrees(array(map(f_Wx,sublist[:,1:3])))
			#print 'xy',xy
			for isort_subf in sort_subf:
				x0,x1,y0,y1 = isort_subf[5:9]
				# find entries for each subfield
				iidx = where((xy[:,0]<x1)&(xy[:,0]>x0)&(xy[:,1]<y1)&(xy[:,1]>y0))[0]
				
				if len(iidx) > 0:
					#print 'iidx j=',j,iidx
					isublist = sublist[iidx]#subfield
					icenter = isort_subf[-2:]#center for subfield
					f_sub = gnom_fun(icenter)
					xy_sub = array(map(f_sub,isublist[:,1:3]))
					xylist[idx[iidx],0] = j2s(j)+1
					xylist[idx[iidx],1:] = xy_sub					
					if j in (4,9): # needs to turn 90 degrees, counterclock
						iy = degrees(xy_sub.T[0])
						ix =- degrees(xy_sub.T[1])
					else:
						ix = degrees(xy_sub.T[0])
						iy = degrees(xy_sub.T[1])
				j+=1
		else:
			j+=len(sort_subfs[i])
	return xylist

ray_fmt=['%.8e','%.8e','%i']
#ixylist,iPrand
def OrganizeSplitFile(ifile):
	'''Organize original CFHT catalogue file into 13 subfields. 
	Output
	'''
	print ifile
	pointings = genfromtxt(split_dir+ifile,usecols=0,dtype=str)
	GBlist = array(map(pointings2GB,pointings))
	if sum(GBlist) > 0:#process only if there's good fields
		field = array(map(field2int,pointings))
		radec = genfromtxt(split_dir+ifile,usecols=(1,2))
		# get subfield, x, y
		radeclist = concatenate((field.reshape(-1,1),radec),axis=1)	
		xylist = list2subfield(radeclist)#a function needs cleaning up
		#print 'xylist', xylist.shape, xylist

		subfields = unique(xylist[:,0])
		subfields = delete(subfields, where(subfields==0)[0]).astype(int)

		for isf in subfields: #save to individual files, isf = (1,2,3..13)
			print 'isf',isf
			idx = where(xylist[:,0]==isf)[0]
			#ifield = field[idx]	
			ixylist = xylist[idx][:,1:]
			iGB = GBlist[idx]
			array_raytrace = concatenate((ixylist,iGB.reshape(-1,1)),axis=1)	
			savetxt(sf_dir(isf)+'MaskBad_subfield%i_%s'%(isf,ifile),array_raytrace,fmt=ray_fmt)
	else:
		print ifile, 'is all bad fields:', unique(pointings)

mask_fcn = lambda sigmaG, i: '/scratch/02977/jialiu/KSsim/mask/CFHT_mask_ngal5_sigma%02d_subfield%02d.fits'%(sigmaG*10,i)
badmask_fcn = lambda sigmaG, i: '/scratch/02977/jialiu/KSsim/mask/BAD_CFHT_mask_ngal5_sigma%02d_subfield%02d.fits'%(sigmaG*10,i)
sigmaG_arr = (0.5, 1, 1.8, 3.5, 5.3, 8.9)
PPA512=2.4633625
ngal_cut = 5.0
def createBadFieldMask (sf):
	sf_splitfiles = os.listdir(sf_dir(sf))
	genfromtxtA = lambda fn: genfromtxt(sf_dir(sf)+fn)
	datas = map(genfromtxtA,sf_splitfiles)#3 columns: RA, DEC, GB
	datas = concatenate(datas,axis=0)
	idx = where(datas[:,-1]==1)[0]
	datas = datas[idx]
	y, x, k = datas.T
	k, galn = WLanalysis.coords2grid(x, y, array([k,]))
	for sigmaG in sigmaG_arr:
		print 'createBadFieldMask sf, sigmaG:', sf, sigmaG
		Allmask = WLanalysis.readFits(mask_fcn(sigmaG, sf))#mask for all field
		badmask_fn = badmask_fcn(sigmaG, sf)#file name for bad pointing mask, which is 75% area of Allmask
		galn_smooth = snd.filters.gaussian_filter(galn.astype(float),sigmaG*PPA512, mode='constant')
		#smooth the galn grid
		Mmask = ones(shape=galn.shape)#create mask grid
		Mmask[where(galn_smooth < ngal_cut)]=0#find the low density region in galn_smooth
		Mmask *= Allmask#since I didn't do redshift cut in badmask, so here it takes care of it, since ALl mask has redshift cuts
		WLanalysis.writeFits(Mmask, badmask_fn)
	
	#return datas

#map(OrganizeSplitFile, splitfiles):
map(createBadFieldMask, range(1,14))	

print 'Done-Done-Done'