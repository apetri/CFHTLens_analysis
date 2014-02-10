#!/vega/astro/users/jl3509/tarball/anacondaa/bin/python
# Jia Liu 2014/2/9
# Overview: this code chops the 4 CFHT fields into 13 subfields 
# that fits to maps size of 2048x2048 pix, 3.46x3.46deg^2
#
# Input: 
# 1) description for subfields centers
# 2) CFHT catalogues, splitted to 130000 lines per file with the 
# following columns (total 85 fields):
#
# 0. field   
# 1. ALPHA_J2000     
# 2. DELTA_J2000     
# 3. KRON_RADIUS     
# 4. FLUX_RADIUS
# 5. e1      
# 6. e2      
# 7. weight  
# 8. fitclass        
# 9. scalelength     
# 10. SNratio 
# 11. MASK    
# 12. m       
# 13. c2      
# 14 - 83 (70 columns). PZ_full - 70 columns sampling P(z) at intervals of dz=0.05. bin centers: x=arange(0.025,3.5,.05)
# 84. MAG_i
# 1,2,3,4,5,6,7,8,9,10,11,12,13,84

# Operations:
# 1) organize the file format, since pz where separated with ','
# 2) take in (W field, ra, dec), calculate (subfield#, x, y)
# 3) convert CFHT PDF to a new PDF at lensing plane redshifts
# 4) pick out the peak redshift, and random draw 2 redshifts
#
# Output:
# 1) re-formatted 13 subfield catalogues, with (x, y, 3 redshifts) info added
# 2) input for ray tracing code (x, y, 3 redshifts), totally 13 subfields

from WLanalysis import * #this includes scipy *, numpy as np
from multiprocessing import Pool

## directories, etc ####
split_dir = split_dir = '/vega/astro/users/jl3509/CFHT_cat/'
subfields = genfromtxt('/vega/astro/users/jl3509/CFHTLens_analysis/jia/subfieldcenters.ls')
splitfiles = genfromtxt('/vega/astro/users/jl3509/CFHT_cat/files.ls',dtype='S3')

### jia's local directories
#split_dir = split_dir = '/Users/jia/weaklensing/CFHTLenS/catalogue/split/'
#subfields=genfromtxt('/Users/jia/Documents/code/CFHTLens_analysis/jia/subfieldcenters.ls')
sort_subfs=[subfields[arange(7)],subfields[[7,8,9]],subfields[[10,11,12,13]],subfields[[14,15]]]
centers = array([[34.5, -7.5], [134.5, -3.25],[214.5, 54.5],[ 332.75, 1.9]])

## find edges for simulation redshift(SIMz) bins
x2 = SIMz
step = (x2[1:]-x2[:-1])/2
binwidth = (step[1:]+step[:-1])/2
midedges = array([x2[1:-1]-binwidth,x2[1:-1]+binwidth]).T
leftedge = [2*x2[0]-midedges[0,0],midedges[0,0]]
rightedge= [midedges[-1,-1], 3.5] # the last bin has edges [2.03957899, 3.5] to cover possible high redshifts
edges = concatenate(([leftedge],midedges,[rightedge]))

## functions
def DrawRedshifts (P, CFHTz=CFHTz, SIMz=SIMz, edges=edges):
	'''Draw 3 redshifts (peak,) from distribution PDF P, interpolated to newP that's centered at simulation lensing redshifts.
	'''
	newP = InterpPDF(CFHTz, P, SIMz, edges=edges)
	z_peak = SIMz[argmax(newP)]
	# draw 2 random 
	z_rand = DrawFromPDF (SIMz, newP, 2)
	return z_peak, z_rand[0], z_rand[1]

field2int = lambda str: int(str[1]) #e.g. pick out 1 from 'W1p2'
j2s=lambda j:[0,1,2,3,10,11,12,4,11,12,5,6,7,8,9,10][j] #dict, from 16 patches to 13 subfields
	
def list2subfield(radeclist): 
	'''Input: radeclist = (Wfield, ra, dec), a 3xN matrix, (ra, dec) in degrees
	Return: (subfield, x, y), a 3xN matrix, (x, y) in radians
	'''
	xylist = zeros(shape = radeclist.shape)
	j = 0 #subfield count
	for i in range(4): #W field count
		idx = where(radeclist[:,0] == i+1)[0] #find W1 enries
		if len(idx) > 0:
			print 'Found entry for W',i+1
			sublist = radeclist[idx] #pick out W1 entries
			print 'idx',idx
			sort_subf = sort_subfs[i] #get W1 configurations
			center = centers[i] #prepare for x,y calc
			f_Wx = gnom_fun(center)
			xy = degrees(array(map(f_Wx,sublist[:,1:3])))
			print 'xy',xy
			for isort_subf in sort_subf:
				x0,x1,y0,y1 = isort_subf[5:9]
				# find entries for each subfield
				iidx = where((xy[:,0]<x1)&(xy[:,0]>x0)&(xy[:,1]<y1)&(xy[:,1]>y0))[0]
				
				if len(iidx) > 0:
					print 'iidx j=',j,iidx
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

full_fmt=['%.8e',]*5+['%.8e',]*2+['%.5e',]*2+['%.3e',]*2+['%.5e','%i','%.4e','%.2e','%.1f']+['%.5e',]*3+['%.8e',]*70
#ixylist,iPrand,idatas,iPz
# 1. ALPHA_J2000     
# 2. DELTA_J2000     
# 3. KRON_RADIUS     
# 4. FLUX_RADIUS
# 5. e1      
# 6. e2      
# 7. weight  
# 8. fitclass        
# 9. scalelength     
# 10. SNratio 
# 11. MASK    
# 12. m       
# 13. c2 
# 84. MAG_i
# e.g. array([  3.28127916e+01,  -6.94027732e+00,   radius3.69069000e+00,
#         radius3.37342000e+00,   e16.67400000e-01,   e22.21800000e-01,
#         w1.53539000e+01,   class 0.00000000e+00,   scale 3.20810000e+00,
#         3.89000000e+01,   0.00000000e+00,  -1.85604000e-04,
#         2.66558000e-04,   2.26114000e+01])
ray_fmt=['%.8e',]*5
#ixylist,iPrand
def OrganizeSplitFile(ifile):
	field = genfromtxt(split_dir+ifile,usecols=0,dtype=str)
	field = array(map(field2int,field))
	print 'field',field.shape,field

	datas = genfromtxt(split_dir+ifile,usecols=range(1,14)+[84])
	print 'datas',datas.shape,datas

	# generate random P
	Pz = genfromtxt(split_dir+ifile,usecols=arange(14,84),dtype=str)
	Pz = (np.core.defchararray.replace(Pz,',','')).astype(float)
	Prand = array(map(DrawRedshifts,Pz))
	print 'Prand',Prand.shape,datas

	# get subfield, x, y
	radeclist = concatenate((field.reshape(-1,1),datas[:,[0,1]]),axis=1)	
	xylist = list2subfield(radeclist)#a function needs cleaning up
	print 'xylist', xylist.shape, xylist

	subfields = unique(xylist[:,0])
	subfields = delete(subfields, where(subfields==0)[0]).astype(int)

	for isf in subfields: #save to individual files, isf = (1,2,3..13)
		print 'isf',isf
		idx = where(xylist[:,0]==isf)[0]
		#ifield = field[idx]
		idatas = datas[idx]
		iPz = Pz[idx]
		
		ixylist = xylist[idx][:,1:]
		iPrand = Prand[idx]
		
		array_raytrace = concatenate((ixylist,iPrand),axis=1)
		array_full = concatenate((ixylist,iPrand,idatas,iPz),axis=1)
		print 'array_raytrace.shape, array_full.shape',array_raytrace.shape, array_full.shape
		
		savetxt(split_dir+'full_subfield%i_%s'%(isf,ifile),array_full,fmt=full_fmt)
		savetxt(split_dir+'raytrace_subfield%i_%s'%(isf,ifile),array_raytrace,fmt=ray_fmt)

#ifile = str(sys.argv[1])#'xfu'
#OrganizeSplitFile(ifile)
processes = Pool(151)
processes.map(OrganizeSplitFile, splitfiles)

done=rand(5)
savetxt(split_dir+'done.txt',done)