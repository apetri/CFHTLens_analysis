cluster_counts = 0
halo_plots = 0
list_peaks_cat = 0 #! generate a list of galaxies for all peaks
project_mass = 0
#junk routines below
update_mag_i = 0
plot_galn_vs_kappa_hist = 0
do_hist_galn_magcut = 0
update_mag_all = 0 #! make a list of galaxie catagues with useful quantities


#cNFW_fcn = lambda zM: 9.0/(1.0+zM[0])*(zM[1]/1.3e13)**(-0.13)#Bullock2001
#cNFW_fcn = lambda zM: 10.0/(1.0+zM[0])*(zM[1]/1e13)**(-0.2)#Takada&Jain2003
#cNFW_fcn = lambda zM: 11.0/(1.0+zM[0])*(zM[1]/1e13)**(-0.13)#Lin&Kilbinger2014
#Rvir = lambda M, z: (M*M_sun/(4.0/3.0*pi*delta_c*rho_mz(z)))**0.3333# free delta c 

########## convert from M100 (get from Lk) to Mvir (needed for NFW) ############

ratio_r100_rvir = lambda gamma, c: (1.0/(c*gamma+1.0)+log(c*gamma+1.0)-1.0)/(1.0/(c+1.0)+log(c+1.0)-1.0) - gamma**3.0*0.5#0.5 = 100/200
ratio_M100_Mvir = lambda c: op.brentq(ratio_r100_rvir, 1e-6, 50, args=(c))**3.0*0.5
# c = 5, M100/Mvir = 1.227
rho_mz = lambda z: OmegaM*rho_c0*(1+z)**3#done, unit g/cm^3

Rvir_fcn = lambda M, z: (M*M_sun/(4.0/3.0*pi*200*rho_mz(z)))**0.3333#set delta_c=178, unit=cm


############################################################################## 

update_weight = 0
if update_weight:
	'''one time use 12/15/2014'''
	color_cat = load(obsPK_dir+'junk/CFHTdata_RA_DEC_ugriyz_2014-12-08T21-58-57.npy')
	RA, DEC, star_flag, weight, MAG_u, MAG_g, MAG_r, MAG_i, MAG_y, MAG_z = color_cat.T
	
	RADEC = RA+1.0j*DEC
	R = 3.0
	for z_lo in (0.5, 0.6, 0.7):
		for noise in (True, False):
			print 'z_lo, noise, R:',',', z_lo,',', noise,',', R
			fn = obsPK_dir+'peaks_IDraDecZ_MAGrziMhalo_dist_zcut%s_R%s_noise%s.npy'%(z_lo, R, noise)
			temp_arr = np.load(fn)
			identifier, ira, idec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance = temp_arr
			iradec = ira+1.0j*idec
			
			###### get only the unique weights and RADEC that're of interest
			idx = where(in1d(RADEC, iradec)==True)[0]
			iRADEC = RADEC[idx]
			iweight = weight[idx]
			#################################

			unique_iradec, idx_inverse = unique(iradec, return_inverse=True)
			weight_arr = iweight[argsort(iRADEC)][idx_inverse]
			new_temp_arr = concatenate([temp_arr,weight_arr.reshape(1,-1)],axis=0)
			#sum(iRADEC[argsort(iRADEC)] - unique_iradec)#=0,pass!
			new_fn = obsPK_dir+'peaks_IDraDecZ_MAGrziMhalo_dist_weight_zcut%s_R%s_noise%s.npy'%(z_lo, R, noise)
			save(new_fn, new_temp_arr)

if update_mag_all:
	## 12/08/2014, code to: 
	## (1) replace Mag_i = -99 items with Mag_y values
	## (2) add ugriz bands to the catalogue
	## (3) convert from MegaCam to SDSS AB system
	color_cat = load(obsPK_dir+'junk/CFHTdata_RA_DEC_ugriyz_2014-12-08T21-58-57.npy')
	RA, DEC, star_flag, weight, MAG_u, MAG_g, MAG_r, MAG_i, MAG_y, MAG_z = color_cat.T
	RADEC = RA+1.0j*DEC
	# merge i and y band, rid of the 99 values
	idx_badi = where(abs(MAG_i)==99)[0]
	MAG_iy = MAG_i.copy()
	MAG_iy[idx_badi]=MAG_y[idx_badi]
	# test # of bad magnitude in i, y, and iy 
	#array([sum(abs(arr)==99) for arr in (MAG_i, MAG_y, MAG_iy)])/7522546.0
	#[963311, 6562757, 3523]
	#[0.128, 0.872, 0.000468]
	
	### convert to SDSS ##############
	### r_SDSS=r_Mega +0.011 (g_Mega - r_Mega)
	### z_SDSS=z_Mega -0.099 (i_Mega - z_Mega)
	r_SDSS=MAG_r + 0.011*(MAG_g - MAG_r)
	z_SDSS=MAG_z - 0.099*(MAG_iy - MAG_z)
	# rz = r_SDSS - z_SDSS # should do after redshift
	idx_badrz = where(amax(abs(array([MAG_g, MAG_r, MAG_iy, MAG_z])), axis=0)==99)[0]
	r_SDSS[idx_badrz] = MAG_r[idx_badrz]
	z_SDSS[idx_badrz] = MAG_z[idx_badrz] # replace bad r_SDSS with MAG_r, in case it's caused by MAG_g
	##################################
	color_cat_reorder = array([weight, MAG_u, MAG_g, MAG_r, MAG_iy, MAG_z, r_SDSS, z_SDSS]).T
	for i in range(1,5):
		print i
		icat = cat_gen_junk(i) #ra, dec, mag_i, z_peak
		iradec = icat.T[0]+1.0j*icat.T[1]
		
		idx = where(in1d(RADEC, iradec)==True)[0]
		if idx.shape[0] != icat.shape[0]:
			print 'Error in shape matching'
		
		iRADEC = RADEC[idx]
		id1 = argsort(iradec)
		id2 = argsort(iRADEC)
			
		### test - the 2 arrays should be identical - pass!
		### iRADEC[id2] - iradec[id1]
		
		icat_new = concatenate([icat[id1][:,[0,1,3]], color_cat_reorder[idx[id2]]], axis=1)
		np.save(obsPK_dir+'W%s_cat_z0213_ra_dec_weight_z_ugriz_SDSSr_SDSSz'%(i), icat_new)
		# columns: ra, dec, z_peak, weight, MAG_u, MAG_g, MAG_r, MAG_iy, MAG_z, r_SDSS, z_SDSS
		
		### test 
		### a=icat[id1][:,-2]
		### b=icat_new[:,-4]
		### sum((a-b)==0) - pass!
	
if plot_galn_vs_kappa_hist:
	Wx=4
	z_lo, z_hi, arcmin = 0.85, '0.4_hi', 3
	#for z_lo in (0.85,):# 0.6, 1.3):
		#for z_hi in ('0.4_hi',):#'0.6_hi'):#'1.3_lo', 
			#for arcmin in (3,):#1.5, 3.0):# 2.0, 
				#print z_lo, z_hi, arcmin
				#allfield_peaks = collect_allfields(z_lo=z_lo, z_hi=z_hi, arcmin=arcmin)
				#allfield_noise = collect_allfields(z_lo=z_lo, z_hi=z_hi, arcmin=arcmin, noise=1)
				#allfield_bmode = collect_allfields(z_lo=z_lo, z_hi=z_hi, arcmin=arcmin, Bmode=1)
				#allfield_bmode_noise = collect_allfields(z_lo=z_lo, z_hi=z_hi, arcmin=arcmin, Bmode=1,noise=1)
				
	allfield_peaks = PeakGaln(Wx, z_lo=z_lo, z_hi=z_hi, arcmin=arcmin)
	allfield_noise = PeakGaln(Wx, z_lo=z_lo, z_hi=z_hi, arcmin=arcmin, noise=1)
	allfield_bmode = PeakGaln(Wx, z_lo=z_lo, z_hi=z_hi, arcmin=arcmin, Bmode=1)
	allfield_bmode_noise = PeakGaln(Wx, z_lo=z_lo, z_hi=z_hi, arcmin=arcmin, Bmode=1,noise=1)
	
	# (Wx, z_lo=0.85, z_hi='1.3_lo', arcmin=2.0, noise=False, Bmode=False)	
	hist_peaks = hist_galn(allfield_peaks)
	hist_noise = hist_galn(allfield_noise)
	hist_Bmode = hist_galn(allfield_bmode)
	hist_Bmode_noise = hist_galn(allfield_bmode_noise)
	errorbar(hist_peaks[0],hist_peaks[1],hist_peaks[2],color='r',label='Kmap-peaks')	
	errorbar(hist_noise[0],hist_noise[1],hist_noise[2],color='r',linestyle='--',label='Kmap-noise')	
	errorbar(hist_Bmode[0],hist_Bmode[1],hist_Bmode[2],color='b',label='Bmode-peaks')	
	errorbar(hist_Bmode_noise[0],hist_Bmode_noise[1],hist_Bmode_noise[2],color='b',linestyle='--',label='Bmode-noise')
	legend(fontsize=12)
	izhi=float(z_hi[:3])
	if izhi == 1.3:
		izhi=0
	title(r'$W%i\,R=%s\, arcmin,\, z_{lo}=[0,\,%s],\, z_{hi}=[%s,\,1.3]$'%(Wx, arcmin, z_lo, izhi))
	xlabel('Kappa')
	ylabel('# of galaxies')
	savefig(plot_dir+'W%i_galn_peaks_%sarcmin_zlo%s_zhi%s.jpg'%(Wx,arcmin, z_lo, z_hi))
	#savefig(plot_dir+'galn_peaks_%sarcmin_zlo%s_zhi%s.jpg'%(arcmin, z_lo, z_hi))
	close()
if update_mag_i:
	## 10/06/2014, replace Mag_i = -99 items with Mag_y values
	# very messy, one time use only, because of CFHT failed i filter
	ra_arr, dec_arr, Mag_y_arr = np.load('/Users/jia/CFHTLenS/catalogue/Mag_y.npy').T
	radecy = np.load('/Users/jia/CFHTLenS/catalogue/Mag_y.npy')[:,:2]
	radecy0 = radecy[:,0]**radecy[:,1]#trick to make use of in1d for 2d
	
	for i in range(1,5):
		print i
		icat = cat_gen(i)
		icat_new = icat.copy()
		idx = where(icat[:,2]==-99)[0]
		print 'bad M_i', len(idx)
		
		radec99 = icat[idx,:2]
		radec990 = radec99[:,0]**radec99[:,1]#trick to make use of in1d for 2d
		# find the index for the intersect arrays, and tested both are unique arrays, no repeating items
		idx_99= nonzero(np.in1d(radec990, radecy0))
		idx_y = nonzero(np.in1d(radecy0, radec990))
		print 'len(idx_99), len(idx_y)',len(idx_99[0]), len(idx_y[0])
		# sort the ra, dec, to match the 2 list, and get index
		idx_sorted99 = argsort(radec990[idx_99])
		idx_sortedy = argsort(radecy0[idx_y])
		
		# check
		# radec990[idx_99[0][idx_sorted99]]-radecy0[idx_y[0][idx_sortedy]]
		# pass - returns 0
		
		icat_new[idx[idx_99[0][idx_sorted99]],-2] = Mag_y_arr[idx_y[0][idx_sortedy]]
		print 'mag_y=-99, mag_y==99',sum(icat_new[:,-2]==-99),sum(icat_new[:,-2]==99)
		np.save('/Users/jia/CFHTLenS/obsPK/W%s_cat_z0213_ra_dec_magy_zpeak'%(i), icat_new)
##
if do_hist_galn_magcut:
	print 'hi'
	mag_cut = -21
	height_arr = ['high', 'med', 'low']
	def idx_height (galn_arr, kappa_arr, height='med'):
		if height == 'low':
			idx = where(kappa_arr<0.03)[0]
		if height == 'med':
			idx = where((kappa_arr>0.03)&(kappa_arr<0.06))[0]
		if height == 'high':
			#idx = where(kappa_arr>0.06)[0]
			idx = where(kappa_arr<100)[0]
		return galn_arr[idx].squeeze()
	
	for height in height_arr:
		f=figure(figsize=(12, 8))
		i = 1
		for z_lo in (0.5, 0.6, 0.7):
			for R in (1.0, 2.0, 3.0):
				z_hi = '%s_hi'%(z_lo)
				galn_arr, kappa_arr = hist_galn_magcut(z_lo, z_hi, mag_cut = mag_cut,noise=False, R=R)
				galn_noise_arr, kappa_noise_arr = hist_galn_magcut(z_lo, z_hi, mag_cut = mag_cut, noise=True, R=R)
				ax=f.add_subplot(3,3,i)
				
				galn_peaks = idx_height (galn_arr, kappa_arr, height=height)
				galn_noise = idx_height (galn_noise_arr, kappa_noise_arr, height=height)
				ax.hist(galn_peaks, histtype='step', bins=20, label='peaks, %s'%(height))
				ax.hist(galn_noise, histtype='step', ls='dashed', bins=20, label='noise, %s'%(height))
				if i >6:
					ax.set_xlabel('# gal')
				if i in (1, 4, 7):
					ax.set_ylabel('# peaks')
				
				if i == 1:
					ax.set_title('%s peaks, M<%s'%(height, mag_cut))
				leg=ax.legend(ncol=1, labelspacing=0.3, prop={'size':10},loc=0, title='z=%s, R=%sarcmin'%(z_lo, R))
				leg.get_frame().set_visible(False)
					
				i+=1
		savefig(plot_dir+'hist_galn_magcut%s_%s_rand.jpg'%(mag_cut, height))
		close()
			
def hist_galn_magcut(z_lo, z_hi, R=2.0, mag_cut=-19, noise=False):
	'''This requires that the icat files exist already.
	This function reads the file, then cut out galaxies by magnitude, then count #galn for each peak.
	'''
	icat0 = np.load('/Users/jia/CFHTLenS/obsPK/peaks_mag_%s_lo_%s_R%s_noise%s.npy'%(z_lo, z_hi, R, noise))#colums 0) identifier, 1) kappa, 2) mag_i, 3) z_peak
	# exclude or include the -99, 99 galaxies?, or get those from other bands?
	icat = icat0[:,where((icat0[2]>-99)&(icat0[2]<99))].squeeze()
	mag_i, z_peak = icat[2:]
	mag_rest = M_rest_fcn(mag_i, z_peak)
	icat_cut = icat[:,where(mag_rest<mag_cut)].squeeze()
	sort_idx = argsort(icat_cut[0])
	unique_idx = nonzero(icat_cut[0,sort_idx[1:]]-icat_cut[0,sort_idx[:-1]])
	unique_idx = concatenate([[0],unique_idx[0]+1])#include 0 into index
	galn_arr = concatenate([unique_idx[1:]-unique_idx[:-1],[len(icat_cut[0])-unique_idx[-1]]])
	kappa_arr = icat_cut[1,sort_idx[unique_idx]]
	return galn_arr, kappa_arr

cat_gen_junk = lambda Wx: np.load('/Users/jia/CFHTLenS/obsPK/W%s_cat_z0213_ra_dec_magy_zpeak.npy'%(Wx))


######### plotting #################
		#identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance, weight = gridofdata[:,idx_fore]
		#ikappa = ikappa[0]
		#figure(figsize=(8,7))
		#subplot(221)
		#scatter(log10(M_halo),log10(icontribute),marker='x',s=5)
		#title ('peak #%i, kappa = %.4f'%(i, ikappa))
		#xlabel('log10(M_halo/M_sun)')
		#ylabel('log10(kappa/kappa_tot)')

		#subplot(222)
		#scatter(log10(rad2arcmin(distance)), log10(icontribute),marker='x',s=5)
		#xlabel('log10(r) arcmin')
		#title ('convergence = %.4f'%(kappa_list[0,i]))
		
		#subplot(223)
		#hist(redshift)
		#xlabel('z')

		#savefig(obsPK_dir+'plot/sample2_contribute_vs_Mhalo_%s.jpg'%(i))
		#close()

def quicktest(Wx):
	'''Check if peaks in kmap is also peaks in bmap, so somehow 
	peaks leak into bmode..
	'''
	bmap = bmodeGen(Wx, z=z_hi)
	kmap = kmapGen(Wx, z=z_hi)
	ipeak_mat = WLanalysis.peaks_mat(kmap)
	ipeak_matb = WLanalysis.peaks_mat(bmap)
	imask = maskGen (Wx, z=z_lo)
	ipeak_mat[where(imask==0)]=nan
	ipeak_matb[where(imask==0)]=nan
	print '[W%i], kmap peaks: %i, bmap peaks: %i, overlapping peaks: %i, bmap-kmap peaks: %i'%(Wx, sum(~isnan(ipeak_mat)), sum(~isnan(ipeak_matb)), sum(~isnan(ipeak_mat+ipeak_matb)), sum(~isnan(ipeak_matb))-sum(~isnan(ipeak_mat)))

def Wcircle (arcmin=2.0, PPA=PPA512):
	'''create a circular mask, =1 for within 2 arcmin, =0 for outside
	'''
	isize = int(PPA*2*arcmin)+1
	if isize%2 == 0:
		isize += 1 #make an odd size, so the middle one can center at the peak
	mask_circle = zeros (shape=(isize, isize))
	y, x = np.indices((isize, isize))
	center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
	r = np.hypot(x - center[0], y - center[1])/PPA
	mask_circle[where(r<arcmin)]=1
	return mask_circle, isize/2


def PeakGaln (Wx, z_lo=0.85, z_hi='1.3_lo', arcmin=2.0, noise=False, Bmode=False):
	'''For a map(kappa or bmode), find peaks, and # gal fall within
	arcmin of that peak.
	'''
	#print 'noise', noise, Wx
	mask_circle, o = Wcircle(arcmin=arcmin)
	if Bmode:
		kmap = bmodeGen(Wx, z=z_hi)
	else:
		kmap = kmapGen(Wx, z=z_hi)
	ipeak_mat = WLanalysis.peaks_mat(kmap)
	imask = maskGen (Wx, z=z_lo)
	ipeak_mat[where(imask==0)]=nan
	igaln = galnGen_lo(Wx, z=z_lo)
	if noise:
		idx_all = where((imask==1)&isnan(ipeak_mat))
		sample = randint(0,len(idx_all[0])-1,sum(~isnan(ipeak_mat)))
		idx = array([idx_all[0][sample],idx_all[1][sample]])
	else:
		idx = where(~isnan(ipeak_mat)==True)
	kappaGaln_arr = zeros(shape=(len(idx[0]),2))
	for i in range(len(idx[0])):
		x, y = idx[0][i], idx[1][i]
		kappaGaln_arr[i,0] = kmap[x, y]
		kappaGaln_arr[i,1] = sum(igaln[x-o:x+o+1, y-o:y+o+1]*mask_circle)
	return kappaGaln_arr.T

def hist_galn (allfield, kmin=-0.04, kmax=0.12, bins=10):
	'''
	make a histogram, for each kappa bin, the average gal#, and std
	allfield = [kappa_arr, galn_arr]
	Output: [kappa, mean, std]
	'''
	kappa_arr, galn_arr = allfield
	edges = linspace(kmin, kmax, bins+1)
	hist_arr = zeros(shape=(bins,3)) # mean, std
	for i in range(bins):
		#print i
		igaln = galn_arr[where((kappa_arr>edges[i])&(kappa_arr<edges[i+1]))]
		hist_arr[i,0]=0.5*(edges[i]+edges[i+1])
		hist_arr[i,1]=mean(igaln)
		hist_arr[i,2]=std(igaln)
	return hist_arr.T

def collect_allfields (z_lo=0.85, z_hi='1.3_lo', arcmin=2.0, noise=False, kmin=-0.04, kmax=0.12, bins=10, Bmode=False):
	'''using grid (not catalgue),
	collect the kappa arr and galn arr for all 4 fields
	'''
	kappaGaln_arr=array([PeakGaln(Wx, z_lo=z_lo, z_hi=z_hi, arcmin=arcmin, noise=noise, Bmode=Bmode) for Wx in range(1,5)])
	kappa_arr = concatenate([kappaGaln_arr[i,0] for i in range(4)])
	galn_arr = concatenate([kappaGaln_arr[i,1] for i in range(4)])
	return kappa_arr, galn_arr

def cat_galn_mag(Wx, z_lo=0.6, z_hi='0.6_lo', R=3.0, noise=False, Bmode=False):
	'''updated 2014/12/09
	First open a kappa map, get a list of peaks (with RA DEC), then open the catalogue for 
	all galaxies, and for each peak, find galaxies within R arcmin of the peak. Document the following values:
	1) identifier for a peak (which links to another file with identifier, kappa_peak, ra, dec)
	2) ra, dec
	3) redshift
	4) r_SDSS-z_SDSS, MAG_z, for finding L_k
	5) MAG_i in rest frame, for galaxie cut
	6) halo mass got from interpolator, using tablulated values (r-z, M_z) -> M_halo
	
	older version: return a list of peaks, with colums 0) identifier, 1) kappa, 2) mag_i, 3) z_peak
	'''
	print Wx
	kappa_arr, peak_ras, peak_decs= PeakPos(Wx, z_lo=z_lo, z_hi=z_hi, noise=noise, Bmode=Bmode)
	#icat = cat_gen(Wx)
	#idx = where(icat[:,-1]<z_lo)#older version, now no cut on z anymore
	#ra_arr, dec_arr, mag_arr, z_arr = icat[idx].T
	ra_arr, dec_arr, z_arr, weight, MAG_u, MAG_g, MAG_r, MAG_iy, MAG_z, r_SDSS, z_SDSS = cat_gen(Wx).T
	def loop_thru_peaks(i):
		'''for each peak, find the galaxies within R, then record their mag, z, Mhalo, etc.
		return colums [identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance from peak]
		'''
		print 'peak#',i
		ra0, dec0 = peak_ras[i], peak_decs[i]
		idx, dist = neighbor_index(ra0, dec0, ra_arr, dec_arr, R=R)
		# here shift magnitude to rest frame, calculate halo mass
		SDSSr_rest = M_rest_fcn(r_SDSS[idx], z_arr[idx])
		SDSSz_rest = M_rest_fcn(z_SDSS[idx], z_arr[idx])
		MAG_z_rest = M_rest_fcn(MAG_z[idx], z_arr[idx])
		MAG_i_rest = M_rest_fcn(MAG_iy[idx], z_arr[idx])
		rminusz = SDSSr_rest - SDSSz_rest		
		M_arr = array([findM(SDSSz_rest[j], rminusz[j]) for j in range(len(MAG_z_rest))])
		M_arr[where(abs(r_SDSS[idx])==99)]=0#set the bad data to 0
		M_arr[where(abs(z_SDSS[idx])==99)]=0#set the bad data to 0
		ipeak_halo_arr = concatenate([i*ones(len(idx))+0.1*Wx, ra_arr[idx], dec_arr[idx], z_arr[idx], SDSSr_rest, SDSSz_rest, MAG_i_rest, M_arr[:,0], dist, weight[idx]]).reshape(9,-1)
		return ipeak_halo_arr
	print 'total peaks:',len(kappa_arr)
	all_peaks_mag_z = map(loop_thru_peaks, range(len(kappa_arr)))
	return concatenate(all_peaks_mag_z,axis=1)

####### test Minterp, extent=(-25.1,-14.4,-5.6,6.3) - pass!#######
#randz = rand(10)*(25.1-14.4)-25.1
#randrminusz = rand(10)*(6.3-5.6)-5.6
#for i in range(10):
	#M_true, M_err = findM(randz[i], randrminusz[i])
	#M_interp = Minterp(randz[i], randrminusz[i])
	#print '%.2f\t%.2f\t%.2f'%(M_interp/M_true-1, M_err, Mminfun(M_interp,randz[i], randrminusz[i])/L_Lsun1(randz[i], randrminusz[i]))
##################################################################

def neighbor_index(ra0, dec0, ra_arr, dec_arr, R=2.0):
	'''find the index of ra_arr, dec_arr for galaxies 
	within R arcmin of (ra0, dec0)
	note: ra0, dec0, ra_arr, dec_arr are all in degrees, white R in arcmin
	return list of index
	'''
	idx_square = where( (abs(dec_arr-dec0)<(1.2*R/60.0))& (abs(ra_arr-ra0)<(1.2*R/60.0)) )
	f = WLanalysis.gnom_fun((ra0,dec0))
	x_rad, y_rad = f((ra_arr[idx_square], dec_arr[idx_square]))
	dist = sqrt(x_rad**2+y_rad**2) #in unit of radians
	idx_dist = where(dist<radians(R/60.0))
	idx = idx_square[0][idx_dist]
	return idx, dist[idx_dist]

##################### MAG_z to M100 ##########
L_Lsun1 = lambda MAG_z, rminusz: 10**(-0.4*MAG_z+1.863+0.444*rminusz)#from mag
def L_Lsun_VO(M): 
	if M<1e19:
		out = 1.23e10*(M/3.7e9)**29.78*(1+(M/3.7e9)**(29.5*0.0255))**(-1.0/0.0255)/h**2
	else:
		out = 1.23e10*(M/3.7e9)**(29.78-29.5)/h**2
	return out
L_Lsun_CM = lambda M: 4.4e11*(M/1e11)**4.0*(0.9+(M/1e11)**(3.85*0.1))**(-1.0/0.1)



datagrid_VO = np.load(obsPK_dir+'Mhalo_interpolator_VO.npy')
Minterp = interpolate.CloughTocher2DInterpolator(datagrid_VO[:,:2],datagrid_VO[:,2])
Mminfun = lambda M, MAG_z, rminusz: L_Lsun_VO(M)-L_Lsun1(MAG_z, rminusz)

############ prepare interpolation
input_arr = array([[SDSSz, rminusz] for SDSSz in linspace(-26, -14, 201) for rminusz in linspace(-6, 8, 201)])

output_arr = pad(input_arr,((0,0),(0,2)),mode='constant')
for i in arange(len(input_arr)):	
	MAG_z, rminusz = input_arr[i]
	x = op.brentq(Mminfun, 1e9, 1e40, args=(MAG_z, rminusz))
	fun = Mminfun(x, MAG_z, rminusz)
	output_arr[i,2]=x
	output_arr[i,3]=fun
	print i, x, fun
save(obsPK_dir+'Mhalo_interpolator_VO.npy', output_arr)
def findM(MAG_z, rminusz):
	try:
		x = op.brentq(Mminfun, 1e9, 1e40, args=(MAG_z, rminusz))
	except Exception:
		print 'have to use interpolation'
		x = Minterp(MAG_z, rminusz)
	fun = Mminfun(x, MAG_z, rminusz)
	if abs(fun) > 1:
		print 'abs(fun) > 1 with: MAG_z, r-z =', MAG_z, rminusz
	return x, fun

def maskGen (Wx, sigmaG=1.0, zcut=0.6, sigma_pix=0):
	'''generate mask using galn (galaxy count) map
	sigma_pix is the smoothing scale of the mask in
	unit of pixels
	z should be the lower bound for the convergence map.
	'''
	print Wx, sigmaG, zcut
	galn = WLanalysis.readFits(obsPK_dir+'maps/W%s_galn_%s_hi_sigmaG%02d.fit'%(Wx,zcut,sigmaG*10))
	mask = ones(shape=galn.shape)
	#mask = zeros(shape=galn.shape)
	#mask[25:-25,25:-25] = 1
	idx = where(galn<0.5)
	mask[idx] = 0
	mask_smooth = WLanalysis.smooth(mask, sigma_pix)
	save(obsPK_dir+'maps/Mask_W%s_%s_sigmaG%02d.npy'%(Wx,zcut,sigmaG*10),mask_smooth)
	#return mask_smooth
for sigmaG in (0.5, 1.0, 1.8, 3.5, 5.3, 8.9):
	for Wx in range(1,5):
		for zcut in (0.5, 0.6, 0.7):
			maskGen(Wx, sigmaG, zcut)
			

# This is smoothed galn
# galnGen = lambda i, z: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_galn_%s_hi_sigmaG10.fit'%(i, z))
galnGen_hi = lambda i, z: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_galn_%s_hi.fit'%(i, z))
galnGen_lo = lambda i, z: WLanalysis.readFits('/Users/jia/CFHTLenS/obsPK/maps/W%i_galn_%s_lo.fit'%(i, z))

##########################################################
noise_arr = np.load (obsPK_dir+'Halos_IDziM_DistContri_k4_kB_zcut%s_R%s_noise%s.npy'%(zcut, R, True))
noiseIDs, noisez_arr, noiseMAGi_arr, noiseMhalo_arr, noised_arr, \
	noisecontri_arr, noisekappaP_arr, noisekappaConv_arr = \
	noise_arr[:, idxcuts(noise_arr)]

#########################################################

################################################
################ operations ####################
################################################
zcenters = arange(0.225, 1.3, 0.05)#center of z bins from CFHT
zbins = linspace(0.2,1.3,23)#edges
zPDF = array([ 0.45445094,  0.80881598,  0.93470199,  0.76456038,  1.10499822,
        0.8803627 ,  1.1195881 ,  1.50167501,  1.48711827,  1.60911659,
        1.47213078,  1.31645756,  1.2022788 ,  1.76004064,  1.00095837,
        1.08889524,  0.97712418,  0.91611398,  0.57378752,  0.31843705,
        0.53337544,  0.17501226])# = dP/dz, normed in between z=0.2-1.3

zPDF_normed = lambda zcut: zPDF[:where(zcenters<=zcut)[0][-1]+1]/sum(zPDF[:where(zcenters<=zcut)[0][-1]+1])

	
def Nhalo_vs_kappa (icontri_arr, iz_arr, izPDF):#iMhalo_arr
	'''(1) count from the most contribution halo, til get 50% of the contribution.
	(2) for a redshift PDF, normalized to N gals, for each redshift bins, assume sqrt(N) noise,
	find peaks that have SNR > 3, say that's the # of peaks.
	return: (Nhalo, Nzpeak)
	'''
	## use redshift to find clusters
	NPDF = len(iz_arr)*izPDF/sum(izPDF)
	ihist = histogram(iz_arr, bins=zbins[:len(izPDF)+1])[0]
	SNR = (ihist-NPDF) / sqrt(NPDF)
	iNclusters = sum(SNR>=3)
	
	## use galaxies, to find # of galaxies needed to contribute to largest mass
	icontri_arr /= sum(icontri_arr)
	iNgals = sum(cumsum(sort(icontri_arr)[::-1])<0.5)+1
	return iNclusters, iNgals

if cluster_counts:
	R, zcut, noise = 3.0, 0.7, True
	Rcut = 3.0
	
	halo_arr = np.load (obsPK_dir+'Halos_IDziM_DistContri_k4_kB_zcut%s_R%s_noise%s.npy'%(zcut, R, noise))

	def idxcuts(halo_arr):
		IDs, z_arr, MAGi_arr, Mhalo_arr, d_arr, contri_arr, \
			kappaP_arr, kappaConv_arr = halo_arr		
		idx_cut = where((MAGi_arr > -24) & (MAGi_arr < -18) & (Mhalo_arr < 5.3e15) &
			(rad2arcmin(d_arr) < Rcut) & (kappaP_arr < 1.0))[0]
		return idx_cut

	idx_cut = idxcuts(halo_arr)
	IDs, z_arr, MAGi_arr, Mhalo_arr, d_arr, contri_arr, \
		kappaP_arr, kappaConv_arr = halo_arr[:,idx_cut]

	uniqueID = unique(IDs)
	izPDF = zPDF_normed(zcut)
	def Nhalo_count(i):#for i in randint(0,11931,20):
		print i
		iidx = where(IDs==uniqueID[i])[0]
		icontri_arr, iz_arr = contri_arr[iidx], z_arr[iidx]	
		iNclusters, iNgals = Nhalo_vs_kappa(icontri_arr, iz_arr, izPDF)
		return uniqueID[i], kappaP_arr[iidx[0]], kappaConv_arr[iidx[0]], iNclusters, iNgals

	#all_Nhalos = map(Nhalo_count, range(len(uniqueID)))
	#save(obsPK_dir+'ClusterCounts_ID_k4_kB_Ncluster_Ngal_zcut%s_R%s_noise%s.npy'%(zcut, Rcut, noise), all_Nhalos)

if halo_plots:
	R = 3.0
	zcut = 0.6
	all_Nhalos = load(obsPK_dir+'ClusterCounts_ID_k4_kB_Ncluster_Ngal_zcut%s_R3.0_noiseFalse.npy'%(zcut))
	all_Nhalos_noise = load(obsPK_dir+'ClusterCounts_ID_k4_kB_Ncluster_Ngal_zcut%s_R3.0_noiseTrue.npy'%(zcut))
	
	ID, kappaP, kappaConv, Ncluster, Nhalo = array(all_Nhalos).T
	nID, nkappaP, nkappaConv, nNcluster, nNhalo = array(all_Nhalos_noise).T


	########### Van Wearbeke Fig.9 ############################
	#kappa_binedges = linspace(0,0.0015, 5)
	#mean_kappaConv = zeros(len(kappa_binedges)-1)
	#nmean_kappaConv = zeros(len(kappa_binedges)-1)
	#for j in range(len(kappa_binedges)-1):
		#idx_bin = where((kappaP<kappa_binedges[j+1])&(kappaP>kappa_binedges[j]))[0]
		#mean_kappaConv[j] = mean(kappaConv[idx_bin])
		
		#nidx_bin = where((nkappaP<kappa_binedges[j+1])&(nkappaP>kappa_binedges[j]))[0]
		#nmean_kappaConv[j] = mean(nkappaConv[idx_bin])
	#plot(kappa_binedges[:-1], mean_kappaConv,'o')
	#plot(kappa_binedges[:-1], nmean_kappaConv,'o')
	#show()
		
	
	########### 2dhist = scatter plot, kappaP vs kappaConv##############
	#figure()
	#hist2d(kappaP, kappaConv,range=((0,0.0015),(-0.05,0.15)), bins=20)
	#xlabel('kappa_project (from foreground halos)')
	#ylabel('kappa_convergence (using background galaxies)')
	#coeff, P = stats.spearmanr(kappaP, kappaConv)
	#title('zcut=%s, coeff=%.5f, P=%.5f'%(zcut,coeff,P))
	#colorbar()
	#show()
	#savefig(plot_dir+'conv_vs_proj_zcut%s.jpg'%(zcut))
	#close()
	####################################################################
	
	######### Yang 2011 Fig.5 ####################
	#Nbin_edges = linspace(0.5, 8.5, 9)
	#kappa_arr = [kappaP, kappaConv]
	#nkappa_arr = [nkappaP, nkappaConv]
	#sP, sC = std(kappaP), std(kappaConv)
	#cuts = ([[-inf, sP],[sP, 3*sP],[3*sP, inf]],[[-inf, sC],[sC, 3*sC],[3*sC, inf]])
	#f=figure(figsize=(12,8))
	#title_arr = [['low (proj)','med (proj)','hi (proj)'],['low (conv)','med (conv)','hi (conv)']]
	#for i in range(2):
		#for j in range(3):
			#x0, x1 = cuts[i][j]
			#idxS = where((kappa_arr[i]<x1)& (kappa_arr[i]>x0))[0]
			#idxN = where((nkappa_arr[i]<x1)& (nkappa_arr[i]>x0))[0]
			#ax=f.add_subplot(2,3,j+1+i*3)
			
			##ax.hist(Ncluster[idxS], bins=Nbin_edges, histtype='step',label='peaks',normed=True)
			##ax.hist(Ncluster[idxN],bins=Nbin_edges, histtype='step',label='rnd. direction',normed=True)
			
			
			#ax.hist(Nhalo[idxS], bins=Nbin_edges, histtype='step',label='peaks',normed=True)
			#ax.hist(nNhalo[idxN],bins=Nbin_edges, histtype='step',label='rnd. direction',normed=True)
			#ax.hist(nNhalo, bins=Nbin_edges, histtype='step',label='rnd. all k',normed=True)
			
			#ax.set_title(title_arr[i][j])
			#if i == 0 and j == 1:
				#ax.set_ylim(0,0.5)
			#if i==0 and j==0:
				#ax.legend(fontsize=10)
			#if i == 1:
				#ax.set_xlabel('N_halo')
				##ax.set_xlabel('N_cluster (SNR>3 in redshift bins)')
			#if j == 0:
				#ax.set_ylabel('Num. peaks')
	#plt.subplots_adjust(wspace=0.25,hspace=0.25)
	#savefig(plot_dir+'Nhalo_Npeaks_zcut%s.jpg'%(zcut))
	##savefig(plot_dir+'Ncluster_Npeaks_zcut%s.jpg'%(zcut))
	#close()
	###############################################
	
	######## look at where I found clusters #########
	#idx_cluster = nonzero(Ncluster)
	#f=figure(figsize=(8,5))
	#subplot(121)
	#hist(kappaP[idx_cluster], range=(0,0.0015), histtype='step',label='clusters',normed=True)
	#hist(kappaP, histtype='step', range=(0,0.0015), label='all peaks',normed=True)
	#legend(fontsize=10)
	#xlabel('kappa_project')
	#title('peak counts')
	#matplotlib.pyplot.locator_params(nbins=4)
	#subplot(122)
	#hist(kappaConv[idx_cluster],range=(-0.05, 0.2),histtype='step',label='clusters',normed=True)
	#hist(kappaConv, range=(-0.05, 0.2), histtype='step',label='all peaks',normed=True)
	#xlabel('kappa_convergence')
	#title('peak counts')
	#matplotlib.pyplot.locator_params(nbins=4)
	#savefig(plot_dir+'PeakCounts_withCluster_zcut%s.jpg'%(zcut))
	#close()

if project_mass:
	R=3.0
	zcut=0.6	
	noise=False
	#for znoise in [[z, noise] for z in (0.5, 0.6, 0.7) for noise in (True, False)]:
		#zcut, noise = znoise
	#zcut = float(sys.argv[1])
	#noise = bool(int(sys.argv[2]))
	
	kappa_list = np.load(obsPK_dir+'AllPeaks_kappa_raDec_zcut%.1f.npy'%(zcut))
	print 'got files'
	## columns: kappa, ra, dec
	alldata = np.load(obsPK_dir+'peaks_IDraDecZ_MAGrziMhalo_dist_weight_zcut%.1f_R%s_noise%s.npy'%(zcut, R, noise))
	## columns: identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance, weight
	
	ids = alldata[0, sort(np.unique(alldata[0], return_index=True)[1])]#all the identifiers

	print 'len(ids)',len(ids)
	def halo_contribution(i):#for i in randint(0,11931,20):
		print zcut, noise, i
		iidx = where(alldata[0]==ids[i])[0]
		oldgrid = alldata[:, iidx]
		idx_fore, icontribute, ikappa = MassProj (oldgrid, zcut)
		if len(idx_fore)==0:
			return nan*zeros(shape=(8,1))
		else:
			newgrid = oldgrid[:, idx_fore]
			identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance, weight = newgrid
			newarr = array([identifier, redshift, MAG_iy_rest, M_halo, distance, icontribute, ikappa, kappa_list[0,i]*ones(len(ikappa))])# things I need for final analysis
			return newarr
	halo_fn = obsPK_dir+'Halos_IDziM_DistContri_k4_kB_zcut%s_R%s_noise%s'%(zcut, R, noise)
	
	pool = MPIPool()
	all_halos = pool.map(halo_contribution, range(len(ids)))
	all_halos = concatenate(all_halos, axis=1)
	#np.save(halo_fn,all_halos)
	

#################################################################

if list_peaks_cat:
	'''create a list of peaks, for all peaks (in 4 fields), into -
	columns: identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance, weight
	'''
	R = 3.0
	for znoise in [[z_lo, noise] for z_lo in (0.5, 0.6, 0.7) for noise in (True, False)]:
		z_lo, noise = znoise
		z_hi = '%s_hi'%(z_lo)
		print 'z_lo, noise, R:',',', z_lo,',', noise,',', R
		fn = obsPK_dir+'peaks_IDraDecZ_MAGrziMhalo_dist_zcut%s_R%s_noise%s.npy'%(z_lo, R, noise)
		#columns: identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance, weight
		seed(int(z_lo*10+R*100))	
		temp_arr = [cat_galn_mag(Wx, z_lo=z_lo, z_hi=z_hi, R=R, noise=noise) for Wx in range(1,5)]
		np.save(fn, concatenate(temp_arr,axis=1))
		### the following block creates the catalogue for peaks [kappa, RA, DEC]
		# all_peaks = [PeakPos(Wx, z_lo=z_lo, z_hi=z_hi, noise=noise) for Wx in range(1,5)]
		# np.save(obsPK_dir+'AllPeaks_kappa_raDec_zcut%s.npy'%(z_lo), concatenate(all_peaks, axis=1))
		#############################################################

def MassProj (gridofdata, zcut, R = 3.0, sigmaG=1.0):
	'''For one peak, I try to get a projected kappa from foreground halos. z>zcut are used for kappa map, z<zcut are foreground halos.
	steps:
	1) cut galaxies to background & foreground by zcut
	2) shoot light rays to each background galaxy, find kappa at that position
	3) smooth within R, find kappa_proj at peak location
	4) output: index for foreground galaxies with non zero contribution to kappa, zero contributions can be due to problematic foreground galaxie with no magnitude data.
	5) note, everything need to be weighted by CFHT weight
	'''
	idx_dist = where(degrees(gridofdata[-2])<R/60.0)[0]
	identifier, ra, dec, redshift, SDSSr_rest, SDSSz_rest, MAG_iy_rest, M_halo, distance, weight = gridofdata[:,idx_dist]
	idx_fore = where((redshift<zcut)&(1e5<M_halo))[0]
	if len(idx_fore)==0: # no foreground galaxies
		return [], [], []
	else:
		idx_back = where((redshift>=zcut))[0]
		
		### weight using gaussian wintow
		weight_arr = exp(-rad2arcmin(distance[idx_back])**2.0/(2*pi*sigmaG**2.0))*weight[idx_back]
		#weight_arr2 = weight_arr/sum(weight_arr)
		
		###### kappa_arr.shape = [ngal_fore, ngal_back]
		kappa_arr = zeros(shape=(len(idx_fore),len(idx_back)))
		
		for j in range(len(idx_fore)):# foreground halo count
			jidx = idx_fore[j]
			z_fore, M, ra_fore, dec_fore = redshift[jidx], M_halo[jidx], ra[jidx], dec[jidx]
			ikappa_proj = kappa_proj (z_fore, M, ra_fore, dec_fore)	
			i = 0
			for iidx_back in idx_back:
				kappa_arr[j,i]=ikappa_proj(redshift[iidx_back], ra[iidx_back], dec[iidx_back])
				i+=1
		#weight_arr1 = weight_arr * (kappa_arr!=0)
		kappa_arr[isnan(kappa_arr)]=0
		icontribute = sum(kappa_arr*weight_arr, axis=1)/sum(weight_arr * (kappa_arr!=0), axis=1)#sum over back ground galaxies
		idx_nonzero=where((icontribute!=0)&(~isnan(icontribute)))[0]
		ikappa = sum(icontribute[idx_nonzero])
		icontribute/=ikappa
		return idx_dist[idx_fore[idx_nonzero]], icontribute[idx_nonzero], ikappa*ones(len(idx_nonzero))

cat_gen_old = lambda Wx: np.load(obsPK_dir+'junk/W%s_cat_z0213_ra_dec_weight_z_ugriz_SDSSr_SDSSz.npy'%(Wx)) #columns: ra, dec, z_peak, weight, MAG_u, MAG_g, MAG_r, MAG_iy, MAG_z, r_SDSS, z_SDSS
def Mhalo_gen (Wx):
	print Wx
	ra, dec, z_arr, weight, MAG_u, MAG_g, MAG_r, MAG_iy, MAG_z, r_SDSS, z_SDSS = cat_gen_old(Wx).T
	idx = where( (abs(r_SDSS)!=99)&(abs(z_SDSS)!=99) )[0]#rid of the mag=99 ones
	SDSSr_rest = M_rest_fcn(r_SDSS[idx], z_arr[idx])
	SDSSz_rest = M_rest_fcn(z_SDSS[idx], z_arr[idx])
	MAG_i_rest = M_rest_fcn(MAG_iy[idx], z_arr[idx])
	rminusz = SDSSr_rest - SDSSz_rest
	M_arr = Minterp(SDSSz_rest, rminusz)
	print '#nan values',len(where(~isnan(M_arr))
	M100 = M_arr[where(~isnan(M_arr))[0]]
	idx_new = idx[where(~isnan(M_arr))[0]]
	Mvir = M100/1.227
	Rvir_arr = Rvir_fcn(Mvir, z_arr[idx_new])
	DL_arr = DL_interp(z_arr[idx_new])	
	new_cat = array([ra[idx_new], dec[idx_new], z_arr[idx_new], weight[idx_new], MAG_i_rest[idx_new], Mvir, Rvir_arr, DL_arr]).T
	save(obsPK_dir+'W%s_cat_z0213_ra_dec_redshift_weight_MAGi_Mvir_Rvir_DL.npy'%(Wx), new_cat)
map(Mhalo_gen, range(1,5))

def kappa_proj_old (z_fore, M100, ra_fore, dec_fore, cNFW=5.0):
	'''return a function, for certain foreground halo, 
	calculate the projected mass between a foreground halo and a background galaxy pair.
	'''
	f = 1.043#=1.0/(log(1+cNFW)-cNFW/(1+cNFW)) with cNFW=5.0
	Mvir = M100/1.227#cNFW = 5, M100/Mvir = 1.227
	Rvir = Rvir_fcn(Mvir, z)#cm
	two_rhos_rs = Mvir*M_sun*f*cNFW**2/(2*pi*Rvir**2)#cgs, see LK2014 footnote
	xy_fcn = WLanalysis.gnom_fun((ra_fore, dec_fore))
	Dl = DL(z_fore)/(1+z_fore)**2 # D_angular = D_luminosity/(1+z)**2
	Dl_cm = Dl*3.08567758e24
	theta_vir = Rvir/Dl_cm

	def kappa_proj_fcn (z_back, ra_back, dec_back):
		Ds = DL(z_back)/(1+z_back)**2
		Dls = Ds - Dl
		DDs = Ds/(Dl*Dls)/3.08567758e24# 3e24 = 1Mpc/1cm
		SIGMAc = (c*1e5)**2/4.0/pi/Gnewton*DDs
		x_rad, y_rad = xy_fcn(array([ra_back, dec_back]))
		theta = sqrt(x_rad**2+y_rad**2)
		x = cNFW*theta/theta_vir
		Gx = Gx_fcn(x, cNFW)
		kappa_p = two_rhos_rs/SIGMAc*Gx
		return kappa_p
	return kappa_proj_fcn

######## organize kappa_predict files on stampede #########
import numpy as np
obsPK_dir = '/home1/02977/jialiu/obsPK/'
lenths = (1936992, 514501, 1269489, 593210)
for Wx in range(1,5):
	print Wx
	ifile = lambda ix: np.load(obsPK_dir+'temp/kappa_proj%i_%07d.npy'%(Wx, ix))
	ix_arr = np.arange(0, lenths[Wx-1], 1e4)
	allkappa = np.concatenate(map(ifile, ix_arr))
	print allkappa.shape
	np.save(obsPK_dir+'kappa_predict_Mmax2e15_W%i.npy'%(Wx), allkappa)
	
	
if make_kappa_predict:
	from scipy.spatial import cKDTree
	zcut = 0.2#0.6
	r = 0.0019#0.002rad = 7arcmin, within which I search for contributing halos

	Wx = 1#int(sys.argv[1])
	center = centers[Wx-1]
	icat = cat_gen(Wx).T

	ra, dec, redshift, weight, MAGi, Mhalo, Rvir, DL = icat
	f_Wx = WLanalysis.gnom_fun(center)
	xy = array(f_Wx(icat[:2])).T

	idx_back = where(redshift>zcut)[0]
	xy_back = xy[idx_back]

	kdt = cKDTree(xy)
#nearestneighbors = kdt.query_ball_point(xy_back[:100], 0.002)
	def kappa_individual_gal (i):
		'''for individual background galaxies, find foreground galaxies within 7 arcmin and sum up the kappa contribution
		'''
		print i
		iidx_fore = array(kdt.query_ball_point(xy_back[i], r))	
		x_back, y_back = xy_back[i]
		z_back, DL_back = redshift[idx_back][i], DL[idx_back][i]
		ikappa = 0
		for jj in iidx_fore:
			x_fore, y_fore = xy[jj]
			jMvir, jRvir, z_fore, DL_fore = Mhalo[jj], Rvir[jj], redshift[jj], DL[jj]
			if z_fore >= z_back:
				kappa_temp = 0
			else:
				kappa_temp = kappa_proj (jMvir, jRvir, z_fore, x_fore, y_fore, DL_fore, z_back, x_back, y_back, DL_back, cNFW=5.0)
				if isnan(kappa_temp):
					kappa_temp = 0
			ikappa += kappa_temp
			
			if kappa_temp>0:
				theta = sqrt((x_fore-x_back)**2+(y_fore-y_back)**2)
				print '%s\t%.2f\t%.3f\t%.3f\t%.4f\t%.6f'%(jj,log10(jMvir), z_fore, z_back, rad2arcmin(theta), kappa_temp)	
		return ikappa

	#a=map(kappa_individual_gal, randint(0,len(idx_back)-1,5))
	step=1e4
	def temp (ix):
		print ix
		kappa_all = map(kappa_individual_gal, arange(ix, amin([len(idx_back), ix+step])))
		np.save(obsPK_dir+'temp/kappa_proj%i_%07d.npy'%(Wx, ix),kappa_all)
	pool = MPIPool()
	ix_arr = arange(0, len(idx_back), step)
	pool.map(temp, ix_arr)

### coords2grid kappa_pred map:
for Wx in arange(1,5):
	sizes = (1330, 800, 1120, 950)
	print Wx
	isize = sizes[Wx-1]
	center = centers[Wx-1]
	icat = cat_gen(Wx).T
	#ra, dec, redshift, weight, MAGi, Mhalo, Rvir, DL = icat
	f_Wx = WLanalysis.gnom_fun(center)
	y, x = array(f_Wx(icat[:2]))
	weight = icat[3]
	k = np.load(obsPK_dir+'kappa_predict_W%i.npy'%(Wx))
	A, galn = WLanalysis.coords2grid(x, y, array([k*weight, weight]), size=isize)
	Mkw, Mw = A
	np.save('/Users/jia/CFHTLenS/catalogue/Me_Mw_galn/W%i_Mkw_pred.npy'%(Wx), Mkw)
	np.save('/Users/jia/CFHTLenS/catalogue/Me_Mw_galn/W%i_Mw_pred.npy'%(Wx), Mw)	