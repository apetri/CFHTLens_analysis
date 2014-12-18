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
		
noise_arr = np.load (obsPK_dir+'Halos_IDziM_DistContri_k4_kB_zcut%s_R%s_noise%s.npy'%(zcut, R, True))
noiseIDs, noisez_arr, noiseMAGi_arr, noiseMhalo_arr, noised_arr, \
	noisecontri_arr, noisekappaP_arr, noisekappaConv_arr = \
	noise_arr[:, idxcuts(noise_arr)]
