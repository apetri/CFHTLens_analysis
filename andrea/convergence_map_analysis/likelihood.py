from __future__ import print_function,division,with_statement

from operator import mul
from functools import reduce

import os,sys,re
import argparse,ConfigParser
import logging

######################################################################
##################LensTools functionality#############################
######################################################################

from lenstools import Ensemble
from lenstools.index import Indexer,MinkowskiAll
from lenstools.constraints import LikelihoodAnalysis
from lenstools.simulations import CFHTemu1
from lenstools.observations import CFHTLens

#################################################################################
####################Borrow the Measurer class from measure_features##############
#################################################################################

from measure_features import Measurement

######################################################################
###################Other functionality################################
######################################################################

import numpy as np
from emcee.utils import MPIPool

#######################################################################
###################Parse feature types#################################
#######################################################################

def parse_features(feature_string):

	all_smoothing_scales = list()
	features_to_measure = dict()

	features = feature_string.replace(" ","").split("*")

	for feature in features:

		feature_name = feature.split(":")[0]
		smoothing_scales = [ float(theta) for theta in feature.split(":")[1].split(",") ]

		features_to_measure[feature_name] = smoothing_scales

		for theta in smoothing_scales:
			if theta not in all_smoothing_scales:
				all_smoothing_scales.append(theta)

	return all_smoothing_scales,features_to_measure

def npy_filename(feature_type):

	if "minkowski" in feature_type:
		return "minkowski_all.npy"
	else:
		return feature_type+".npy"

def output_string(feature_string):
	return feature_string.replace(" ","").replace(":","--").replace("*","_").replace(",","-")

######################################################################
###################Main execution#####################################
######################################################################

if __name__=="__main__":

	#################################################
	############Option parsing#######################
	#################################################

	#Parse command line options
	parser = argparse.ArgumentParser()
	parser.add_argument("-f","--file",dest="options_file",action="store",type=str,help="analysis options file")
	parser.add_argument("-v","--verbose",dest="verbose",action="store_true",default=False,help="turn on verbosity")
	parser.add_argument("-m","--mask_scale",dest="mask_scale",action="store_true",default=False,help="scale peaks and power spectrum to unmasked area")
	parser.add_argument("-s","--save_points",dest="save_points",action="store",default=None,help="save points in parameter space to external npy file")
	parser.add_argument("-ss","--save_debug",dest="save_debug",action="store_true",default=False,help="save a bunch of debugging info for the analysis")

	cmd_args = parser.parse_args()

	if cmd_args.options_file is None:
		parser.print_help()
		sys.exit(0)

	#Set verbosity level
	if cmd_args.verbose:
		logging.basicConfig(level=logging.DEBUG)
	else:
		logging.basicConfig(level=logging.INFO)

	#Initialize MPI Pool
	try:
		pool = MPIPool()
	except:
		pool = None

	if (pool is not None) and (not pool.is_master()):
		pool.wait()
		sys.exit(0)

	#Parse INI options file
	logging.debug("Parsing options from {0}".format(cmd_args.options_file))

	options = ConfigParser.ConfigParser()
	with open(cmd_args.options_file,"r") as configfile:
		options.readfp(configfile)

	#Read the save path from options
	save_path = options.get("analysis","save_path")

	#Construct an index for the minkowski functionals, it will be useful for later
	th_minkowski = np.ogrid[options.getfloat("minkowski_functionals","th_min"):options.getfloat("minkowski_functionals","th_max"):(options.getint("minkowski_functionals","num_bins")+1)*1j]
	mink_idx = MinkowskiAll(thresholds=th_minkowski).separate()
	mink_idx = Indexer(mink_idx)

	#Get the names of all the simulated models available for the CFHT analysis, including smoothing scales and subfields
	all_simulated_models = CFHTemu1.getModels(root_path=options.get("simulations","root_path"))

	#Get also the observation model instance
	observed_model = CFHTLens(root_path=options.get("observations","root_path"))

	#Select subset of training models
	training_models = all_simulated_models
	#Use this model for the covariance matrix
	covariance_model = options.getint("analysis","covariance_model") - 1
	
	#Parse from options which subfields and smoothing scale to consider
	subfields = [ int(subfield) for subfield in options.get("analysis","subfields").split(",") ]

	#Parse from options which type of descriptors (features) to use
	feature_string = options.get("analysis","feature_types")
	smoothing_scales,features_to_measure = parse_features(feature_string)

	#Create a LikelihoodAnalysis instance and load the training models into it
	analysis = LikelihoodAnalysis()

	###########################################################
	###############Feature loading#############################
	###########################################################

	#Start loading the data
	logging.debug("Loading features...")
	for feature_type in features_to_measure.keys():
		logging.info("{0}, smoothing scales: {1} arcmin".format(feature_type,",".join([ str(s) for s in features_to_measure[feature_type] ])))
	
	for n,model in enumerate(training_models):

		logging.debug("Model {0}".format(n))
		logging.debug(model)

		#First create an empty ensemble
		ensemble_all_subfields = Ensemble()

		#Then cycle through all the subfields and gather the features for each one
		for subfield in subfields:
			
			m = dict()
			for smoothing_scale in smoothing_scales:
				m[smoothing_scale] = Measurement(model=model,options=options,subfield=subfield,smoothing_scale=smoothing_scale,measurer=None)
				m[smoothing_scale].get_all_map_names()

			#Construct one ensemble for each feature (with included smoothing scales) and load in the data
			ensemble_subfield = list()
			for feature_type in features_to_measure.keys():
				
				for smoothing_scale in features_to_measure[feature_type]:
					
					ens = Ensemble.fromfilelist([os.path.join(m[smoothing_scale].full_save_path,npy_filename(feature_type))])
					ens.load(from_old=True)

					#Check the masked fraction of the field of view
					masked_fraction = m[smoothing_scale].maskedFraction

					#Scale to the non masked area (only for power spectrum and peaks if option is enabled)
					if cmd_args.mask_scale:
						
						if feature_type=="power_spectrum":
							ens.scale(1.0/(1.0 - masked_fraction)**2)
						elif feature_type=="peaks":
							ens.scale(1.0/(1.0 - masked_fraction))

					#Check if we want to discard some of the Minkowski functionals
					num = re.match(r"minkowski_([0-2]+)",feature_type)
					if num is not None:
						mink_to_measure = [ int(n_mf) for n_mf in list(num.group(1)) ]
						ens_split = ens.split(mink_idx)
						[ ensemble_subfield.append(ens_split[n_mf]) for n_mf in mink_to_measure ]
					else:
						ensemble_subfield.append(ens)


			#Add the features to the cumulative subfield ensemble
			ensemble_all_subfields += reduce(mul,ensemble_subfield)

		#Add the feature to the LikelihoodAnalysis
		analysis.add_model(parameters=model.squeeze(),feature=ensemble_all_subfields.mean())

		#If this is the correct model, compute the covariance matrix too
		if n==covariance_model:
			features_covariance = ensemble_all_subfields.covariance()

	#Finally, measure the observed feature
	logging.debug("Loading observations...")
	ensemble_all_subfields = Ensemble()

	for subfield in subfields:
		
		m = dict()
		for smoothing_scale in smoothing_scales:
			m[smoothing_scale] = Measurement(model=observed_model,options=options,subfield=subfield,smoothing_scale=smoothing_scale,measurer=None)
			m[smoothing_scale].get_all_map_names()

		#Construct one ensemble for each feature (with included smoothing scales) and load in the data
		ensemble_subfield = list()
		for feature_type in features_to_measure.keys():
			
			for smoothing_scale in features_to_measure[feature_type]:
				
				ens = Ensemble.fromfilelist([os.path.join(m[smoothing_scale].full_save_path,npy_filename(feature_type))])
				ens.load(from_old=True)

				#Check the masked fraction of the field of view
				masked_fraction = m[smoothing_scale].maskedFraction

				#Scale to the non masked area (only for power spectrum and peaks if option is enabled)
				if cmd_args.mask_scale:
				
					if feature_type=="power_spectrum":
						ens.scale(1.0/(1.0 - masked_fraction)**2)
					elif feature_type=="peaks":
						logging.debug("Scaling peak counts of subfield {0}, masked fraction {1}".format(subfield,masked_fraction))
						ens.scale(1.0/(1.0 - masked_fraction))

				#Check if we want to discard some of the Minkowski functionals
				num = re.match(r"minkowski_([0-2]+)",feature_type)
				if num is not None:
					mink_to_measure = [ int(n_mf) for n_mf in list(num.group(1)) ]
					ens_split = ens.split(mink_idx)
					[ ensemble_subfield.append(ens_split[n_mf]) for n_mf in mink_to_measure ]
				else:
					ensemble_subfield.append(ens)

		#Add the features to the cumulative subfield ensemble
		ensemble_all_subfields += reduce(mul,ensemble_subfield)

	#Compute the average over subfields
	observed_feature = ensemble_all_subfields.mean()

	################################################################################################
	#####################Feature loading complete, ready for analysis###############################
	################################################################################################

	#If save_debug is enabled, save the training features, covariance and observed feature to npy files for check
	if cmd_args.save_debug:
		
		logging.debug("Saving debug info...")
		np.save("training_parameters.npy",analysis.parameter_set)
		np.save("training_{0}.npy".format(output_string(feature_string)),analysis.training_set)
		np.save("covariance_{0}.npy".format(output_string(feature_string)),features_covariance)
		np.save("covariance_observed_{0}.npy".format(output_string(feature_string)),ensemble_all_subfields.covariance())
		np.save("observation_{0}.npy".format(output_string(feature_string)),observed_feature)


	#Train the interpolators using the simulated features
	logging.debug("Training interpolators...")
	analysis.train()

	#If save_debug is enabled, test the interpolators for a fiducial cosmological model and save the result
	if cmd_args.save_debug:

		test_parameters = np.array([0.26,-1.0,0.8])
		logging.debug("Testing simple interpolation for Omega_m={0[0]},w={0[1]},sigma8={0[2]}...".format(test_parameters))

		test_interpolated_feature = analysis.predict(test_parameters)

		np.save("testinterp_{0}.npy".format(output_string(feature_string)),test_interpolated_feature)

	#Set the points in parameter space on which to compute the chi2 
	Om = np.ogrid[options.getfloat("Omega_m","min"):options.getfloat("Omega_m","max"):options.getint("Omega_m","num_points")*1j]
	w = np.ogrid[options.getfloat("w","min"):options.getfloat("w","max"):options.getint("w","num_points")*1j]
	si8 = np.ogrid[options.getfloat("sigma8","min"):options.getfloat("sigma8","max"):options.getint("sigma8","num_points")*1j]

	num_points = len(Om) * len(w) * len(si8) 

	points = np.array(np.meshgrid(Om,w,si8,indexing="ij")).reshape(3,num_points).transpose()
	if cmd_args.save_points is not None:
		logging.debug("Saving points to {0}.npy".format(cmd_args.save_points.rstrip(".npy")))
		np.save(cmd_args.save_points.rstrip(".npy")+".npy",points)

	#Now compute the chi2 at each of these points
	if pool:
		split_chunks = pool.size
		logging.debug("Computing chi squared for {0} parameter combinations using {1} cores...".format(points.shape[0],pool.size))
	else:
		split_chunks = None
		logging.debug("Computing chi squared for {0} parameter combinations using 1 core...".format(points.shape[0]))
	
	chi_squared = analysis.chi2(points,observed_feature=observed_feature,features_covariance=features_covariance,pool=pool,split_chunks=split_chunks)

	#Close MPI Pool
	if pool is not None:
		pool.close()

	#save output
	likelihoods_dir = os.path.join(options.get("analysis","save_path"),"likelihoods")
	if not os.path.isdir(likelihoods_dir):
		os.mkdir(likelihoods_dir)
	
	chi2_file = os.path.join(likelihoods_dir,"chi2_{0}.npy".format(output_string(feature_string)))
	likelihood_file = os.path.join(likelihoods_dir,"likelihood_{0}.npy".format(output_string(feature_string)))

	logging.debug("Saving chi2 to {0}".format(chi2_file))
	np.save(chi2_file,chi_squared.reshape(Om.shape + w.shape + si8.shape))

	logging.debug("Saving full likelihood to {0}".format(likelihood_file))
	np.save(likelihood_file,analysis.likelihood(chi_squared.reshape(Om.shape + w.shape + si8.shape)))

	logging.info("DONE!!")
