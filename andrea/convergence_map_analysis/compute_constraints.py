from __future__ import print_function,division,with_statement

from operator import mul
from functools import reduce

import os,sys
import argparse,ConfigParser
import logging

######################################################################
##################LensTools functionality#############################
######################################################################

from lenstools import Ensemble
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

######################################################################
###################Main execution#####################################
######################################################################

if __name__=="__main__":

	#Parse command line options
	parser = argparse.ArgumentParser()
	parser.add_argument("-f","--file",dest="options_file",action="store",type=str,help="analysis options file")
	parser.add_argument("-v","--verbose",dest="verbose",action="store_true",default=False,help="turn on verbosity")

	cmd_args = parser.parse_args()

	if cmd_args.options_file is None:
		parser.print_help()
		sys.exit(0)

	#Set verbosity level
	if cmd_args.verbose:
		logging.basicConfig(level=logging.DEBUG)
	else:
		logging.basicConfig(level=logging.INFO)

	#Parse INI options file
	options = ConfigParser.ConfigParser()
	with open(cmd_args.options_file,"r") as configfile:
		options.readfp(configfile)

	#Read the save path from options
	save_path = options.get("analysis","save_path")
	#Load feature index
	l = np.load(os.path.join(save_path,"ell.npy"))

	#Get the names of all the simulated models available for the CFHT analysis, including smoothing scales and subfields
	all_simulated_models = CFHTemu1.getModels(root_path=options.get("simulations","root_path"))

	#Get also the observation model instance
	observed_model = CFHTLens(root_path=options.get("observations","root_path"))

	#Select subset of training models
	training_models = all_simulated_models[:17]
	#Use this model for the covariance matrix
	covariance_model = 16
	
	#Parse from options which subfields and smoothing scale to consider
	subfields = [ int(subfield) for subfield in options.get("analysis","subfields").split(",") ]
	smoothing_scale = options.getfloat("analysis","smoothing_scale")

	#Parse from options which type of descriptors to use
	feature_types = options.get("analysis","feature_types").split(",")

	#Create a LikelihoodAnalysis instance and load the training models into it
	analysis = LikelihoodAnalysis()
	
	for n,model in enumerate(training_models):

		#First create an empty ensemble
		ensemble_all_subfields = Ensemble()

		#Then cycle through all the subfields and gather the features for each one
		for subfield in subfields:
			
			m = Measurement(model=model,options=options,subfield=subfield,smoothing_scale=smoothing_scale,measurer=None)
			m.get_all_map_names()

			#Load the measured features
			ensemble_subfield = [ Ensemble.fromfilelist([os.path.join(m.full_save_path,feature_type + ".npy")]) for feature_type in feature_types ]
			for ens in ensemble_subfield: 
				ens.load(from_old=True)

			#Add the features to the cumulative subfield ensemble
			ensemble_all_subfields += reduce(mul,ensemble_subfield)

		#Add the feature to the LikelihoodAnalysis
		analysis.add_model(parameters=model.squeeze(),feature=ensemble_all_subfields.mean())

		#If this is the correct model, compute the covariance matrix too
		if n==covariance_model:
			features_covariance = ensemble_all_subfields.covariance()

	#Finally, measure the observed feature
	ensemble_all_subfields = Ensemble()

	for subfield in subfields:

		m = Measurement(model=observed_model,options=options,subfield=subfield,smoothing_scale=smoothing_scale,measurer=None)
		m.get_all_map_names()

		#Load the measured feature
		ensemble_subfield = [ Ensemble.fromfilelist([os.path.join(m.full_save_path,feature_type + ".npy")]) for feature_type in feature_types]
		for ens in ensemble_subfield:
			ens.load(from_old=True)

		#Add the features to the cumulative subfield ensemble
		ensemble_all_subfields += reduce(mul,ensemble_subfield)

	#Compute the average over subfields
	observed_feature = ensemble_all_subfields.mean()

	################################################################################################
	#####################Feature loading complete, ready for analysis###############################
	################################################################################################

	#Train the interpolators using the simulated features
	logging.debug("Training interpolators...")
	analysis.train()

	#Set the points in parameter space on which to compute the chi2 
	Om = np.ogrid[options.getfloat("Omega_m","min"):options.getfloat("Omega_m","max"):options.getint("Omega_m","num_points")*1j]
	w = np.ogrid[options.getfloat("w","min"):options.getfloat("w","max"):options.getint("w","num_points")*1j]
	si8 = np.ogrid[options.getfloat("sigma8","min"):options.getfloat("sigma8","max"):options.getint("sigma8","num_points")*1j]

	num_points = len(Om) * len(w) * len(si8) 

	points = np.array(np.meshgrid(Om,w,si8)).reshape(3,num_points).transpose()

	#Now compute the chi2 at each of these points
	logging.debug("Computing chi squared...")
	chi_squared = analysis.chi2(points,observed_feature=observed_feature,features_covariance=features_covariance)

	#save output
	np.save("likelihood_{0}.npy".format("-".join(feature_types)),analysis.likelihood(chi_squared.reshape(Om.shape + w.shape + si8.shape)))

