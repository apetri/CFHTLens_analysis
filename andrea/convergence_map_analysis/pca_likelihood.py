from __future__ import print_function,division,with_statement

import os,sys
import argparse
import logging
import time

#################################################################################
####################LensTools functionality######################################
#################################################################################

from lenstools.simulations import CFHTemu1,CFHTcov
from lenstools.observations import CFHTLens
from lenstools.constraints import LikelihoodAnalysis
from lenstools.utils import pca_transform

#################################################################################
####################Borrow the FeatureLoader class from train####################
#################################################################################

from train import FeatureLoader
from train import output_string

#################################################################################
####################Borrow the ContourPlot class from contours###################
#################################################################################

from contours import ContourPlot

######################################################################
###################Other functionality################################
######################################################################

import numpy as np
from emcee.utils import MPIPool

#######################################################################
###################DEBUG_PLUS##########################################
#######################################################################

from train import DEBUG_PLUS

######################################################################
###################Main execution#####################################
######################################################################

def main(n_components,cmd_args,pool):

	#################################################################################################################
	#################Info gathering: covariance matrix, observation and emulator#####################################
	#################################################################################################################

	#start
	start = time.time()
	last_timestamp = start

	#Instantiate a FeatureLoader object that will take care of the data loading
	feature_loader = FeatureLoader(cmd_args)

	#Create a LikelihoodAnalysis instance by unpickling one of the emulators
	emulators_dir = os.path.join(feature_loader.options.get("analysis","save_path"),"emulators")
	emulator_file = os.path.join(emulators_dir,"emulator{0}_{1}.p".format(cmd_args.prefix,output_string(feature_loader.feature_string)))
	logging.info("Unpickling emulator from {0}...".format(emulator_file))
	analysis = LikelihoodAnalysis.load(emulator_file)

	#timestamp
	now = time.time()
	logging.info("emulator unpickled in {0:.1f}s".format(now-last_timestamp))
	last_timestamp = now

	######################Compute PCA components here#####################################
	pca = analysis.principalComponents()

	now = time.time()
	logging.info("Principal components computed in {0:.1f}s".format(now-last_timestamp))
	last_timestamp = now

	####################Transform feature space by projecting on PCA eigenvectors############################
	analysis = analysis.transform(pca_transform,pca=pca,n_components=n_components)

	now = time.time()
	logging.info("Projection on first {1} principal components completed in {0:.1f}s".format(now-last_timestamp,analysis.training_set.shape[1]))
	last_timestamp = now

	####################Retrain emulator######################################################################
	analysis.train()

	now = time.time()
	logging.info("Emulator re-training completed in {0:.1f}s".format(now-last_timestamp))
	last_timestamp = now

	###########################################################################################################################################
	###########################################################################################################################################

	#Use this model for the covariance matrix (from the new set of 50 N body simulations)
	covariance_model = CFHTcov.getModels(root_path=feature_loader.options.get("simulations","root_path"))
	logging.info("Measuring covariance matrix from model {0}".format(covariance_model))
	
	#Load in the covariance matrix
	fiducial_feature_ensemble = feature_loader.load_features(covariance_model)

	#If options is enabled, use only the first N realizations to estimate the covariance matrix
	if cmd_args.realizations:

		first_realization = feature_loader.options.getint("mocks","first_realization")
		last_realization = feature_loader.options.getint("mocks","last_realization")

		logging.info("Using only the realizations {0}-{1} to build the fiducial ensemble".format(first_realization,last_realization))
		fiducial_feature_ensemble = fiducial_feature_ensemble.subset(range(first_realization-1,last_realization))
		assert fiducial_feature_ensemble.num_realizations==last_realization-first_realization+1


	###############Insert PCA transform here##############################
	fiducial_feature_ensemble = fiducial_feature_ensemble.transform(pca_transform,pca=pca,n_components=n_components)

	now = time.time()
	logging.info("Projection on first {1} principal components for covariance ensemble completed in {0:.1f}s".format(now-last_timestamp,analysis.training_set.shape[1]))
	last_timestamp = now

	fiducial_feature = fiducial_feature_ensemble.mean()
	features_covariance = fiducial_feature_ensemble.covariance()

	#timestamp
	now = time.time()
	logging.info("covariance computed in {0:.1f}s".format(now-last_timestamp))
	last_timestamp = now

	################################################################################################################################################

	#Get also the observation instance

	if cmd_args.observations_mock:

		logging.info("Using fiducial ensemble as mock observations")
		
		if cmd_args.realization_pick is not None:
			logging.info("Using realization {0} as data".format(cmd_args.realization_pick))
			observed_feature = fiducial_feature_ensemble[cmd_args.realization_pick]
		else:
			observed_feature=fiducial_feature

	else:
		observation = CFHTLens(root_path=feature_loader.options.get("observations","root_path"))
		logging.info("Measuring the observations from {0}".format(observation))

		#And load the observations
		observed_feature_ensemble = feature_loader.load_features(observation)

		###############Insert PCA transform here##############################
		observed_feature_ensemble = observed_feature_ensemble.transform(pca_transform,pca=pca,n_components=n_components)

		now = time.time()
		logging.info("Projection on first {1} principal components for observation completed in {0:.1f}s".format(now-last_timestamp,analysis.training_set.shape[1]))
		last_timestamp = now

		observed_feature = observed_feature_ensemble.mean()

	#timestamp
	now = time.time()
	logging.info("observation loaded in {0:.1f}s".format(now-last_timestamp))
	last_timestamp = now

	################################################################################################################################################
	################################################################################################################################################
	#############Everything is projected on the PCA components now, ready for chi2 computations#####################################################
	################################################################################################################################################
	################################################################################################################################################

	logging.info("Initializing chi2 meshgrid...")

	#Set the points in parameter space on which to compute the chi2 (read from options)
	Om = np.ogrid[feature_loader.options.getfloat("Omega_m","min"):feature_loader.options.getfloat("Omega_m","max"):feature_loader.options.getint("Omega_m","num_points")*1j]
	w = np.ogrid[feature_loader.options.getfloat("w","min"):feature_loader.options.getfloat("w","max"):feature_loader.options.getint("w","num_points")*1j]
	si8 = np.ogrid[feature_loader.options.getfloat("sigma8","min"):feature_loader.options.getfloat("sigma8","max"):feature_loader.options.getint("sigma8","num_points")*1j]

	num_points = len(Om) * len(w) * len(si8) 

	points = np.array(np.meshgrid(Om,w,si8,indexing="ij")).reshape(3,num_points).transpose()
	
	#Now compute the chi2 at each of these points
	if pool:
		split_chunks = pool.size
		logging.info("Computing chi squared for {0} parameter combinations using {1} cores...".format(points.shape[0],pool.size))
	else:
		split_chunks = None
		logging.info("Computing chi squared for {0} parameter combinations using 1 core...".format(points.shape[0]))
	
	chi_squared = analysis.chi2(points,observed_feature=observed_feature,features_covariance=features_covariance,pool=pool,split_chunks=split_chunks)

	now = time.time()
	logging.info("chi2 calculations completed in {0:.1f}s".format(now-last_timestamp))
	last_timestamp = now

	#save output
	likelihoods_dir = os.path.join(feature_loader.options.get("analysis","save_path"),"likelihoods")
	if not os.path.isdir(likelihoods_dir):
		os.mkdir(likelihoods_dir)

	#Output filename formatting
	output_prefix=cmd_args.prefix
	
	if cmd_args.observations_mock:
		output_prefix+="mock"

	if cmd_args.realization_pick is not None:
		output_prefix+="real{0}".format(cmd_args.realization_pick)
	
	if cmd_args.realizations:
		output_prefix+="{0}-{1}".format(first_realization,last_realization)
	
	chi2_file = os.path.join(likelihoods_dir,"chi2{0}_{1}_ncomp{2}.npy".format(output_prefix,output_string(feature_loader.feature_string),n_components))
	likelihood_file = os.path.join(likelihoods_dir,"likelihood{0}_{1}_ncomp{2}.npy".format(output_prefix,output_string(feature_loader.feature_string),n_components))

	logging.info("Saving chi2 to {0}".format(chi2_file))
	np.save(chi2_file,chi_squared.reshape(Om.shape + w.shape + si8.shape))

	logging.info("Saving full likelihood to {0}".format(likelihood_file))
	likelihood_cube = analysis.likelihood(chi_squared.reshape(Om.shape + w.shape + si8.shape))
	np.save(likelihood_file,likelihood_cube)


if __name__=="__main__":

	#################################################
	############Option parsing#######################
	#################################################

	#Parse command line options
	parser = argparse.ArgumentParser()
	parser.add_argument("-f","--file",dest="options_file",action="store",type=str,help="analysis options file")
	parser.add_argument("-v","--verbose",dest="verbose",action="store_true",default=False,help="turn on verbosity")
	parser.add_argument("-vv","--verbose_plus",dest="verbose_plus",action="store_true",default=False,help="turn on additional verbosity")
	parser.add_argument("-m","--mask_scale",dest="mask_scale",action="store_true",default=False,help="scale peaks and power spectrum to unmasked area")
	parser.add_argument("-c","--cut_convergence",dest="cut_convergence",action="store",default=None,help="select convergence values in (min,max) to compute the likelihood. Safe for single descriptor only!!")
	parser.add_argument("-g","--group_subfields",dest="group_subfields",action="store_true",default=False,help="group feature realizations by taking the mean over subfields, this makes a big difference in the covariance matrix")
	parser.add_argument("-p","--prefix",dest="prefix",action="store",default="",help="prefix of the emulator to pickle")
	parser.add_argument("-r","--realizations",dest="realizations",action="store_true",default=False,help="use only a realization subset to build the fiducial ensemble (read from options file)")
	parser.add_argument("-rr","--realization_pick",dest="realization_pick",action="store",type=int,default=None,help="use this particular realization as data")
	parser.add_argument("-d","--differentiate",dest="differentiate",action="store_true",default=False,help="differentiate the first minkowski functional to get the PDF")
	parser.add_argument("-o","--observations_mock",dest="observations_mock",action="store_true",default=False,help="use the fiducial simulations as mock observations")

	cmd_args = parser.parse_args()

	if cmd_args.options_file is None:
		parser.print_help()
		sys.exit(0)

	#Set verbosity level
	if cmd_args.verbose_plus:
		logging.basicConfig(level=DEBUG_PLUS)
	elif cmd_args.verbose:
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

	if pool is not None:
		logging.info("Started MPI Pool.")

	test_components = [3,4,5,6,8,10,20,30,40,50]
	for n_components in test_components:
		main(n_components,cmd_args,pool)


	#Close MPI Pool
	if pool is not None:
		pool.close()
		logging.info("Closed MPI Pool.")
