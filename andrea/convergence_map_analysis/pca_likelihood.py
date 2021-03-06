from __future__ import print_function,division,with_statement

import os,sys
import argparse,ConfigParser
import logging
import time
import json

from operator import add,mul
from functools import reduce

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
################FeatureLoaderCross####################################
######################################################################

class FeatureLoaderCross(FeatureLoader):

	@classmethod
	def fromArgs(cls,cmd_args):

		feature_loader = cls(cmd_args)

		#Check if -x option is specified
		if cmd_args.cross:

			feature_types = feature_loader.options.get("analysis","feature_types").replace(" ","").split("*")
			feature_loader_collection = list()
			
			for feature_type in feature_types:
				feature_loader_collection.append(cls(cmd_args,feature_string=feature_type))

			return feature_loader_collection

		else:
			return [feature_loader]


#####################################################################
###########Emulator reparametrizations###############################
#####################################################################

def Sigma8reparametrize(p,a=0.55):

	q = p.copy()

	#Change only the last parameter
	q[:,2] = p[:,2]*(p[:,0]/0.27)**a

	#Done
	return q


####################################################################################
###########Dictionary for emulator reparametrizations###############################
####################################################################################

reparametrization = dict()
reparametrization["Omega_m-w-sigma8"] = None 
reparametrization["Omega_m-w-Sigma8Om0.55"] = Sigma8reparametrize


######################################################################
###################Main execution#####################################
######################################################################

def main(n_components_collection,cmd_args,pool):

	#################################################################################################################
	#################Info gathering: covariance matrix, observation and emulator#####################################
	#################################################################################################################

	#start
	start = time.time()
	last_timestamp = start

	#Instantiate a FeatureLoader object that will take care of the data loading
	feature_loader_collection = FeatureLoaderCross.fromArgs(cmd_args)
	fiducial_feature_ensemble_collection = list()
	observed_feature_ensemble_collection = list()
	analysis_collection = list()
	formatted_output_string_collection = list()

	#Sanity check
	if type(n_components_collection)==list:
		assert len(n_components_collection)==len(feature_loader_collection)

	#Cycle over feature types
	for nc,feature_loader in enumerate(feature_loader_collection):

		#Use the same number of components for all or not?
		if type(n_components_collection)==list:
			n_components = n_components_collection[nc]
		else:
			n_components = n_components_collection

		#Format the output string
		formatted_output_string_collection.append(output_string(feature_loader.feature_string)+"_ncomp{0}".format(n_components))

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

		#Append to the collection
		analysis_collection.append(analysis)

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

		#Append to the collection
		fiducial_feature_ensemble_collection.append(fiducial_feature_ensemble)

		#timestamp
		now = time.time()
		logging.info("covariance computed in {0:.1f}s".format(now-last_timestamp))
		last_timestamp = now

		################################################################################################################################################

		#Get also the observation instance

		if cmd_args.observations_mock:

			pass

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

			observed_feature_ensemble_collection.append(observed_feature_ensemble)

		#timestamp
		now = time.time()
		logging.info("observation loaded in {0:.1f}s".format(now-last_timestamp))
		last_timestamp = now


	################################################################################################################################################
	################################Reduce the collections##########################################################################################
	################################################################################################################################################

	analysis = reduce(mul,analysis_collection)
	fiducial_feature_ensemble = reduce(mul,fiducial_feature_ensemble_collection)

	#Sanity check
	if type(n_components_collection)==list:
		assert analysis.training_set.shape[1]==reduce(add,n_components_collection)
		assert fiducial_feature_ensemble.data.shape[1]==reduce(add,n_components_collection)
	else:
		assert analysis.training_set.shape[1]==n_components*len(feature_loader_collection)
		assert fiducial_feature_ensemble.data.shape[1]==n_components*len(feature_loader_collection)

	#Covariance matrix
	features_covariance = fiducial_feature_ensemble.covariance()

	if cmd_args.observations_mock:

		logging.info("Using fiducial ensemble as mock observations")
		
		if cmd_args.realization_pick is not None:
			logging.info("Using realization {0} as data".format(cmd_args.realization_pick))
			observed_feature = fiducial_feature_ensemble[cmd_args.realization_pick]
		else:
			observed_feature=fiducial_feature_ensemble.mean()

	else:

		#And load the observations
		observed_feature_ensemble = reduce(mul,observed_feature_ensemble_collection)
		observed_feature = observed_feature_ensemble.mean()

	#Sanity check
	if type(n_components_collection)==list:
		assert observed_feature.shape[0]==reduce(add,n_components_collection)
	else:
		assert observed_feature.shape[0]==n_components*len(feature_loader_collection)

	################################################################################################################################################
	################################################################################################################################################
	#############Everything is projected on the PCA components now, ready for chi2 computations#####################################################
	################################################################################################################################################
	################################################################################################################################################

	logging.info("Initializing chi2 meshgrid...")

	#Read parameters to use from options
	use_parameters = feature_loader.options.get("parameters","use_parameters").replace(" ","").split(",")
	assert len(use_parameters)==3
	
	#Reparametrization hash key
	use_parameters_hash = "-".join(use_parameters)

	########################################################################################
	#Might need to reparametrize the emulator here, use a dictionary for reparametrizations#
	########################################################################################

	assert use_parameters_hash in reparametrization.keys(),"No reparametrization scheme specified for {0} parametrization".format(use_parameters_hash)
	
	if reparametrization[use_parameters_hash] is not None:
		
		#Reparametrize
		logging.info("Reparametrizing emulator according to {0} parametrization".format(use_parameters_hash))
		analysis.reparametrize(reparametrization[use_parameters_hash])

		#Retrain for safety
		analysis.train()

	#Log current parametrization to user
	logging.info("Using parametrization {0}".format(use_parameters_hash))

	#Set the points in parameter space on which to compute the chi2 (read extremes from options)
	par = list()
	for p in range(3):
		assert feature_loader.options.has_section(use_parameters[p]),"No extremes specified for parameter {0}".format(use_parameters[p])
		par.append(np.ogrid[feature_loader.options.getfloat(use_parameters[p],"min"):feature_loader.options.getfloat(use_parameters[p],"max"):feature_loader.options.getint(use_parameters[p],"num_points")*1j])

	num_points = len(par[0]) * len(par[1]) * len(par[2]) 

	points = np.array(np.meshgrid(par[0],par[1],par[2],indexing="ij")).reshape(3,num_points).transpose()
	
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
	likelihoods_dir = os.path.join(feature_loader.options.get("analysis","save_path"),"likelihoods_{0}".format(use_parameters_hash))
	if not os.path.isdir(likelihoods_dir):
		os.mkdir(likelihoods_dir)

	#Output filename formatting
	output_prefix=""
	
	if cmd_args.observations_mock:
		output_prefix+="mock"

	if cmd_args.cross:
		output_prefix+="_cross"

	if cmd_args.realization_pick is not None:
		output_prefix+="real{0}".format(cmd_args.realization_pick)
	
	if cmd_args.realizations:
		output_prefix+="{0}-{1}".format(first_realization,last_realization)

	output_prefix += cmd_args.prefix 

	formatted_output_string = "-".join(formatted_output_string_collection)
	
	chi2_file = os.path.join(likelihoods_dir,"chi2{0}_{1}.npy".format(output_prefix,formatted_output_string))
	likelihood_file = os.path.join(likelihoods_dir,"likelihood{0}_{1}.npy".format(output_prefix,formatted_output_string))

	logging.info("Saving chi2 to {0}".format(chi2_file))
	np.save(chi2_file,chi_squared.reshape(par[0].shape + par[1].shape + par[2].shape))

	logging.info("Saving full likelihood to {0}".format(likelihood_file))
	likelihood_cube = analysis.likelihood(chi_squared.reshape(par[0].shape + par[1].shape + par[2].shape))
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
	parser.add_argument("-x","--cross",dest="cross",action="store_true",default=False,help="do PCA on each descriptor separately, and co-add after that")

	cmd_args = parser.parse_args()

	if cmd_args.options_file is None:
		parser.print_help()
		sys.exit(0)

	#Need the options here too
	options = ConfigParser.ConfigParser()
	options.read(cmd_args.options_file)

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

	test_components = json.loads(options.get("pca","num_components"))
	for n_components_collection in test_components:
		main(n_components_collection,cmd_args,pool)


	#Close MPI Pool
	if pool is not None:
		pool.close()
		logging.info("Closed MPI Pool.")
