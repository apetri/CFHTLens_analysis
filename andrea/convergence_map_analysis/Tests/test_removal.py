from __future__ import print_function,division,with_statement

import os,sys
import argparse
import logging
import time

#################################################################################
####################LensTools functionality######################################
#################################################################################

from lenstools.simulations import CFHTemu1
from lenstools.observations import CFHTLens
from lenstools.constraints import LikelihoodAnalysis

#################################################################################
####################Borrow the FeatureLoader class from train####################
#################################################################################

from train import FeatureLoader
from train import output_string

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

def main():

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
	parser.add_argument("-s","--save_points",dest="save_points",action="store",default=None,help="save points in parameter space to external npy file")
	parser.add_argument("-ss","--save_debug",dest="save_debug",action="store_true",default=False,help="save a bunch of debugging info for the analysis")
	parser.add_argument("-p","--prefix",dest="prefix",action="store",default="",help="prefix of the emulator to pickle")
	parser.add_argument("-r","--remove",dest="remove",action="store",type=int,default=24,help="model to remove from the analysis")

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

	#################################################################################################################
	#################Info gathering: covariance matrix, observation and emulator#####################################
	#################################################################################################################

	#start
	start = time.time()
	last_timestamp = start

	#Instantiate a FeatureLoader object that will take care of the memory loading
	feature_loader = FeatureLoader(cmd_args)

	###########################################################################################################################################

	#Get the names of all the simulated models available for the CFHT analysis, including smoothing scales and subfields
	all_simulated_models = CFHTemu1.getModels(root_path=feature_loader.options.get("simulations","root_path"))

	#Use this model for the covariance matrix
	covariance_model = all_simulated_models[feature_loader.options.getint("analysis","covariance_model") - 1]
	#Load in the covariance matrix
	features_covariance = feature_loader.load_features(covariance_model).covariance()

	#timestamp
	now = time.time()
	logging.info("covariance loaded in {0:.1f}s".format(now-last_timestamp))
	last_timestamp = now

	################################################################################################################################################

	#Create a LikelihoodAnalysis instance by unpickling one of the emulators
	emulators_dir = os.path.join(feature_loader.options.get("analysis","save_path"),"emulators")
	emulator_file = os.path.join(emulators_dir,"emulator{0}_{1}.p".format(cmd_args.prefix,output_string(feature_loader.feature_string)))
	logging.info("Unpickling emulator from {0}...".format(emulator_file))
	analysis = LikelihoodAnalysis.load(emulator_file)

	#timestamp
	now = time.time()
	logging.info("emulator unpickled in {0:.1f}s".format(now-last_timestamp))
	last_timestamp = now

	##################################################################################################################################################

	#Remove one of the models from the emulator, and treat it as data to fit
	model_to_remove = all_simulated_models[cmd_args.remove]

	logging.info("Removing model {0} from emulator training".format(model_to_remove))
	
	parameters_to_remove = model_to_remove.squeeze()
	remove_index = analysis.find(parameters_to_remove)[0]
	
	logging.info("Removing model {0} with parameters {1} from emulator...".format(remove_index,analysis.parameter_set[remove_index]))
	analysis.remove_model(remove_index)

	#Retrain without the removed model
	analysis.train()

	#Treat the removed model as data
	logging.info("Treating model {0} as data, loading features...".format(model_to_remove))
	observed_feature = feature_loader.load_features(model_to_remove).mean()

	####################################################################################################################
	######################################Compute the chi2 cube#########################################################
	####################################################################################################################

	logging.info("Initializing chi2 meshgrid...")

	#Set the points in parameter space on which to compute the chi2 (read from options)
	Om = np.ogrid[feature_loader.options.getfloat("Omega_m","min"):feature_loader.options.getfloat("Omega_m","max"):feature_loader.options.getint("Omega_m","num_points")*1j]
	w = np.ogrid[feature_loader.options.getfloat("w","min"):feature_loader.options.getfloat("w","max"):feature_loader.options.getint("w","num_points")*1j]
	si8 = np.ogrid[feature_loader.options.getfloat("sigma8","min"):feature_loader.options.getfloat("sigma8","max"):feature_loader.options.getint("sigma8","num_points")*1j]

	num_points = len(Om) * len(w) * len(si8) 

	points = np.array(np.meshgrid(Om,w,si8,indexing="ij")).reshape(3,num_points).transpose()
	if cmd_args.save_points is not None:
		logging.info("Saving points to {0}.npy".format(cmd_args.save_points.rstrip(".npy")))
		np.save(cmd_args.save_points.rstrip(".npy")+".npy",points)

	#Now compute the chi2 at each of these points
	if pool:
		split_chunks = pool.size
		logging.info("Computing chi squared for {0} parameter combinations using {1} cores...".format(points.shape[0],pool.size))
	else:
		split_chunks = None
		logging.info("Computing chi squared for {0} parameter combinations using 1 core...".format(points.shape[0]))
	
	chi_squared = analysis.chi2(points,observed_feature=observed_feature,features_covariance=features_covariance,pool=pool,split_chunks=split_chunks)

	#Close MPI Pool
	if pool is not None:
		pool.close()
		logging.info("Closed MPI Pool.")

	now = time.time()
	logging.info("chi2 calculations completed in {0:.1f}s".format(now-last_timestamp))
	last_timestamp = now

	#Save output

	likelihood_file = "likelihood_remove{0}_{1}.npy".format(cmd_args.remove,output_string(feature_loader.feature_string))
	chi2_file = "chi2_remove{0}_{1}.npy".format(cmd_args.remove,output_string(feature_loader.feature_string))

	logging.info("Saving chi2 to {0}".format(chi2_file))
	np.save(chi2_file,chi_squared.reshape(Om.shape + w.shape + si8.shape))

	logging.info("Saving full likelihood to {0}".format(likelihood_file))
	np.save(likelihood_file,analysis.likelihood(chi_squared.reshape(Om.shape + w.shape + si8.shape)))

	end = time.time()

	logging.info("DONE!!")
	logging.info("Completed in {0:.1f}s".format(end-start))


if __name__=="__main__":
	main()
