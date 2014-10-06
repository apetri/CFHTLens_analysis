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

	#Use this model for the covariance matrix (from the new set of 50 N body simulations)
	covariance_model = CFHTcov.getModels(root_path=feature_loader.options.get("simulations","root_path"))
	logging.info("Measuring covariance matrix from model {0}".format(covariance_model))
	
	#Load in the covariance matrix
	fiducial_feature_ensemble = feature_loader.load_features(covariance_model)
	fiducial_features = fiducial_feature_ensemble.mean()
	features_covariance = fiducial_feature_ensemble.covariance()

	#timestamp
	now = time.time()
	logging.info("covariance loaded in {0:.1f}s".format(now-last_timestamp))
	last_timestamp = now

	################################################################################################################################################

	#Treat the 50N-body simulation set as data
	observation = CFHTcov.getModels(root_path=feature_loader.options.get("observations","root_path"))
	logging.info("Measuring the observations from {0}".format(observation))
	
	#And load the observations
	observed_feature = feature_loader.load_features(observation)

	#timestamp
	now = time.time()
	logging.info("observation loaded in {0:.1f}s".format(now-last_timestamp))
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

	#Allocate array for best fit
	first_realization = feature_loader.options.getint("mocks","first_realization")
	last_realization = feature_loader.options.getint("mocks","last_realization")
	best_fit_all = np.zeros((last_realization-first_realization+1,analysis.parameter_set.shape[1]))

	#Cycle through the realizations and obtain a best fit for each one of them
	
	for nreal in range(first_realization-1,last_realization):
	
		chi_squared = analysis.chi2(points,observed_feature=observed_feature[nreal],features_covariance=features_covariance,pool=pool,split_chunks=split_chunks)

		now = time.time()
		logging.info("realization {0}, chi2 calculations completed in {1:.1f}s".format(nreal+1,now-last_timestamp))
		last_timestamp = now

		#After chi2, compute the likelihood
		likelihood_cube = analysis.likelihood(chi_squared.reshape(Om.shape + w.shape + si8.shape))


		#Find the maximum of the likelihood using ContourPlot functionality
		contour = ContourPlot()
		contour.getLikelihood(likelihood_cube)
		contour.getUnitsFromOptions(feature_loader.options)
		parameters_maximum = contour.getMaximum()
		parameter_keys = parameters_maximum.keys()
		parameter_keys.sort(key=contour.parameter_axes.get)

		#Display the new best fit before exiting
		best_fit_parameters = np.array([ parameters_maximum[par_key] for par_key in parameter_keys ])
		logging.info("Best fit for realization {3} is [ {0[0]:.2f} {0[1]:.2f} {0[2]:.2f} ], chi2={1[0]:.3f}({2} dof)".format(best_fit_parameters,analysis.chi2(best_fit_parameters,features_covariance=features_covariance,observed_feature=observed_feature[nreal]),analysis.training_set.shape[1],nreal+1))

		#Update global array with best fit parameters
		best_fit_all[nreal,:] = best_fit_parameters.copy()

	#######################################################################################################################################################################
	
	#Close MPI Pool
	if pool is not None:
		pool.close()
		logging.info("Closed MPI Pool.")

	#Save the best fit parameters for all realizations
	best_fit_filename = os.path.join(feature_loader.options.get("analysis","save_path"),"troubleshoot","best_fit_all_{0}.npy".format(output_string(feature_loader.feature_string)))
	logging.info("Saving best fit to {0}...".format(best_fit_filename))
	np.save(best_fit_filename,best_fit_all)

	end = time.time()

	logging.info("DONE!!")
	logging.info("Completed in {0:.1f}s".format(end-start))

##########################################################################################################################################

if __name__=="__main__":
	main()
