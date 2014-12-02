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
	parser.add_argument("-r","--realizations",dest="realizations",type=int,default=None,help="use only the first N realizations to estimate the covariance matrix")
	parser.add_argument("-d","--differentiate",dest="differentiate",action="store_true",default=False,help="differentiate the first minkowski functional to get the PDF")

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

	#Create a LikelihoodAnalysis instance by unpickling one of the emulators
	emulators_dir = os.path.join(feature_loader.options.get("analysis","save_path"),"emulators")
	emulator_file = os.path.join(emulators_dir,"emulator{0}_{1}.p".format(cmd_args.prefix,output_string(feature_loader.feature_string)))
	logging.info("Unpickling emulator from {0}...".format(emulator_file))
	analysis = LikelihoodAnalysis.load(emulator_file)

	#Return
	return fiducial_feature_ensemble,analysis


if __name__=="__main__":
	fiducial_feature_ensemble,analysis = main()