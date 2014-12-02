from __future__ import print_function,division,with_statement

from operator import mul
from functools import reduce

import os,sys,re
import argparse,ConfigParser
import logging
import time

######################################################################
##################LensTools functionality#############################
######################################################################

from lenstools import Ensemble
from lenstools.index import Indexer,MinkowskiAll
from lenstools.constraints import LikelihoodAnalysis
from lenstools.simulations import CFHTemu1

#################################################################################
####################Borrow the Measurer class from measure_features##############
#################################################################################

from measure_features import Measurement

######################################################################
###################Other functionality################################
######################################################################

import numpy as np

#######################################################################
###################DEBUG_PLUS##########################################
#######################################################################

DEBUG_PLUS = 5

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

##########################################################################################################################
#########This is the main class that takes care of loading the features in memory and construct the Ensembles#############
##########################################################################################################################

class FeatureLoader(object):

	################################################################################
	##############Constructor takes care of options parsing#########################
	################################################################################

	def __init__(self,cmd_args):

		self.cmd_args = cmd_args

		#Parse INI options file
		logging.info("Parsing options from {0}".format(cmd_args.options_file))

		self.options = ConfigParser.ConfigParser()
		with open(cmd_args.options_file,"r") as configfile:
			self.options.readfp(configfile)

		#Read the save path from options
		self.save_path = self.options.get("analysis","save_path")

		#Construct an index for the minkowski functionals, it will be useful for later
		th_minkowski = np.ogrid[self.options.getfloat("minkowski_functionals","th_min"):self.options.getfloat("minkowski_functionals","th_max"):(self.options.getint("minkowski_functionals","num_bins")+1)*1j]
		mink_idx = MinkowskiAll(thresholds=th_minkowski).separate()
		self.mink_idx = Indexer(mink_idx)

		#Load the peaks and minkowski thresholds
		self.kappa_peaks = np.load(os.path.join(self.save_path,"th_peaks.npy"))
		self.kappa_minkowski = np.load(os.path.join(self.save_path,"th_minkowski.npy")) 

		#Parse convergence cuts from command line
		if cmd_args.cut_convergence is not None:
			self.kappa_min,self.kappa_max = [ float(kappa_lim.lstrip("\\")) for kappa_lim in cmd_args.cut_convergence.split(",") ]
			assert self.kappa_min<self.kappa_max
	
		#Parse from options which subfields and smoothing scale to consider
		self.subfields = [ int(subfield) for subfield in self.options.get("analysis","subfields").split(",") ]

		#Parse from options which type of descriptors (features) to use
		self.feature_string = self.options.get("analysis","feature_types")
		self.smoothing_scales,self.features_to_measure = parse_features(self.feature_string)

		#Get masked area information
		self.get_masked_fractions()

	#################################################################################################################################################
	##################################We need to gather all the masked area fractions, for all smoothing scales######################################
	#################################################################################################################################################

	def get_masked_fractions(self):

		logging.info("Gathering masked area fractions...")

		#It's convenient to group those in a two level dictionary, indexed by smoothing scale and subfield
		self.masked_fraction = dict()
		self.total_non_masked_fraction = dict()

		#Loop over smoothing scales and subfields
		for smoothing_scale in self.smoothing_scales:

			self.masked_fraction[smoothing_scale] = dict()
			self.total_non_masked_fraction[smoothing_scale] = 0.0

			for subfield in self.subfields:

				m = Measurement(model=None,options=self.options,subfield=subfield,smoothing_scale=smoothing_scale,measurer=None)
				
				self.masked_fraction[smoothing_scale][subfield] = m.maskedFraction
				logging.debug("Masked fraction of subfield {0}, {1} arcmin smoothing is {2}".format(subfield,smoothing_scale,self.masked_fraction[smoothing_scale][subfield]))

				self.total_non_masked_fraction[smoothing_scale] += 1.0 - self.masked_fraction[smoothing_scale][subfield]

			logging.debug("Total non masked area fraction of CFHT subfields with {0} arcmin smoothing is {1}".format(smoothing_scale,self.total_non_masked_fraction[smoothing_scale]))

		#Finished gathering the info
		logging.info("Gathered masked area info.")



	#################################################################################################################################################
	##################################This is the function that does the dirty work, probably all you need to care about#############################
	#################################################################################################################################################

	def load_features(self,model,save_new=False):

		#First create an empty ensemble
		ensemble_all_subfields = Ensemble()

		#Then cycle through all the subfields and gather the features for each one
		for subfield in self.subfields:
		
			#Dictionary that holds all the measurements
			m = dict()

			for smoothing_scale in self.smoothing_scales:
				m[smoothing_scale] = Measurement(model=model,options=self.options,subfield=subfield,smoothing_scale=smoothing_scale,measurer=None)
				m[smoothing_scale].get_all_map_names()

			#Construct one ensemble for each feature (with included smoothing scales) and load in the data
			ensemble_subfield = list()
			
			#Prevent randomness in dictionary keys
			features_to_measure = self.features_to_measure.keys()
			features_to_measure.sort()

			for feature_type in features_to_measure:
			
				for smoothing_scale in self.features_to_measure[feature_type]:
					
					#Construct the subfield/smoothing scale/feature specific ensemble
					ens = Ensemble.fromfilelist([os.path.join(m[smoothing_scale].full_save_path,npy_filename(feature_type))])
					ens.load(from_old=True)

					#Check if we want to cut out some of the peaks
					if self.cmd_args.cut_convergence and feature_type=="peaks":
						new_thresholds = ens.cut(self.kappa_min,self.kappa_max,feature_label=self.kappa_peaks)
						logging.log(DEBUG_PLUS,"Performed cut on the peaks convergence, new limits are {0},{1}".format(new_thresholds[0],new_thresholds[-1]))
						if save_new:
							logging.info("Saving new kappa values to {0}...".format(os.path.join(self.save_path,"th_new_peaks.npy")))
							np.save(os.path.join(self.save_path,"th_new_peaks.npy"),new_thresholds)

					#Check the masked fraction of the field of view
					masked_fraction = self.masked_fraction[smoothing_scale][subfield]

					###########################################################################################################################################################################################################
					#Scale to the non masked area: if we treat each subfield independently (i.e. group_subfields is False) then we need to scale each subfield to the same area when considering the power spectrum and peaks##
					#if on the other hand we group subfields together, then the power spectrum and peaks are simply added between subfields, but the MFs and the moments need to be scaled#####################################
					###########################################################################################################################################################################################################

					if (self.cmd_args.mask_scale) and not(self.cmd_args.group_subfields):
					
						if feature_type=="power_spectrum":
							logging.log(DEBUG_PLUS,"Scaling power spectrum of subfield {0}, masked fraction {1}, multiplying by {2}".format(subfield,masked_fraction,1.0/(1.0 - masked_fraction)**2))
							ens.scale(1.0/(1.0 - masked_fraction)**2)
						elif feature_type=="peaks":
							logging.log(DEBUG_PLUS,"Scaling peak counts of subfield {0}, masked fraction {1}, multiplying by {2}".format(subfield,masked_fraction,1.0/(1.0 - masked_fraction)))
							ens.scale(1.0/(1.0 - masked_fraction))

					elif (self.cmd_args.mask_scale) and (self.cmd_args.group_subfields):

						if "minkowski" in feature_type or "moments" in feature_type:
							logging.log(DEBUG_PLUS,"Scaling {0} of subfield {1}, masked fraction {2}, multiplying by {3}".format(feature_type,subfield,masked_fraction,(1.0 - masked_fraction)/self.total_non_masked_fraction[smoothing_scale]))
							ens.scale((1.0 - masked_fraction)/self.total_non_masked_fraction[smoothing_scale])

					###############################################################################
					##MFs only: check if we want to discard some of the Minkowski functionals######
					###############################################################################

					num = re.match(r"minkowski_([0-2]+)",feature_type)
					if num is not None:
						
						mink_to_measure = [ int(n_mf) for n_mf in list(num.group(1)) ]
						ens_split = ens.split(self.mink_idx)

						#Differentiate Minkowski 0 to find the PDF?
						if self.cmd_args.differentiate:
							logging.log(DEBUG_PLUS,"Differentiating Minkowski 0 to get the PDF")
							ens_split[0] = ens_split[0].differentiate(step=self.kappa_minkowski[0]-self.kappa_minkowski[1])

						#Perform the convergence cut if option is enabled
						if self.cmd_args.cut_convergence:
							new_thresholds = [ ens_split[n_mf].cut(self.kappa_min,self.kappa_max,feature_label=self.kappa_minkowski) for n_mf in mink_to_measure ]
							logging.log(DEBUG_PLUS,"Performed cut on the minkowski convergence, new limits are {0},{1}".format(new_thresholds[0][0],new_thresholds[0][-1]))
							if save_new:
								logging.info("Saving new kappa values to {0}...".format(os.path.join(self.save_path,"th_new_minkowski.npy")))
								np.save(os.path.join(self.save_path,"th_new_minkowski.npy"),new_thresholds[0])
					
						[ ensemble_subfield.append(ens_split[n_mf]) for n_mf in mink_to_measure ]
				
					else:
					
						ensemble_subfield.append(ens)
						if self.cmd_args.cut_convergence:
							logging.log(DEBUG_PLUS,"Convergence cut on MFs not performed, select minkowski_012 instead of minkowski_all")

					#############################################################################################


			#Add the features to the cumulative subfield ensemble
			ensemble_all_subfields += reduce(mul,ensemble_subfield)

		#If option is specified, group all the subfields together, for each realization
		if self.cmd_args.group_subfields:
			logging.log(DEBUG_PLUS,"Taking means over the {0} subfields...".format(len(self.subfields)))
			ensemble_all_subfields.group(group_size=len(self.subfields),kind="sparse")

		#Return the created ensemble
		return ensemble_all_subfields

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
	parser.add_argument("-p","--prefix",dest="prefix",action="store",default="",help="give a prefix to the name of the pickled emulator")
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

	#Instantiate a FeatureLoader object that will take care of the memory loading
	feature_loader = FeatureLoader(cmd_args)

	#Get the names of all the simulated models available for the CFHT analysis, including smoothing scales and subfields
	all_simulated_models = CFHTemu1.getModels(root_path=feature_loader.options.get("simulations","root_path"))

	#Select subset of training models
	training_models = all_simulated_models

	#Create a LikelihoodAnalysis instance and load the training models into it
	analysis = LikelihoodAnalysis()

	###########################################################
	###############Feature loading#############################
	###########################################################

	#Start loading the data
	logging.info("Loading features...")
	
	for feature_type in feature_loader.features_to_measure.keys():
		logging.info("{0}, smoothing scales: {1} arcmin".format(feature_type,",".join([ str(s) for s in feature_loader.features_to_measure[feature_type] ])))
	
	#Start
	start = time.time()

	#Load the simulated features
	for n,model in enumerate(training_models):

		logging.debug("Model {0}".format(n))
		logging.debug(model)

		ensemble_all_subfields = feature_loader.load_features(model)

		#Add the feature to the LikelihoodAnalysis
		analysis.add_model(parameters=model.squeeze(),feature=ensemble_all_subfields.mean())

	#Log timestamp
	now = time.time()
	logging.info("Simulated features loaded in {0:.1f}s".format(now-start))
	last_timestamp = now

	########################################################################################################
	#####################Feature loading complete, can build the emulator now###############################
	########################################################################################################

	#Train the interpolators using the simulated features
	logging.info("Training interpolators...")
	analysis.train()

	#Log timestamp
	now = time.time()
	logging.info("Emulator trained in {0:.1f}s".format(now-last_timestamp))
	last_timestamp = now

	#Pickle the emulator and save it to a .p file

	emulators_dir = os.path.join(feature_loader.options.get("analysis","save_path"),"emulators")
	if not os.path.isdir(emulators_dir):
		os.mkdir(emulators_dir)
	
	emulator_file = os.path.join(emulators_dir,"emulator{0}_{1}.p".format(cmd_args.prefix,output_string(feature_loader.feature_string)))
	logging.info("Pickling emulator and saving it to {0}".format(emulator_file))
	analysis.save(emulator_file)

	#Log timestamp and finish
	end = time.time()

	logging.info("DONE!!")
	logging.info("Completed in {0:.1f}s".format(end-start))

if __name__=="__main__":
	main()
