from __future__ import print_function,division,with_statement

import sys
import argparse,ConfigParser

######################################################################
##################LensTools functionality#############################
######################################################################

from lenstools.simulations import CFHTemu1
from lenstools.defaults import convergence_measure_all 
from lenstools.index import Indexer,PowerSpectrum,PDF,Peaks,MinkowskiAll,Moments

########################################################################
##################Other functionality###################################
########################################################################

import numpy as np
from astropy.io import fits

##########################################################################################################################
##################FITS loader for the maps, must set angle explicitely since it's not contained in the header#############
##########################################################################################################################

def cfht_fits_loader(*args):

	kappa_file = fits.open(args[0])
	angle = 3.4641016151377544

	kappa = kappa_file[0].data.astype(np.float)

	kappa_file.close()

	return angle,kappa

######################################################################################
##########Measurement object, handles the feature measurements from the maps##########
######################################################################################

class Measurement(object):

	"""
	Class handler for the maps feature measurements
	
	"""

	def __init__(self,model,options,measurer):

		self.model = model
		self.options = options
		self.measurer = measurer



#######################################################
###############Main execution##########################
#######################################################

if __name__=="__main__":

	#Parse command line options
	parser = argparse.ArgumentParser()
	parser.add_argument("-f","--file",dest="options_file",action="store",type=str,help="analysis options file")

	cmd_args = parser.parse_args()

	if cmd_args.options_file is None:
		parser.print_help()
		sys.exit(0)

	#Parse INI options file
	options = ConfigParser.ConfigParser()
	with open(cmd_args.options_file,"r") as configfile:
		options.readfp(configfile)

	#Get the names of all the simulated models available for the CFHT analysis
	all_simulated_models = CFHTemu1.getModels(root_path=options.get("simulations","root_path"))

	#Build an Indexer instance, that will contain info on all the features to measure, including binning, etc... (read from options)
	feature_list = list()

	if options.has_section("power_spectrum"):
		l_edges = np.ogrid[options.getfloat("power_spectrum","lmin"):options.getfloat("power_spectrum","lmax"):(options.getint("power_spectrum","num_bins")+1)*1j]
		feature_list.append(PowerSpectrum(l_edges))

	if options.has_section("moments"):
		feature_list.append(Moments())

	if options.has_section("peaks"):
		th_peaks = np.ogrid[options.getfloat("peaks","th_min"):options.getfloat("peaks","th_max"):(options.getint("peaks","num_bins")+1)*1j]
		feature_list.append(Peaks(th_peaks))

	if options.has_section("minkowski_functionals"):
		th_minkowski = np.ogrid[options.getfloat("minkowski_functionals","th_min"):options.getfloat("minkowski_functionals","th_max"):(options.getint("minkowski_functionals","num_bins")+1)*1j]
		feature_list.append(MinkowskiAll(th_minkowski))

	idx = Indexer.stack(feature_list)

	#Cycle through the models and perform the measurements of the selected features



