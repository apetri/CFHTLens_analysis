from __future__ import print_function,division,with_statement

import os,sys
import argparse,ConfigParser
import logging
import StringIO

######################################################################
##################LensTools functionality#############################
######################################################################

from lenstools.simulations import CFHTemu1
from lenstools.observations import CFHTLens
from lenstools.defaults import convergence_measure_all 
from lenstools.index import Indexer,PowerSpectrum,PDF,Peaks,MinkowskiAll,Moments
from lenstools import Ensemble

########################################################################
##################Other functionality###################################
########################################################################

import numpy as np
from astropy.io import fits

import progressbar

###########################################################################
#############Read INI options file and write summary information###########
###########################################################################

def write_info(options):

	s = StringIO.StringIO()

	s.write("""
Realizations to analyze: 1 to {0}

###########################################

""".format(options.get("analysis","num_realizations")))

	s.write("""Implemented descriptors
-------------------

""")

	if options.has_section("power_spectrum"):
		s.write("""Power spectrum: {0} bins between l={1} and l={2}\n\n""".format(options.get("power_spectrum","num_bins"),options.get("power_spectrum","lmin"),options.get("power_spectrum","lmax")))

	if options.has_section("moments"):
		s.write("""The set of 9 moments\n\n""")

	if options.has_section("peaks"):
		s.write("""Peak counts: {0} bins between kappa={1} and kappa={2}\n\n""".format(options.get("peaks","num_bins"),options.get("peaks","th_min"),options.get("peaks","th_max")))

	if options.has_section("minkowski_functionals"):
		s.write("""Minkowski functionals: {0} bins between kappa={1} and kappa={2}\n\n""".format(options.get("minkowski_functionals","num_bins"),options.get("minkowski_functionals","th_min"),options.get("minkowski_functionals","th_max")))

	s.seek(0)
	return s.read()

##########################################################################################################################
##################FITS loader for the maps, must set angle explicitely since it's not contained in the header#############
##########################################################################################################################

def cfht_fits_loader(filename):

	kappa_file = fits.open(filename)
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

	def __init__(self,model,options,subfield,smoothing_scale,measurer,**kwargs):

		self.model = model
		self.options = options
		self.subfield = subfield
		self.smoothing_scale = smoothing_scale
		self.measurer = measurer
		self.kwargs = kwargs

		#Build elements of save path for the features
		self.save_path = options.get("analysis","save_path")
		
		try:
			self.cosmo_id = self.model._cosmo_id_string
		except:
			pass

		self.subfield_name = "subfield{0}".format(self.subfield)
		self.smoothing_name = "sigma{0:02d}".format(int(self.smoothing_scale * 10))


	def get_all_map_names(self):
		"""
		Builds a list with all the names of the maps to be analyzed, for each subfield and smoothing scale

		"""

		if type(self.model) == CFHTemu1:
			realizations = range(1,self.options.getint("analysis","num_realizations")+1)
			self.map_names = self.model.getNames(realizations=realizations,subfield=self.subfield,smoothing=self.smoothing_scale)
			self.full_save_path = os.path.join(self.save_path,self.cosmo_id,self.subfield_name,self.smoothing_name)
		elif type(self.model) == CFHTLens:
			self.map_names = [self.model.getName(subfield=self.subfield,smoothing=self.smoothing_scale)]
			self.full_save_path = os.path.join(self.save_path,"observations",self.subfield_name,self.smoothing_name)
		else:
			raise TypeError("Your model is not supported in this analysis!")

	def measure(self,pool=None):
		"""
		Measures the features specified in the Indexer for all the maps whose names are calculated by get_all_map_names; saves the ensemble results in numpy array format

		"""

		#Build the ensemble
		ens = Ensemble.fromfilelist(self.map_names)

		#Load the data into the ensemble by calling the measurer on each map
		ens.load(callback_loader=self.measurer,pool=pool,**self.kwargs)

		#Break the ensemble into sub-ensemble, one for each feature
		single_feature_ensembles = ens.split(self.kwargs["index"])

		#For each of the sub_ensembles, save it in the appropriate directory
		for n,ensemble in enumerate(single_feature_ensembles):
			ensemble.save(os.path.join(self.full_save_path,self.kwargs["index"][n].name) + ".npy")



#######################################################
###############Main execution##########################
#######################################################

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

	#Set progressbar attributes
	widgets = ["Progress: ",progressbar.Percentage(),' ',progressbar.Bar(marker="+")]

	#Parse INI options file
	options = ConfigParser.ConfigParser()
	with open(cmd_args.options_file,"r") as configfile:
		options.readfp(configfile)

	#Read the save path from options
	save_path = options.get("analysis","save_path")

	#Get the names of all the simulated models available for the CFHT analysis, including smoothing scales and subfields
	all_simulated_models = CFHTemu1.getModels(root_path=options.get("simulations","root_path"))

	#Get also the observation model instance
	observed_model = CFHTLens(root_path=options.get("observations","root_path"))

	#Select subset
	models = all_simulated_models
	subfields = [ int(subfield) for subfield in options.get("analysis","subfields").split(",") ]
	smoothing_scales = [options.getfloat("analysis","smoothing_scale")]

	#Append the observation to the maps to process
	models.append(observed_model)

	#Build an Indexer instance, that will contain info on all the features to measure, including binning, etc... (read from options)
	feature_list = list()

	if options.has_section("power_spectrum"):
		l_edges = np.ogrid[options.getfloat("power_spectrum","lmin"):options.getfloat("power_spectrum","lmax"):(options.getint("power_spectrum","num_bins")+1)*1j]
		np.save(os.path.join(save_path,"ell.npy"),0.5*(l_edges[1:]+l_edges[:-1]))
		feature_list.append(PowerSpectrum(l_edges))

	if options.has_section("moments"):
		feature_list.append(Moments())

	if options.has_section("peaks"):
		th_peaks = np.ogrid[options.getfloat("peaks","th_min"):options.getfloat("peaks","th_max"):(options.getint("peaks","num_bins")+1)*1j]
		np.save(os.path.join(save_path,"th_peaks.npy"),0.5*(th_peaks[1:]+th_peaks[:-1]))
		feature_list.append(Peaks(th_peaks))

	if options.has_section("minkowski_functionals"):
		th_minkowski = np.ogrid[options.getfloat("minkowski_functionals","th_min"):options.getfloat("minkowski_functionals","th_max"):(options.getint("minkowski_functionals","num_bins")+1)*1j]
		np.save(os.path.join(save_path,"th_minkowski.npy"),0.5*(th_minkowski[1:]+th_minkowski[:-1]))
		feature_list.append(MinkowskiAll(th_minkowski))

	idx = Indexer.stack(feature_list)

	#Write an info file with all the analysis information
	with open(os.path.join(save_path,"INFO.txt"),"w") as infofile:
		infofile.write(write_info(options))

	#Build the progress bar
	pbar = progressbar.ProgressBar(widgets=widgets,maxval=len(models)*len(subfields)*len(smoothing_scales)).start()
	i = 0

	#Cycle through the models and perform the measurements of the selected features (create the appropriate directories to save the outputs)
	for model in models:

		if type(model) == CFHTemu1:
			dir_to_make = os.path.join(save_path,model._cosmo_id_string)
		elif type(model) == CFHTLens:
			dir_to_make = os.path.join(save_path,"observations")
		else:
			raise TypeError("Your model is not supported in this analysis!")

		base_model_dir = dir_to_make
		
		if not os.path.exists(dir_to_make):
			os.mkdir(dir_to_make)

		for subfield in subfields:

			dir_to_make = os.path.join(base_model_dir,"subfield{0}".format(subfield))
			if not os.path.exists(dir_to_make):
				os.mkdir(dir_to_make)

			for smoothing_scale in smoothing_scales:

				dir_to_make = os.path.join(base_model_dir,"subfield{0}".format(subfield),"sigma{0:02d}".format(int(smoothing_scale*10)))
				if not os.path.exists(dir_to_make):
					os.mkdir(dir_to_make)
	
				m = Measurement(model=model,options=options,subfield=subfield,smoothing_scale=smoothing_scale,measurer=convergence_measure_all,fits_loader=cfht_fits_loader,index=idx)
				m.get_all_map_names()
				m.measure()

				i+=1
				pbar.update(i)

	pbar.finish()
	logging.info("DONE!")


