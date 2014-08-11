from __future__ import print_function,division,with_statement

import os,sys
import argparse,ConfigParser

######################################################################
##################LensTools functionality#############################
######################################################################

from lenstools import Ensemble
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
import matplotlib.pyplot as plt

######################################################################
###################Main execution#####################################
######################################################################

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

	#Read the save path from options
	save_path = options.get("analysis","save_path")
	#Load feature indices
	l = np.load(os.path.join(save_path,"ell.npy"))
	v_pk = np.load(os.path.join(save_path,"th_peaks.npy"))
	v_mf = np.load(os.path.join(save_path,"th_minkowski.npy"))
	num_mf_bins = len(v_mf)

	#Get the names of all the simulated models available for the CFHT analysis, including smoothing scales and subfields
	all_simulated_models = CFHTemu1.getModels(root_path=options.get("simulations","root_path"))

	#Get also the observation model instance
	observed_model = CFHTLens(root_path=options.get("observations","root_path"))

	#Select subset of training models
	training_models = all_simulated_models

	#Parse from options which subfields and smoothing scale to consider
	subfields = [ int(subfield) for subfield in options.get("analysis","subfields").split(",") ]
	smoothing_scale = options.getfloat("analysis","smoothing_scale")

	#Make a dedicated directory for the plots
	if not os.path.isdir(os.path.join(save_path,"plots")):
		os.mkdir(os.path.join(save_path,"plots"))

	#Cycle through the models and plot
	for subfield in subfields:

		fig_power,ax_power = plt.subplots()
		fig_peaks,ax_peaks = plt.subplots()
		fig_minkowski,ax_minkowski = plt.subplots(1,3,figsize=(24,8))

		#Plot for the sumulations 
		for model in training_models + [observed_model]:

			m = Measurement(model=model,options=options,subfield=subfield,smoothing_scale=smoothing_scale,measurer=None)
			m.get_all_map_names()

			#Load the features and plot

			#Power spectrum
			ensemble_power = Ensemble.fromfilelist([os.path.join(m.full_save_path,"power_spectrum.npy")])
			ensemble_power.load(from_old=True)
			P = ensemble_power.mean()

			if type(model) == CFHTLens:
				ax_power.plot(l,l*(l+1)*P/(2*np.pi),linestyle="--",color="black")		
			elif type(model) == CFHTemu1:
				ax_power.plot(l,l*(l+1)*P/(2*np.pi))

			#Peaks
			ensemble_peaks = Ensemble.fromfilelist([os.path.join(m.full_save_path,"peaks.npy")])
			ensemble_peaks.load(from_old=True)
			pk = ensemble_peaks.mean()
			
			if type(model) == CFHTLens:
				ax_peaks.plot(v_pk,pk,linestyle="--",color="black")
			elif type(model) ==  CFHTemu1:
				ax_peaks.plot(v_pk,pk)

			#Minkowski functionals
			ensemble_minkowski = Ensemble.fromfilelist([os.path.join(m.full_save_path,"minkowski_all.npy")])
			ensemble_minkowski.load(from_old=True)
			
			mf_all = ensemble_minkowski.mean()
			mf0 = mf_all[:num_mf_bins]
			mf1 = mf_all[num_mf_bins:2*num_mf_bins]
			mf2 = mf_all[2*num_mf_bins:]
			
			if type(model) == CFHTLens:
				ax_minkowski[0].plot(v_mf,mf0,linestyle="--",color="black")
				ax_minkowski[1].plot(v_mf,mf1,linestyle="--",color="black")
				ax_minkowski[2].plot(v_mf,mf2,linestyle="--",color="black")	
			elif type(model) == CFHTemu1:
				ax_minkowski[0].plot(v_mf,mf0)
				ax_minkowski[1].plot(v_mf,mf1)
				ax_minkowski[2].plot(v_mf,mf2)
		

		fig_minkowski.tight_layout()

		#Save the figures
		fig_power.savefig(os.path.join(save_path,"plots","power_subfield{0}_sigma{1:02d}.png".format(subfield,int(smoothing_scale*10))))
		fig_peaks.savefig(os.path.join(save_path,"plots","peaks_subfield{0}_sigma{1:02d}.png".format(subfield,int(smoothing_scale*10))))
		fig_minkowski.savefig(os.path.join(save_path,"plots","minkowski_subfield{0}_sigma{1:02d}.png".format(subfield,int(smoothing_scale*10))))


	##########################################################################################
	#######If there is more than one subfield plot also average between subfields#############
	##########################################################################################

	if len(subfields)>1:
		
		fig_power,ax_power = plt.subplots()
		fig_peaks,ax_peaks = plt.subplots()
		fig_minkowski,ax_minkowski = plt.subplots(1,3,figsize=(24,8))

		for model in training_models + [observed_model]:

			ensemble_power_all = Ensemble()
			ensemble_peaks_all = Ensemble()
			ensemble_minkowski_all = Ensemble()

			#Accumulate subfields
			for subfield in subfields:

				m = Measurement(model=model,options=options,subfield=subfield,smoothing_scale=smoothing_scale,measurer=None)
				m.get_all_map_names()

				#Load the features and plot

				#Power spectrum
				ensemble_power = Ensemble.fromfilelist([os.path.join(m.full_save_path,"power_spectrum.npy")])
				ensemble_power.load(from_old=True)
				ensemble_power_all += ensemble_power

				#Peaks
				ensemble_peaks = Ensemble.fromfilelist([os.path.join(m.full_save_path,"peaks.npy")])
				ensemble_peaks.load(from_old=True)
				ensemble_peaks_all += ensemble_peaks

				#Minkowski functionals
				ensemble_minkowski = Ensemble.fromfilelist([os.path.join(m.full_save_path,"minkowski_all.npy")])
				ensemble_minkowski.load(from_old=True)
				ensemble_minkowski_all += ensemble_minkowski

			#Average and plot
			if type(model)==CFHTLens:
				
				P = ensemble_power_all.mean()
				ax_power.plot(l,l*(l+1)*P/(2*np.pi),linestyle="--",color="black")

				pk = ensemble_peaks_all.mean()
				ax_peaks.plot(v_pk,pk,linestyle="--",color="black")

				mf_all = ensemble_minkowski_all.mean()
				mf0 = mf_all[:num_mf_bins]
				mf1 = mf_all[num_mf_bins:2*num_mf_bins]
				mf2 = mf_all[2*num_mf_bins:]
			
				ax_minkowski[0].plot(v_mf,mf0,linestyle="--",color="black")
				ax_minkowski[1].plot(v_mf,mf1,linestyle="--",color="black")
				ax_minkowski[2].plot(v_mf,mf2,linestyle="--",color="black")

			elif type(model)==CFHTemu1:

				P = ensemble_power_all.mean()
				ax_power.plot(l,l*(l+1)*P/(2*np.pi))

				pk = ensemble_peaks_all.mean()
				ax_peaks.plot(v_pk,pk)

				mf_all = ensemble_minkowski_all.mean()
				mf0 = mf_all[:num_mf_bins]
				mf1 = mf_all[num_mf_bins:2*num_mf_bins]
				mf2 = mf_all[2*num_mf_bins:]
			
				ax_minkowski[0].plot(v_mf,mf0)
				ax_minkowski[1].plot(v_mf,mf1)
				ax_minkowski[2].plot(v_mf,mf2)

		#Set the axes labels
		ax_power.set_xlabel(r"$l$")
		ax_power.set_ylabel(r"$l(l+1)P_l/2\pi$")

		ax_peaks.set_xlabel(r"$\kappa$")
		ax_peaks.set_ylabel(r"$dN/d\kappa$")

		ax_minkowski[0].set_xlabel(r"$\kappa$")
		ax_minkowski[1].set_xlabel(r"$\kappa$")
		ax_minkowski[2].set_xlabel(r"$\kappa$")

		ax_minkowski[0].set_ylabel(r"$V_0$")
		ax_minkowski[1].set_ylabel(r"$V_1$")
		ax_minkowski[2].set_ylabel(r"$V_2$")

		fig_minkowski.tight_layout()

		#Save the figures
		fig_power.savefig(os.path.join(save_path,"plots","power_all_subfields_sigma{1:02d}.png".format(int(smoothing_scale*10))))
		fig_peaks.savefig(os.path.join(save_path,"plots","peaks_all_subfields_sigma{1:02d}.png".format(int(smoothing_scale*10))))
		fig_minkowski.savefig(os.path.join(save_path,"plots","minkowski_all_subfields_sigma{1:02d}.png".format(int(smoothing_scale*10))))

