from __future__ import print_function,division,with_statement

import os,sys
import argparse,ConfigParser
import logging

import numpy as np
from scipy import stats
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import rc

from lenstools.contours import ContourPlot

################################################################
#####################Main execution#############################
################################################################

def main():

	#Parameters of which we want to compute the confidence estimates
	parameter_axes = {"Omega_m":0,"w":1,"sigma8":2}
	cosmo_labels = {"Omega_m":r"$\Omega_m$","w":r"$w$","sigma8":r"$\sigma_8$"}
	
	#Parse command line options
	parser = argparse.ArgumentParser(prog=sys.argv[0])
	parser.add_argument("likelihood_npy_file",nargs="*")
	parser.add_argument("-f","--file",dest="options_file",action="store",type=str,help="analysis options file")
	parser.add_argument("-a","--all",dest="all",action="store_true",help="If specified, plots all the contours in a single figure")

	cmd_args = parser.parse_args()

	if cmd_args.options_file is None:
		parser.print_help()
		sys.exit(0)

	#Parse options from configuration file
	options = ConfigParser.ConfigParser()
	with open(cmd_args.options_file,"r") as configfile:
		options.readfp(configfile)

	#Decide the confidence levels to display
	levels = [ float(level) for level in options.get("contours","levels").split(",") ]
	#Parse from options a list of pretty colors
	colors = options.get("contours","colors").split(",")[:len(levels)]

	#Decide if showing percentages and maximum on plot
	display_percentages = options.getboolean("contours","display_percentages")
	display_maximum = options.getboolean("contours","display_maximum")

	if cmd_args.all:

		#These are all the names of the likelihood files
		likelihood_dir = os.path.join(options.get("analysis","save_path"),"likelihoods")
		likelihood_files = ["likelihood_power_spectrum--1.0.npy","likelihood_peaks--1.0.npy","likelihood_moments--1.0.npy","likelihood_minkowski_0--1.0.npy","likelihood_minkowski_1--1.0.npy","likelihood_minkowski_2--1.0.npy"] 

		#Build a figure that contains all the plots
		fig,ax = plt.subplots(3,2,figsize=(16,24))

		#Plot the contours
		for i in range(3):
			for j in range(2):

				contour = ContourPlot(fig=fig,ax=ax[i,j])
				contour.getLikelihood(os.path.join(likelihood_dir,likelihood_files[2*i + j]),parameter_axes=parameter_axes,parameter_labels=cosmo_labels)
				

				contour.getUnitsFromOptions(options)
				
				#Marginalize over one of the parameters
				if options.get("contours","marginalize_over")!="none" and options.get("contours","slice_over")!="none":
					raise ValueError("marginalize_over and slice_over cannot be both not none!")

				if options.get("contours","marginalize_over")!="none":
					contour.marginalize(options.get("contours","marginalize_over"))
					print("{0} marginalized likelihood is maximum at {1}".format(options.get("contours","marginalize_over"),contour.getMaximum(which="reduced")))

				if options.get("contours","slice_over")!="none":
					contour.slice(options.get("contours","slice_over"),options.getfloat("contours","slice_value"))
					print("{0}={1} likelihood slice is maximum at {2}".format(options.get("contours","slice_over"),options.getfloat("contours","slice_value"),contour.getMaximum(which="reduced")))
				
				contour.show()
				contour.getLikelihoodValues(levels=levels)
				contour.plotContours(colors=colors,fill=False,display_percentages=True)


		#Save the result
		contours_dir = os.path.join(options.get("analysis","save_path"),"contours")
		if not os.path.isdir(contours_dir):
			os.mkdir(contours_dir)

		figure_name = options.get("contours","figure_name")
		if figure_name=="None":
			figure_name = os.path.join(contours_dir,"likelihood_all--1.0.png")
	
		contour.savefig(figure_name)




	else:
		
		#Build the contour plot with the ContourPlot class handler
		contour = ContourPlot()
		#Load the likelihood
		contour.getLikelihood(cmd_args.likelihood_npy_file[0],parameter_axes=parameter_axes,parameter_labels=cosmo_labels)
		#Set the physical units
		contour.getUnitsFromOptions(options)
		#Find the maximum value of the likelihood
		print("Full likelihood is maximum at {0}".format(contour.getMaximum(which="full")))
		
		#Marginalize over one of the parameters
		if options.get("contours","marginalize_over")!="none" and options.get("contours","slice_over")!="none":
			raise ValueError("marginalize_over and slice_over cannot be both not none!")

		if options.get("contours","marginalize_over")!="none":
			contour.marginalize(options.get("contours","marginalize_over"))
			print("{0} marginalized likelihood is maximum at {1}".format(options.get("contours","marginalize_over"),contour.getMaximum(which="reduced")))

		if options.get("contours","slice_over")!="none":
			contour.slice(options.get("contours","slice_over"),options.getfloat("contours","slice_value"))
			print("{0}={1} likelihood slice is maximum at {2}".format(options.get("contours","slice_over"),options.getfloat("contours","slice_value"),contour.getMaximum(which="reduced")))
		

		#Show the full likelihood
		contour.show()
		#Compute the likelihood levels
		contour.getLikelihoodValues(levels=levels)
		print("Desired p_values:",contour.original_p_values)
		print("Calculated p_values",contour.computed_p_values)
		#Display the contours
		contour.plotContours(colors=colors,fill=False,display_percentages=True)

		#Save the result
		contours_dir = os.path.join(options.get("analysis","save_path"),"contours")
		if not os.path.isdir(contours_dir):
			os.mkdir(contours_dir)

		figure_name = options.get("contours","figure_name")
		if figure_name=="None":
			figure_name = cmd_args.likelihood_npy_file[0].replace("npy","png").replace("likelihoods","contours")
	
		contour.savefig(figure_name)


if __name__=="__main__":
	main()
