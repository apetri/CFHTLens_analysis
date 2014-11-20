from __future__ import print_function,division,with_statement

import os,sys
import argparse,ConfigParser
import logging

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from contours import ContourPlot

def main():

	#Components to display
	test_components = [3,4,5,6,8,10,20,30,40,50]

	#Parameters of which we want to compute the confidence estimates
	parameter_axes = {"Omega_m":0,"w":1,"sigma8":2}
	cosmo_labels = {"Omega_m":r"$\Omega_m$","w":r"$w$","sigma8":r"$\sigma_8$"}
	
	#Parse command line options
	parser = argparse.ArgumentParser(prog=sys.argv[0])
	parser.add_argument("likelihood_npy_file_root",nargs="*")
	parser.add_argument("-f","--file",dest="options_file",action="store",type=str,help="analysis options file")

	cmd_args = parser.parse_args()

	if cmd_args.options_file is None:
		parser.print_help()
		sys.exit(0)

	#Parse options from configuration file
	options = ConfigParser.ConfigParser()
	with open(cmd_args.options_file,"r") as configfile:
		options.readfp(configfile)

	#Decide the confidence levels to display
	levels = [ 0.683 ]
	#Parse from options a list of pretty colors
	colors = options.get("contours","colors").split(",")

	#Set up the plot
	fig,ax = plt.subplots()
	proxy = list()

	#Loop over the components
	for n,n_components in enumerate(test_components):

		#Build the contour plot with the ContourPlot class handler
		contour = ContourPlot(fig=fig,ax=ax)
		#Load the likelihood
		contour.getLikelihood(cmd_args.likelihood_npy_file_root[0]+"{0}.npy".format(n_components),parameter_axes=parameter_axes,parameter_labels=cosmo_labels)
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
		

		#Compute the likelihood levels
		contour.getLikelihoodValues(levels=levels)
		print("Desired p_values:",contour.original_p_values)
		print("Calculated p_values",contour.computed_p_values)
		
		#Display the contours
		contour.plotContours(colors=[colors[n]],fill=False,display_percentages=False,display_maximum=False)


	contour.title_label="PCA test (68%)"
	contour.labels(contour_label=[r"$N_c={0}$".format(n) for n in test_components])

	#Save the result
	figure_name = options.get("contours","figure_name")
	if figure_name=="None":
		figure_name = cmd_args.likelihood_npy_file_root[0].replace("likelihoods","contours")+".png"
	
	fig.savefig(figure_name)


if __name__=="__main__":
	main()

