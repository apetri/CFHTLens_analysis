from __future__ import print_function,division,with_statement

import os,sys
import argparse,ConfigParser
import logging

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#################################################################################################
###############Find the likelihood values that correspond to the confidence contours#############
#################################################################################################

def likelihood_values(likelihood,levels=[0.684],epsilon=0.01,max_iterations=1000):

	#Check sanity of input, likelihood must be normalized
	assert likelihood.ndim == 2
	np.testing.assert_approx_equal(likelihood.sum(),1.0)

	#Initialize list of likelihood values
	values = list()
	p_values = list()
	f = stats.chi2(2)

	#Maximum value of the likelihood
	max_likelihood = likelihood.max()

	#Initial step for the search
	step = max_likelihood
	direction = 0 

	#Loop through levels to find corresponding likelihood values
	for level in levels:

		#Iteration counter
		iterations = 0

		#Start with a guess based on a chi2 distribution with 2 degrees of freedom
		value = max_likelihood*np.exp(-0.5*f.ppf(level))
		confidence_integral = likelihood[likelihood > value].sum() 

		#Continue looping until we reach the requested precision
		while np.abs(confidence_integral/level - 1.0) > epsilon:

			#Break loop if too many iterations
			iterations += 1
			if iterations > max_iterations:
				break

			if confidence_integral>level:
				
				if direction==-1:
					logging.debug("Change direction, accuracy={0}".format(np.abs(confidence_integral/level - 1.0)))
					step /= 10.0
				value += step
				direction = 1
			
			else:

				if direction==1:
					logging.debug("Change direction, accuracy={0}".format(np.abs(confidence_integral/level - 1.0)))
					step /= 10.0
				value -= step
				direction = -1

			confidence_integral = likelihood[likelihood > value].sum() 

		#Append the found likelihood value to the output
		values.append(value)
		p_values.append(confidence_integral)

	#Return
	return values,p_values

######################################################################
##############Plot the contours on top of the likelihood##############
######################################################################

def plot_contours(ax,likelihood,values,levels,display_percentages,**kwargs):

	assert "colors" in kwargs.keys() and "extent" in kwargs.keys()
	assert len(kwargs["colors"]) == len(values)

	ax1 = ax.imshow(likelihood,origin="lower",cmap=plt.cm.binary_r,extent=kwargs["extent"],aspect="auto")
	plt.colorbar(ax1,ax=ax)

	#Build contour levels
	fmt = dict()
	
	for n,value in enumerate(values):
		fmt[value] = "{0:.1f}%".format(levels[n]*100)

	cs = ax.contour(likelihood,values,colors=kwargs["colors"],origin="lower",extent=kwargs["extent"],aspect="auto")
	
	if display_percentages:
		plt.clabel(cs,fmt=fmt,inline=1,fontsize=9)

################################################################
#####################Main execution#############################
################################################################

if __name__=="__main__":

	#Parameters of which we want to compute the confidence estimates
	cosmo_parameters = ["Omega_m","w","sigma8"]
	cosmo_labels = {"Omega_m":r"$\Omega_m$","w":r"$w$","sigma8":r"$\sigma_8$"}
	
	#Parse command line options
	parser = argparse.ArgumentParser(prog=sys.argv[0])
	parser.add_argument("likelihood_npy_file",nargs="+")
	parser.add_argument("-f","--file",dest="options_file",action="store",type=str,help="analysis options file")

	cmd_args = parser.parse_args()

	if cmd_args.options_file is None:
		parser.print_help()
		sys.exit(0)

	full_likelihood = np.load(cmd_args.likelihood_npy_file[0])

	#Parse options from configuration file
	options = ConfigParser.ConfigParser()
	with open(cmd_args.options_file,"r") as configfile:
		options.readfp(configfile)

	#Decide the axis on which to marginalize
	marginalize_over = options.get("contours","marginalize_over")
	if marginalize_over == "Omega_m":
		marginalize_axis = 0
	elif marginalize_over == "w":
		marginalize_axis = 1
	elif marginalize_over == "sigma8":
		marginalize_axis = 2
	else:
		raise ValueError("Invalid parameter name")

	#Decide the confidence levels to display
	levels = [ float(level) for level in options.get("contours","levels").split(",") ]
	#Parse a list of pretty colors
	colors = options.get("contours","colors").split(",")

	#Set the extent of the plot once the parameters to display are known
	cosmo_parameters.pop(cosmo_parameters.index(marginalize_over))
	extent = (options.getfloat(cosmo_parameters[0],"min"),options.getfloat(cosmo_parameters[0],"max"),options.getfloat(cosmo_parameters[1],"min"),options.getfloat(cosmo_parameters[1],"max"))

	#Decide if showing percentages on plot
	display_percentages = options.getboolean("contours","display_percentages")

	#Marginalize over one of the parameters
	if full_likelihood.ndim == 3:
		marginalized_likelihood = full_likelihood.sum(marginalize_axis).transpose()
	else:
		marginalized_likelihood = full_likelihood.transpose()

	#Normalize
	marginalized_likelihood /= marginalized_likelihood.sum()

	#Find values and plot contours
	fig,ax = plt.subplots()
	
	values,p_values = likelihood_values(marginalized_likelihood,levels=levels)
	print("Original p_values:",levels)
	print("Computed p_values:",p_values)
	
	plot_contours(ax,marginalized_likelihood,values=values,levels=levels,display_percentages=display_percentages,extent=extent,colors=colors[:len(values)])
	
	ax.set_xlabel(cosmo_labels[cosmo_parameters[0]])
	ax.set_ylabel(cosmo_labels[cosmo_parameters[1]])

	#Save the contours figure as png
	contours_dir = os.path.join(options.get("analysis","save_path"),"contours")
	if not os.path.isdir(contours_dir):
		os.mkdir(contours_dir)

	figure_name = options.get("contours","figure_name")
	if figure_name=="None":
		figure_name = cmd_args.likelihood_npy_file[0].replace("npy","png").replace("likelihoods","contours")
	
	fig.savefig(figure_name) 

