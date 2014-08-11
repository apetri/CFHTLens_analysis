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

def plot_contours(ax,likelihood,values,**kwargs):

	assert "colors" in kwargs.keys() and "extent" in kwargs.keys()
	assert len(kwargs["colors"]) == len(values)

	ax1 = ax.imshow(likelihood,origin="lower",cmap=plt.cm.binary_r,extent=kwargs["extent"],aspect="auto")
	plt.colorbar(ax1,ax=ax)
	
	ax.contour(likelihood,values,colors=kwargs["colors"],origin="lower",extent=kwargs["extent"],aspect="auto")

################################################################
#####################Main execution#############################
################################################################

if __name__=="__main__":

	if len(sys.argv < 2):
		print("Usage {0} <3D-numpy-likelihood>".format(sys.argv[0]))
		sys.exit(0)

	full_likelihood = np.load(sys.argv[1])

	#Marginalize over w
	marginalized_likelihood = full_likelihood.sum(1)

	#Normalize
	marginalized_likelihood /= marginalized_likelihood.sum()

	#Find values and plot contours
	fig,ax = plt.subplots()
	
	values,p_values = likelihood_values(marginalized_likelihood,levels=[0.684,0.90,0.99])
	print("p_values:",p_values)
	
	plot_contours(ax,marginalized_likelihood,values=values,colors=["red","green","blue"])
	fig.savefig(sys.argv[1].replace(".npy",".png")) 

