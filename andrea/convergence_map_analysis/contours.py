from __future__ import print_function,division,with_statement

import os,sys
import argparse,ConfigParser
import logging

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#############################################################
##################ContourPlot class##########################
#############################################################

class ContourPlot(object):

	"""
	A class handler for contour plots

	"""

	def __init__(self):

		self.fig,self.ax = plt.subplots()

	def getUnitsFromOptions(self,options):
		
		"""
		Parse options file to get physical units of axes

		"""

		assert hasattr(self,"parameter_axes"),"You have to load in the likelihood first!"
		parameters = self.parameter_axes.keys()

		self.min = dict()
		self.max = dict()
		self.npoints = dict()
		self.unit = dict()

		for parameter in parameters:
			
			self.min[parameter],self.max[parameter],self.npoints[parameter] = options.getfloat(parameter,"min"),options.getfloat(parameter,"max"),options.getint(parameter,"num_points")
			self.unit[parameter] = (self.max[parameter] - self.min[parameter]) / (self.npoints[parameter] - 1)

	def getLikelihood(self,likelihood_filename,parameter_axes={"Omega_m":0,"w":1,"sigma8":2},parameter_labels={"Omega_m":r"$\Omega_m$","w":r"$w$","sigma8":r"$\sigma_8$"}):
		
		"""
		Load the likelihood function from a numpy file

		"""

		self.parameter_axes = parameter_axes
		self.parameter_labels = parameter_labels
		self.likelihood = np.load(likelihood_filename)

		assert len(self.parameter_axes.keys()) == self.likelihood.ndim,"The number of parameters should be the same as the number of dimensions of the likelihood!"

	def marginalize(self,parameter_name="w"):

		"""
		Marginalize the likelihood over one of the parameters

		"""

		assert hasattr(self,"likelihood"),"You have to load in the likelihood first!"
		
		if self.likelihood.ndim<3:
			
			print("The likelihood is already marginal!")
			self.marginalized_likelihood = self.likelihood / self.likelihood.sum()
			self.remaining_parameters = self.parameter_axes.keys()

		else:

			self.marginalized_likelihood = self.likelihood.sum(self.parameter_axes[parameter_name])
			self.marginalized_likelihood /= self.marginalized_likelihood.sum()

			#Find the remaining parameters
			self.remaining_parameters = self.parameter_axes.keys()
			self.remaining_parameters.pop(self.remaining_parameters.index(parameter_name))
		
		self.extent = (self.min[self.remaining_parameters[0]],self.max[self.remaining_parameters[0]],self.min[self.remaining_parameters[1]],self.max[self.remaining_parameters[1]])
		self.ax.set_xlim(self.extent[0],self.extent[1])
		self.ax.set_ylim(self.extent[2],self.extent[3])

	def show(self):

		"""
		Show the 2D marginalized likelihood

		"""

		assert self.marginalized_likelihood.ndim == 2,"The marginalized likelihood must be two dimensional!!"
		
		self.likelihood_image = self.ax.imshow(self.marginalized_likelihood.transpose(),origin="lower",cmap=plt.cm.binary_r,extent=self.extent,aspect="auto")
		self.colorbar = plt.colorbar(self.likelihood_image,ax=self.ax)

		self.ax.set_xlabel(self.parameter_labels[self.remaining_parameters[0]])
		self.ax.set_ylabel(self.parameter_labels[self.remaining_parameters[1]])

	def point(self,coordinate_x,coordinate_y,color="green",marker="o"):

		"""
		Draws a point in parameter space at the specified physical coordinates

		"""

		#First translate the physical coordinates into pixels, to obtain the likelihood value
		px = int((coordinate_x - self.min[self.remaining_parameters[0]]) / self.unit[self.remaining_parameters[0]])
		py = int((coordinate_y - self.min[self.remaining_parameters[1]]) / self.unit[self.remaining_parameters[1]])

		#Draw the point
		self.ax.plot(coordinate_x,coordinate_y,color=color,marker=marker)

		#Return the likelihood value at the specified point
		return self.marginalized_likelihood[px,py]


	#################################################################################################
	###############Find the likelihood values that correspond to the confidence contours#############
	#################################################################################################

	def getLikelihoodValues(self,levels,epsilon=0.01,max_iterations=1000):

		"""
		Find the likelihood values that correspond to the selected p_values
		"""

		likelihood = self.marginalized_likelihood
		self.original_p_values = levels

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
		self.computed_p_values = p_values
		self.likelihood_values = values
		
		return values

	######################################################################
	##############Plot the contours on top of the likelihood##############
	######################################################################

	def plotContours(self,colors=["red","green","blue"],display_percentages=True,display_maximum=True):

		"""
		Display the confidence likelihood contours

		"""

		if not hasattr(self,"likelihood_values"):
			self.getLikelihoodValues(levels=[0.683,0.95,0.997])

		assert len(colors) == len(self.likelihood_values)

		extent = self.extent
		likelihood = self.marginalized_likelihood.transpose()
		values = self.likelihood_values

		unit_j = (extent[1] - extent[0])/(likelihood.shape[1] - 1)
		unit_i = (extent[3] - extent[2])/(likelihood.shape[0] - 1) 

		#Build contour levels
		fmt = dict()
		
		for n,value in enumerate(values):
			fmt[value] = "{0:.1f}%".format(self.computed_p_values[n]*100)

		self.contour = self.ax.contour(likelihood,values,colors=colors,origin="lower",extent=extent,aspect="auto")
		
		if display_percentages:
			plt.clabel(self.contour,fmt=fmt,inline=1,fontsize=9)

		if display_maximum:
			
			#Find the maximum
			likelihood_max = likelihood.max()
			imax,jmax = np.where(likelihood==likelihood_max)

			#Plot scaling to physical values
			self.ax.plot(extent[0] + np.arange(likelihood.shape[1])*unit_j,np.ones(likelihood.shape[1])*imax[0]*unit_i + extent[2],linestyle="--",color="green")
			self.ax.plot(extent[0] + np.ones(likelihood.shape[0])*jmax[0]*unit_j,extent[2] + np.arange(likelihood.shape[0])*unit_i,linestyle="--",color="green")

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

	#Decide if showing percentages and maximum on plot
	display_percentages = options.getboolean("contours","display_percentages")
	display_maximum = options.getboolean("contours","display_maximum")

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
	
	plot_contours(ax,marginalized_likelihood,values=values,levels=levels,display_percentages=display_percentages,display_maximum=display_maximum,extent=extent,colors=colors[:len(values)])
	
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

