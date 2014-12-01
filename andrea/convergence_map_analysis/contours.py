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

	def __init__(self,fig=None,ax=None):

		try:
			
			if (fig is None) or (ax is None):
				self.fig,self.ax = plt.subplots()
				self.ax.proxy = list()
			else:
				self.fig = fig
				self.ax = ax

				if not hasattr(self.ax,"proxy"):
					self.ax.proxy = list()

		except:

			print("Warning, no matplotlib functionalities!")
			pass
		
		self.min = dict()
		self.max = dict()
		self.npoints = dict()
		self.unit = dict()

	def savefig(self,figname):

		"""
		Save the plot to file

		"""

		self.fig.savefig(figname)

	def getUnitsFromOptions(self,options):
		
		"""
		Parse options file to get physical units of axes

		"""

		assert hasattr(self,"parameter_axes"),"You have to load in the likelihood first!"
		parameters = self.parameter_axes.keys()

		for parameter in parameters:
			
			self.min[parameter],self.max[parameter],self.npoints[parameter] = options.getfloat(parameter,"min"),options.getfloat(parameter,"max"),options.getint(parameter,"num_points")
			assert self.npoints[parameter] == self.likelihood.shape[self.parameter_axes[parameter]]
			self.unit[parameter] = (self.max[parameter] - self.min[parameter]) / (self.npoints[parameter] - 1)

	def setUnits(self,parameter,parameter_min,parameter_max,parameter_unit):

		"""
		Set manually the physical units for each of the likelihood axes

		"""
		assert hasattr(self,"parameter_axes"),"You have to load in the likelihood first!"
		assert parameter in self.parameter_axes.keys(),"You are trying to set units for a parameter that doesn't exist!"

		self.min[parameter] = parameter_min
		self.max[parameter] = parameter_max
		self.unit[parameter] = parameter_unit

		print("Units set for {0}; min={1:.3f} max={2:.3f} unit={3:.3f}".format(parameter,parameter_min,parameter_max,parameter_unit))

	def value(self,*coordinates):

		"""
		Compute the (un-normalized) likelihood value at the specified point in parameter space

		"""

		assert len(coordinates) == self.likelihood.ndim,"You must specify a coordinate (and only one) for each axis"

		#Compute the physical values of the pixels
		pix = np.zeros(len(coordinates))
		for parameter in self.parameter_axes.keys():

			assert parameter in self.unit.keys() and parameter in self.min.keys()
			axis = self.parameter_axes[parameter]
			pix[axis] = int((coordinates[axis] - self.min[parameter])/(self.unit[parameter]))

		#Return the found likelihood value
		try:
			return self.likelihood[tuple(pix)]
		except IndexError:
			print("Out of bounds!")
			return None


	def getLikelihood(self,likelihood_filename,parameter_axes={"Omega_m":0,"w":1,"sigma8":2},parameter_labels={"Omega_m":r"$\Omega_m$","w":r"$w$","sigma8":r"$\sigma_8$"}):
		
		"""
		Load the likelihood function from a numpy file

		"""

		self.parameter_axes = parameter_axes
		self.parameter_labels = parameter_labels

		if type(likelihood_filename)==str:
			
			self.likelihood = np.load(likelihood_filename)
			#Construct title label
			self.title_label = os.path.split(likelihood_filename)[1].lstrip("likelihood_").rstrip(".npy")
		
		elif type(likelihood_filename)==np.ndarray:
			
			self.likelihood = likelihood_filename
			#Construct title label
			self.title_label = "Default"

		assert len(self.parameter_axes.keys()) == self.likelihood.ndim,"The number of parameters should be the same as the number of dimensions of the likelihood!"

	def getMaximum(self,which="full"):

		"""
		Find the point in parameter space on which the likelihood is maximum

		"""
		max_parameters = dict()

		if which=="full":
			
			max_loc = np.where(self.likelihood==self.likelihood.max())
			for parameter in self.parameter_axes.keys():
				max_parameters[parameter] = max_loc[self.parameter_axes[parameter]][0] * self.unit[parameter] + self.min[parameter]
		
		elif which=="reduced":
			
			max_loc = np.where(self.reduced_likelihood==self.reduced_likelihood.max())
			for n,parameter in enumerate(self.remaining_parameters):
				max_parameters[parameter] = max_loc[n][0] * self.unit[parameter] + self.min[parameter]
		
		else:
			raise ValueError("which must be either 'full' or 'reduced'")

		return max_parameters

	def marginalize(self,parameter_name="w"):

		"""
		Marginalize the likelihood over one of the parameters

		"""

		assert hasattr(self,"likelihood"),"You have to load in the likelihood first!"
		assert parameter_name in self.parameter_axes.keys(),"You are trying to marginalize over a parameter that does not exist!"
		
		if self.likelihood.ndim<3:
			
			print("The likelihood is already marginal!")
			self.reduced_likelihood = self.likelihood / self.likelihood.sum()
			self.remaining_parameters = self.parameter_axes.keys()

		else:

			self.reduced_likelihood = self.likelihood.sum(self.parameter_axes[parameter_name])

			#Normalize
			self.reduced_likelihood /= self.reduced_likelihood.sum()

			#Find the remaining parameters
			self.remaining_parameters = self.parameter_axes.keys()
			self.remaining_parameters.pop(self.remaining_parameters.index(parameter_name))
			#Sort the remaining parameter names so that the corresponding axes are in increasing order
			self.remaining_parameters.sort(key=self.parameter_axes.get)
		
		self.extent = (self.min[self.remaining_parameters[0]],self.max[self.remaining_parameters[0]],self.min[self.remaining_parameters[1]],self.max[self.remaining_parameters[1]])
		self.ax.set_xlim(self.extent[0],self.extent[1])
		self.ax.set_ylim(self.extent[2],self.extent[3])

	def slice(self,parameter_name="w",parameter_value=-1.0):

		"""
		Slice the likelihood cube by fixing one of the parameters

		"""

		assert hasattr(self,"likelihood"),"You have to load in the likelihood first!"
		assert parameter_name in self.parameter_axes.keys(),"You are trying to get a slice with a parameter that does not exist!"
		
		if self.likelihood.ndim<3:
			
			print("The likelihood is already sliced!")
			self.reduced_likelihood = self.likelihood / self.likelihood.sum()
			self.remaining_parameters = self.parameter_axes.keys()

		else:
			
			#Select the slice
			slice_axis = self.parameter_axes[parameter_name]
			slice_index = int((parameter_value - self.min[parameter_name]) / self.unit[parameter_name])
			assert slice_index<self.npoints[parameter_name],"Out of bounds!"

			#Get the slice
			self.reduced_likelihood = np.split(self.likelihood,self.npoints[parameter_name],axis=slice_axis)[slice_index].squeeze()
			
			#Normalize
			self.reduced_likelihood /= self.reduced_likelihood.sum()

			#Find the remaining parameters
			self.remaining_parameters = self.parameter_axes.keys()
			self.remaining_parameters.pop(self.remaining_parameters.index(parameter_name))
			#Sort the remaining parameter names so that the corresponding axes are in increasing order
			self.remaining_parameters.sort(key=self.parameter_axes.get)
		
		self.extent = (self.min[self.remaining_parameters[0]],self.max[self.remaining_parameters[0]],self.min[self.remaining_parameters[1]],self.max[self.remaining_parameters[1]])
		self.ax.set_xlim(self.extent[0],self.extent[1])
		self.ax.set_ylim(self.extent[2],self.extent[3])


	def show(self):

		"""
		Show the 2D marginalized likelihood

		"""

		assert self.reduced_likelihood.ndim == 2,"The marginalized likelihood must be two dimensional!!"
		
		self.likelihood_image = self.ax.imshow(self.reduced_likelihood.transpose(),origin="lower",cmap=plt.cm.binary_r,extent=self.extent,aspect="auto")
		self.colorbar = plt.colorbar(self.likelihood_image,ax=self.ax)
		self.labels()
		

	def labels(self,contour_label=None):

		"""
		Put the labels on the plot

		"""

		self.ax.set_xlabel(self.parameter_labels[self.remaining_parameters[0]])
		self.ax.set_ylabel(self.parameter_labels[self.remaining_parameters[1]])
		self.ax.set_title(self.title_label)

		if contour_label is not None:
			self.ax.legend(self.ax.proxy,contour_label)

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
		return self.reduced_likelihood[px,py]


	#################################################################################################
	###############Find the likelihood values that correspond to the confidence contours#############
	#################################################################################################

	def getLikelihoodValues(self,levels,epsilon=0.01,max_iterations=1000):

		"""
		Find the likelihood values that correspond to the selected p_values
		"""

		likelihood = self.reduced_likelihood
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

	def plotContours(self,colors=["red","green","blue"],display_percentages=True,display_maximum=True,fill=False,**kwargs):

		"""
		Display the confidence likelihood contours

		"""

		if not hasattr(self,"likelihood_values"):
			self.getLikelihoodValues(levels=[0.683,0.95,0.997])

		assert len(colors) == len(self.likelihood_values)

		extent = self.extent
		likelihood = self.reduced_likelihood.transpose()
		values = self.likelihood_values

		unit_j = (extent[1] - extent[0])/(likelihood.shape[1] - 1)
		unit_i = (extent[3] - extent[2])/(likelihood.shape[0] - 1) 

		#Build contour levels
		fmt = dict()
		
		for n,value in enumerate(values):
			fmt[value] = "{0:.1f}%".format(self.computed_p_values[n]*100)

		if fill:
			self.contour = self.ax.contourf(likelihood,values,colors=colors,origin="lower",extent=extent,aspect="auto",**kwargs)
		else:
			self.contour = self.ax.contour(likelihood,values,colors=colors,origin="lower",extent=extent,aspect="auto",**kwargs)

		#Contour labels
		self.ax.proxy += [ plt.Rectangle((0,0),1,1,fc=color) for color in colors if color!="#eeeeee" ]
		
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
