import sys,os,argparse,ConfigParser

from lenstools.constraints import LikelihoodAnalysis
from lenstools.simulations import Design

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from contours import ContourPlot

axes_facecolor = rc.func_globals["rcParams"]["axes.facecolor"]

root_dir = "/Users/andreapetri/Documents/Columbia/CFHTLens_analysis/andrea/convergence_map_analysis/cfht_masked_BAD_clipped"
design_points = "/Users/andreapetri/Documents/Cosmology_software/LensTools/lenstools/data/CFHTemu1_array.npy"
brew_colors = ["red","green","blue","black","orange","magenta","cyan"]
brew_colors_11 = ["#a50026","#d73027","#f46d43","#fdae61","#fee08b","#ffffbf","#d9ef8b","#a6d96a","#66bd63","#1a9850","#006837"]
brew_colors_diverging = ["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a","#ffff99","#b15928"]

#Descriptor list
descriptors=dict()
descriptors["power_spectrum"]=r"$\mathrm{PS}$"
descriptors["minkowski_0"]=r"$V_0$"
descriptors["minkowski_1"]=r"$V_1$"
descriptors["minkowski_2"]=r"$V_2$"
descriptors["moments"]=r"$\mathrm{Moments}$"

#Number of principal components
num_components = dict()
num_components["power_spectrum"] = 3
num_components["minkowski_0"] = 3
num_components["minkowski_1"] = 3
num_components["minkowski_2"] = 3
num_components["moments"] = 3  

keys = descriptors.keys()
keys.sort()

def design(cmd_args):

	#Load the design
	design = Design.load(design_points,[r"$\Omega_m$",r"$w$",r"$\sigma_8$"])

	#Create the figure
	fig,ax = plt.subplots(1,2,figsize=(16,8))

	#Visualize the design
	design.visualize(fig,ax[0],parameters=[r"$\Omega_m$",r"$w$"])
	design.visualize(fig,ax[1],parameters=[r"$\Omega_m$",r"$\sigma_8$"])

	#Show also the fiducial point
	ax[0].scatter(0.26,-1.0,color="red")
	ax[1].scatter(0.26,0.8,color="red")

	#Save the figure 
	fig.savefig("design.{0}".format(cmd_args.type))


##############################################################################################################################################

def pca(cmd_args):

	#Smoothing scales in arcmin
	smoothing_scale=1.0

	#Create figure
	fig,ax = plt.subplots(1,2,figsize=(16,8))

	#Cycle over descriptors to plot PCA eigenvalues
	for n,descr in enumerate(keys):

		#Unpickle the emulator
		an = LikelihoodAnalysis.load(os.path.join(root_dir,"emulators","emulator_{0}--{1:.1f}.p".format(descr,smoothing_scale)))

		#Compute PCA
		pca = an.principalComponents()

		#Plot the eigenvalues on the left and the cumulative sum on the right
		ax[0].plot(pca.eigenvalues,label=descriptors[descr],color=brew_colors[n])
		ax[1].plot(pca.eigenvalues.cumsum()/pca.eigenvalues.sum(),label=descriptors[descr],color=brew_colors[n])


	#Draw a line at 3 components
	ax[0].plot(3*np.ones(100),np.linspace(1.0e-10,1.0e2,100),color="black",linestyle="--")
	ax[1].plot(3*np.ones(100),np.linspace(0.9,1.01,100),color="black",linestyle="--")
	ax[1].set_ylim(0.9,1.01)

	#Legend
	ax[0].legend()

	#Scale
	ax[0].set_yscale("log")

	#Labels
	ax[0].set_xlabel(r"$i$",fontsize=18)
	ax[1].set_xlabel(r"$n$",fontsize=18)
	ax[0].set_ylabel(r"$\Sigma^2_i$",fontsize=18)
	ax[1].set_ylabel(r"$\Sigma_{i=0}^n\Sigma^2_i/\Sigma^2_{tot}$",fontsize=18)

	#Save figure
	fig.savefig("pca_components.{0}".format(cmd_args.type))


##############################################################################################################################################

def contours(cmd_args):

	#Smoothing scales in arcmin
	smoothing_scale=1.0
	levels = [0.684,0.001]

	#Parameters of which we want to compute the confidence estimates
	parameter_axes = {"Omega_m":0,"w":1,"sigma8":2}
	cosmo_labels = {"Omega_m":r"$\Omega_m$","w":r"$w$","sigma8":r"$\sigma_8$"}

	#Parse options from configuration file
	options = ConfigParser.ConfigParser()
	with open(cmd_args.options_file,"r") as configfile:
		options.readfp(configfile)

	#Create figure
	fig,ax = plt.subplots(1,2,figsize=(16,8))

	#Cycle over descriptors
	for n,descr in enumerate(keys):

		#Instantiate contour plot
		contour_marg = ContourPlot(fig=fig,ax=ax[0])
		contour_slice = ContourPlot(fig=fig,ax=ax[1])

		#Load the likelihood
		likelihood_file = os.path.join(root_dir,"likelihoods","likelihood_{0}--{1:.1f}_ncomp{2}.npy".format(descr,smoothing_scale,num_components[descr]))
		contour_marg.getLikelihood(likelihood_file,parameter_axes=parameter_axes,parameter_labels=cosmo_labels)
		contour_slice.getLikelihood(likelihood_file,parameter_axes=parameter_axes,parameter_labels=cosmo_labels)
		
		#Set the physical units
		contour_marg.getUnitsFromOptions(options)
		contour_slice.getUnitsFromOptions(options)

		#Marginalize
		contour_marg.marginalize("w")
		
		#Slice on best fit for w
		maximum = contour_slice.getMaximum()
		print("Likelihood with {0} is maximum at {1}".format(descr,maximum))
		contour_slice.slice("w",maximum["w"])

		#Get levels
		contour_marg.getLikelihoodValues(levels=levels)
		contour_slice.getLikelihoodValues(levels=levels)

		#Plot contours
		contour_marg.plotContours(colors=[brew_colors[n],axes_facecolor],fill=False,display_maximum=False,display_percentages=False,alpha=1.0)
		contour_slice.plotContours(colors=[brew_colors[n],axes_facecolor],fill=False,display_maximum=False,display_percentages=False,alpha=1.0)

	
	#Legend
	contour_marg.title_label=""
	contour_slice.title_label=""
	contour_marg.labels([descriptors[key]+r"$({0})$".format(num_components[key]) for key in keys])
	contour_slice.labels(None)

	#Save
	fig.savefig("contours_data.{0}".format(cmd_args.type))


##################################################################################################################################################

def robustness(cmd_args):

	#Smoothing scales in arcmin
	smoothing_scale=1.0

	#Likelihood levels
	levels = [0.684]

	#Descriptors
	descriptors_robustness = ["minkowski_0--{0:.1f}","pdf_minkowski_0--{0:.1f}","minkowski_1--{0:.1f}","minkowski_2--{0:.1f}","power_spectrum--{0:.1f}","moments--{0:.1f}"]
	descriptor_titles = dict()
	descriptor_titles["minkowski_0--{0:.1f}"] = r"$V_0$"
	descriptor_titles["pdf_minkowski_0--{0:.1f}"] = r"$\partial V_0(\mathrm{PDF})$"
	descriptor_titles["minkowski_1--{0:.1f}"] = r"$V_1$"
	descriptor_titles["minkowski_2--{0:.1f}"] = r"$V_2$"
	descriptor_titles["power_spectrum--{0:.1f}"] = r"$\mathrm{PS}$"
	descriptor_titles["moments--{0:.1f}"] = r"$\mathrm{Moments}$"

	#Number of principal components to display
	principal_components = dict()
	for descr in descriptors_robustness:
		principal_components[descr] = [3,5,10,20,30,40]

	principal_components["moments--{0:.1f}"] = [3,5,9]

	#Parameters of which we want to compute the confidence estimates
	parameter_axes = {"Omega_m":0,"w":1,"sigma8":2}
	cosmo_labels = {"Omega_m":r"$\Omega_m$","w":r"$w$","sigma8":r"$\sigma_8$"}

	#Parse options from configuration file
	options = ConfigParser.ConfigParser()
	with open(cmd_args.options_file,"r") as configfile:
		options.readfp(configfile)


	#Create figure
	fig,ax = plt.subplots(3,2,figsize=(16,24))
	ax_flat = ax.reshape(6)

	#Cycle over descriptors
	for d,descr in enumerate(descriptors_robustness):
		for n,n_components in enumerate(principal_components[descr]):

			#Instantiate contour plot
			contour = ContourPlot(fig=fig,ax=ax_flat[d])

			#Load the likelihood
			likelihood_file = os.path.join(root_dir,"likelihoods","likelihoodmock_"+descr.format(smoothing_scale)+"_ncomp{0}.npy".format(n_components))
			contour.getLikelihood(likelihood_file,parameter_axes=parameter_axes,parameter_labels=cosmo_labels)

			#Set physical units
			contour.getUnitsFromOptions(options)

			#Marginalize
			contour.marginalize("w")

			#Get levels
			contour.getLikelihoodValues(levels=levels)

			#Plot the contour
			contour.plotContours(colors=[brew_colors_diverging[n]],fill=False,display_percentages=False,display_maximum=False)

		#Labels
		contour.title_label=descriptor_titles[descr]
		contour.labels(contour_label=[r"$n={0}$".format(n) for n in principal_components[descr]])


	#Save the figure
	fig.savefig("robustness_pca.{0}".format(cmd_args.type))

##################################################################################################################################################

def comparison(cmd_args):

	#Smoothing scales in arcmin
	smoothing_scale=1.0
	n_components=30

	#Likelihood levels
	levels = [0.684]

	#Descriptors
	descriptors_comparison = ["minkowski_0--{0:.1f}","minkowski_1--{0:.1f}","minkowski_2--{0:.1f}","minkowski_012--{0:.1f}_power_spectrum--{0:.1f}","minkowski_0--{0:.1f}_minkowski_1--{0:.1f}_minkowski_2--{0:.1f}","power_spectrum--{0:.1f}"]
	descriptor_titles = dict()
	descriptor_titles["minkowski_0--{0:.1f}"] = r"$V_0$"
	descriptor_titles["minkowski_1--{0:.1f}"] = r"$V_1$"
	descriptor_titles["minkowski_2--{0:.1f}"] = r"$V_2$"
	descriptor_titles["minkowski_0--{0:.1f}_minkowski_1--{0:.1f}_minkowski_2--{0:.1f}"] = r"$V_0 \times V_1 \times V_2$"
	descriptor_titles["power_spectrum--{0:.1f}"] = r"$\mathrm{PS}$"
	descriptor_titles["moments--{0:.1f}"] = r"$\mathrm{Moments}$"
	descriptor_titles["minkowski_012--{0:.1f}_power_spectrum--{0:.1f}"] = r"$V_0 \times V_1 \times V_2 \times$ $\mathrm{PS}$"

	#Parameters of which we want to compute the confidence estimates
	parameter_axes = {"Omega_m":0,"w":1,"sigma8":2}
	cosmo_labels = {"Omega_m":r"$\Omega_m$","w":r"$w$","sigma8":r"$\sigma_8$"}

	#Parse options from configuration file
	options = ConfigParser.ConfigParser()
	with open(cmd_args.options_file,"r") as configfile:
		options.readfp(configfile)

	#Create the figure
	fig,ax = plt.subplots()

	#Cycle over the desctiptors
	for n,descr in enumerate(descriptors_comparison):

		#Instantiate contour plot
		contour = ContourPlot(fig=fig,ax=ax)

		#Load the likelihood
		likelihood_file = os.path.join(root_dir,"likelihoods","likelihoodmock_"+descr.format(smoothing_scale)+"_ncomp{0}.npy".format(n_components))
		contour.getLikelihood(likelihood_file,parameter_axes=parameter_axes,parameter_labels=cosmo_labels)

		#Set physical units
		contour.getUnitsFromOptions(options)

		#Marginalize
		contour.marginalize("w")

		#Get levels
		contour.getLikelihoodValues(levels=levels)

		#Plot the contour
		contour.plotContours(colors=[brew_colors[n]],fill=False,display_percentages=False,display_maximum=False)

	#Labels
	contour.title_label=""
	contour.labels(contour_label=[descriptor_titles[descr] for descr in descriptors_comparison])

	#Save the figure
	fig.savefig("comparison_pca{0}.{1}".format(n_components,cmd_args.type))



##################################################################################################################################################

def w_likelihood(cmd_args):

	#Parameters of which we want to compute the confidence estimates
	parameter_axes = {"Omega_m":0,"w":1,"sigma8":2}
	cosmo_labels = {"Omega_m":r"$\Omega_m$","w":r"$w$","sigma8":r"$\sigma_8$"}

	#Parse options from configuration file
	options = ConfigParser.ConfigParser()
	with open(cmd_args.options_file,"r") as configfile:
		options.readfp(configfile)

	#Smoothing scales in arcmin
	smoothing_scale=1.0

	#Number of components
	n_components=3

	#Create figure
	fig,ax = plt.subplots()

	#Cycle over descriptors
	for n,descr in enumerate(keys):

		contour = ContourPlot()
		likelihood_file = os.path.join(root_dir,"likelihoods","likelihood_{0}--{1:.1f}_ncomp{2}.npy".format(descr,smoothing_scale,n_components))

		contour.getLikelihood(likelihood_file,parameter_axes=parameter_axes,parameter_labels=cosmo_labels)

		#Set physical units
		contour.getUnitsFromOptions(options)

		#Compute marginal likelihood over w
		w,l = contour.marginal("w")
		ax.plot(w,l,label=descriptors[descr],color=brew_colors[n])

	#Legend
	ax.set_xlabel(r"$w$",fontsize=18)
	ax.set_ylabel(r"$\mathcal{L}(w)$",fontsize=18)
	ax.legend()

	#Save
	fig.savefig("w_likelihood.{0}".format(cmd_args.type))

############################################################################################################################
############################################################################################################################

figure_method = dict()
figure_method["2"] = design
figure_method["3"] = pca
figure_method["4"] = robustness
figure_method["5"] = contours


if __name__=="__main__":

	#Parse command line options
	parser = argparse.ArgumentParser(prog=sys.argv[0])
	parser.add_argument("figure_numbers",nargs="*")
	parser.add_argument("-f","--file",dest="options_file",action="store",type=str,help="analysis options file")
	parser.add_argument("-t","--type",dest="type",action="store",type=str,default="png",help="image format")

	cmd_args = parser.parse_args()

	if len(cmd_args.figure_numbers)==0 or cmd_args.options_file is None:
		parser.print_help()
		sys.exit(0)

	#Generate all figures specified in input
	for fig_n in cmd_args.figure_numbers:
		figure_method[fig_n](cmd_args)

