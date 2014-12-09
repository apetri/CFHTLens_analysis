#!/usr/bin/env python

import sys,os,argparse,ConfigParser

from lenstools.constraints import LikelihoodAnalysis
from lenstools.simulations import Design

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from contours import ContourPlot

axes_facecolor = rc.func_globals["rcParams"]["axes.facecolor"]

#Locations
root_dir = "/Users/andreapetri/Documents/Columbia/CFHTLens_analysis/andrea/convergence_map_analysis/cfht_masked_BAD_clipped"
design_points = "/Users/andreapetri/Documents/Cosmology_software/LensTools/lenstools/data/CFHTemu1_array.npy"

#Colors
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
num_components["minkowski_0"] = 10
num_components["minkowski_1"] = 10
num_components["minkowski_2"] = 10
num_components["moments"] = 9 

#Smoothing scales
smoothing_scales = dict()
smoothing_scales["power_spectrum"] = 1.0
smoothing_scales["minkowski_0"] = 1.0
smoothing_scales["minkowski_1"] = 1.0
smoothing_scales["minkowski_2"] = 1.0 
smoothing_scales["moments"] = 1.0

keys = descriptors.keys()
keys.sort()

#################################################################################################
###################Consider these for the combinations###########################################
#################################################################################################

single = ["power_spectrum","minkowski_0","minkowski_1","minkowski_2","moments"]
multiple = [("minkowski_0","minkowski_1","minkowski_2"),("power_spectrum","minkowski_0","minkowski_1","minkowski_2"),("power_spectrum","minkowski_0","minkowski_1","minkowski_2","moments")]
all_descriptors = single + multiple

################################################################################################################################################
################################################################################################################################################

def _cross_name(*args):

	names = list()
	for arg in args:
		names.append("{0}--{1:.1f}_ncomp{2}".format(arg,smoothing_scales[arg],num_components[arg]))

	return "-".join(names) 

def _cross_label(*args):

	labels = list()
	for arg in args:
		labels.append(descriptors[arg]+r"$({0})$".format(num_components[arg]))

	return r" $\times$ ".join(labels)


################################################################################################################################################

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
	for n,descr in enumerate(single):

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

##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################

def robustness(cmd_args,parameter_axes={"Omega_m":0,"w":1,"sigma8":2},cosmo_labels={"Omega_m":r"$\Omega_m$","w":r"$w$","sigma8":r"$\sigma_8$"},select="w",marginalize_over="me"):

	assert marginalize_over in ["me","others"]

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
		principal_components[descr] = [3,4,5,10,20,30,40,50]

	principal_components["moments--{0:.1f}"] = [3,4,5,6,8,9]
	principal_components["pdf_minkowski_0--{0:.1f}"] = [3,4,5,10,20,30,40,49]

	#Paramteter hash
	par = parameter_axes.keys()
	par.sort(key=parameter_axes.__getitem__)
	par_hash = "-".join(par)

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
			if marginalize_over=="me":
				contour = ContourPlot(fig=fig,ax=ax_flat[d])
			else:
				contour = ContourPlot()
				contour.close()

			#Load the likelihood
			likelihood_file = os.path.join(root_dir,"likelihoods_{0}".format(par_hash),"likelihoodmock_"+descr.format(smoothing_scale)+"_ncomp{0}.npy".format(n_components))
			contour.getLikelihood(likelihood_file,parameter_axes=parameter_axes,parameter_labels=cosmo_labels)

			#Set physical units
			contour.getUnitsFromOptions(options)

			if marginalize_over=="me":
			
				#Marginalize
				contour.marginalize(select)

				#Get levels
				contour.getLikelihoodValues(levels=levels)

				#Plot the contour
				contour.plotContours(colors=[brew_colors_diverging[n]],fill=False,display_percentages=False,display_maximum=False)

			else:

				#Plot the likelihood
				p,l,pmax = contour.marginal(select)
				ax_flat[d].plot(p,l,color=brew_colors_diverging[n],label=r"$n={0}$".format(n_components))

		#Labels
		if marginalize_over=="me":
			contour.title_label=descriptor_titles[descr]
			contour.labels(contour_label=[r"$n={0}$".format(n) for n in principal_components[descr]])
		else:
			ax_flat[d].set_xlabel(cosmo_labels[select])
			ax_flat[d].set_ylabel(r"$\mathcal{L}$"+"$($"+cosmo_labels[select]+"$)$")
			ax_flat[d].set_title(descriptor_titles[descr],fontsize=22)
			ax_flat[d].legend()


	#Save the figure
	if marginalize_over=="me":
		par.pop(par.index(select))
		par_hash = "-".join(par).replace(".","")
		fig.savefig("robustness_pca_{0}.{1}".format(par_hash,cmd_args.type))
	else:
		fig.savefig("robustness_pca_{0}.{1}".format(select.replace(".",""),cmd_args.type))


##################################################################################################################################################

def robustness_1d(cmd_args):

	robustness(cmd_args,marginalize_over="others")

##################################################################################################################################################

def robustness_1d_reparametrize(cmd_args):

	robustness(cmd_args,parameter_axes={"Omega_m":0,"w":1,"Sigma8Om0.55":2},cosmo_labels={"Omega_m":r"$\Omega_m$","w":r"$w$","Sigma8Om0.55":r"$\sigma_8(\Omega_m/0.27)^{0.55}$"},select="Sigma8Om0.55",marginalize_over="others")

##################################################################################################################################################

def robustness_reparametrize(cmd_args):

	robustness(cmd_args,parameter_axes={"Omega_m":0,"w":1,"Sigma8Om0.55":2},cosmo_labels={"Omega_m":r"$\Omega_m$","w":r"$w$","Sigma8Om0.55":r"$\sigma_8(\Omega_m/0.27)^{0.55}$"},select="Omega_m")

##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################

def contours_combine(cmd_args,descriptors_in_plot=multiple,parameter_axes={"Omega_m":0,"w":1,"sigma8":2},cosmo_labels={"Omega_m":r"$\Omega_m$","w":r"$w$","sigma8":r"$\sigma_8$"},select="w",marginalize_over="me",mock=False,cross_label="cross"):

	#decide if consider data or simulations
	if mock:
		mock_prefix="mock"
	else:
		mock_prefix=""

	#Smoothing scales in arcmin
	levels = [0.684]

	#Parametrization hash
	par = parameter_axes.keys()
	par.sort(key=parameter_axes.__getitem__)
	par_hash = "-".join(par)

	#Parse options from configuration file
	options = ConfigParser.ConfigParser()
	with open(cmd_args.options_file,"r") as configfile:
		options.readfp(configfile)

	#Create figure
	fig,ax = plt.subplots(figsize=(8,8))

	#Plot labels
	contour_labels = list()

	#Cycle over descriptors
	for n,descr in enumerate(descriptors_in_plot):

		#Instantiate contour plot
		if marginalize_over=="me":
			contour = ContourPlot(fig=fig,ax=ax)
		elif marginalize_over=="others":
			contour = ContourPlot()
		else:
			raise ValueError("marginalize_over must be in [me,others]")


		#Construct the likelihood file
		if type(descr)==str:
			likelihood_file = os.path.join(root_dir,"likelihoods_{0}".format(par_hash),"likelihood{0}_{1}--{2:.1f}_ncomp{3}.npy".format(mock_prefix,descr,smoothing_scales[descr],num_components[descr]))
			contour_labels.append(descriptors[descr]+r"$({0})$".format(num_components[descr]))
		elif type(descr)==tuple:
			likelihood_file = os.path.join(root_dir,"likelihoods_{0}".format(par_hash),"likelihood{0}_cross_{1}.npy".format(mock_prefix,_cross_name(*descr)))
			contour_labels.append(_cross_label(*descr))
		else:
			raise TypeError("type not valid")

		#Log filename
		print("Loading likelihood from {0}".format(likelihood_file))
		
		#Load the likelihood
		contour.getLikelihood(likelihood_file,parameter_axes=parameter_axes,parameter_labels=cosmo_labels)
		
		#Set the physical units
		contour.getUnitsFromOptions(options)

		if marginalize_over=="me":
		
			#Marginalize
			contour.marginalize(select)
		
			#Best fit
			maximum = contour.getMaximum()
			print("Likelihood with {0} is maximum at {1}".format(descr,maximum))

			#Get levels
			contour.getLikelihoodValues(levels=levels)

			#Plot contours
			contour.plotContours(colors=[brew_colors[n]],fill=False,display_maximum=False,display_percentages=False,alpha=1.0)

		else:
			
			p,l,pmax,p0 = contour.marginal(select,levels=[0.684])
			print(pmax,p0)
			ax.plot(p,l,color=brew_colors[n],label=contour_labels[-1])		
			

	
	if marginalize_over=="me":
	
		#Legend
		contour.title_label=""
		contour.labels(contour_labels)

		#Save
		par.pop(par.index(select))
		par_hash = "-".join(par).replace(".","")
		fig.savefig("contours{0}{1}_{2}.{3}".format(mock_prefix,par_hash,cross_label,cmd_args.type))	

	else:

		#Legend
		ax.set_xlabel(cosmo_labels[select],fontsize=22)
		ax.set_ylabel(r"$\mathcal{L}$" + "$($" + cosmo_labels[select] + "$)$",fontsize=22)
		ax.legend(loc="upper left",prop={"size":10})

		#Save
		fig.savefig("contours{0}{1}_{2}.{3}".format(mock_prefix,select.replace(".",""),cross_label,cmd_args.type))

##################################################################################################################################################

def contours_single(cmd_args):

	contours_combine(cmd_args,descriptors_in_plot=single,cross_label="single")

##################################################################################################################################################

def contours_single_mock(cmd_args):

	contours_combine(cmd_args,descriptors_in_plot=single,cross_label="single",mock=True)

##################################################################################################################################################

def contours_single_reparametrize(cmd_args):

	contours_combine(cmd_args,descriptors_in_plot=single,cross_label="single",parameter_axes={"Omega_m":0,"w":1,"Sigma8Om0.55":2},cosmo_labels={"Omega_m":r"$\Omega_m$","w":r"$w$","Sigma8Om0.55":r"$\Sigma_8$"},select="Omega_m")

##################################################################################################################################################

def contours_single_mock_reparametrize(cmd_args):

	contours_combine(cmd_args,descriptors_in_plot=single,cross_label="single",parameter_axes={"Omega_m":0,"w":1,"Sigma8Om0.55":2},cosmo_labels={"Omega_m":r"$\Omega_m$","w":r"$w$","Sigma8Om0.55":r"$\Sigma_8$"},select="Omega_m",mock=True)

##################################################################################################################################################

def Si8_likelihood_single(cmd_args):

	contours_combine(cmd_args,descriptors_in_plot=single,cross_label="single",parameter_axes={"Omega_m":0,"w":1,"Sigma8Om0.55":2},cosmo_labels={"Omega_m":r"$\Omega_m$","w":r"$w$","Sigma8Om0.55":r"$\Sigma_8$"},select="Sigma8Om0.55",marginalize_over="others")

##################################################################################################################################################

def contours_combine_reparametrize(cmd_args):

	contours_combine(cmd_args,parameter_axes={"Omega_m":0,"w":1,"Sigma8Om0.55":2},cosmo_labels={"Omega_m":r"$\Omega_m$","w":r"$w$","Sigma8Om0.55":r"$\Sigma_8$"},select="Omega_m")

##################################################################################################################################################

def contours_combine_mock(cmd_args):

	contours_combine(cmd_args,mock=True)

##################################################################################################################################################

def contours_combine_mock_reparametrize(cmd_args):

	contours_combine(cmd_args,parameter_axes={"Omega_m":0,"w":1,"Sigma8Om0.55":2},cosmo_labels={"Omega_m":r"$\Omega_m$","w":r"$w$","Sigma8Om0.55":r"$\sigma_8(\Omega_m/0.27)^{0.55}$"},select="Omega_m",mock=True)

###################################################################################################################################################

def w_likelihood_combine(cmd_args):

	contours_combine(cmd_args,marginalize_over="others")

###################################################################################################################################################

def w_mock_likelihood_combine(cmd_args):

	contours_combine(cmd_args,marginalize_over="others",mock=True)

###################################################################################################################################################

def Si8_likelihood_combine(cmd_args):

	contours_combine(cmd_args,parameter_axes={"Omega_m":0,"w":1,"Sigma8Om0.55":2},cosmo_labels={"Omega_m":r"$\Omega_m$","w":r"$w$","Sigma8Om0.55":r"$\Sigma_8$"},select="Sigma8Om0.55",marginalize_over="others")

##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################

figure_method = dict()
figure_method["2"] = design
figure_method["3"] = pca
figure_method["4"] = robustness
figure_method["4s"] = robustness_1d
figure_method["4rep"] = robustness_reparametrize
figure_method["4srep"] = robustness_1d_reparametrize
figure_method["5"] = contours_single
figure_method["5b"] = contours_single_mock
figure_method["6"] = contours_single_reparametrize
figure_method["7"] = Si8_likelihood_single
figure_method["8"] = contours_combine
figure_method["8b"] = contours_combine_reparametrize
figure_method["9"] = Si8_likelihood_combine


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

