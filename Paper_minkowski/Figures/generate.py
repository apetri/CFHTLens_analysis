import sys,os,argparse,ConfigParser
from lenstools.constraints import LikelihoodAnalysis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from contours import ContourPlot

axes_facecolor = rc.func_globals["rcParams"]["axes.facecolor"]

root_dir = "/Users/andreapetri/Documents/Columbia/CFHTLens_analysis/andrea/convergence_map_analysis/cfht_masked_BAD_clipped"
brew_colors = ["red","green","blue","black","orange"]

#Descriptor list
descriptors=dict()
descriptors["power_spectrum"]="Power spectrum"
descriptors["minkowski_0"]=r"$V_0$"
descriptors["minkowski_1"]=r"$V_1$"
descriptors["minkowski_2"]=r"$V_2$"
descriptors["moments"]="Moments"

keys = descriptors.keys()
keys.sort()


def pca():

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
	fig.savefig("pca_components.eps")


def contours():

	#Smoothing scales in arcmin
	smoothing_scale=1.0
	n_components=3
	levels = [0.684,0.001]

	#Parameters of which we want to compute the confidence estimates
	parameter_axes = {"Omega_m":0,"w":1,"sigma8":2}
	cosmo_labels = {"Omega_m":r"$\Omega_m$","w":r"$w$","sigma8":r"$\sigma_8$"}
	
	#Parse command line options
	parser = argparse.ArgumentParser(prog=sys.argv[0])
	parser.add_argument("-f","--file",dest="options_file",action="store",type=str,help="analysis options file")

	cmd_args = parser.parse_args()

	if cmd_args.options_file is None:
		parser.print_help()
		sys.exit(0)

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
		likelihood_file = os.path.join(root_dir,"likelihoods","likelihood_{0}--{1:.1f}_ncomp{2}.npy".format(descr,smoothing_scale,n_components))
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
		contour_marg.plotContours(colors=[brew_colors[n],axes_facecolor],fill=True,display_maximum=False,display_percentages=False,alpha=0.45)
		contour_slice.plotContours(colors=[brew_colors[n],axes_facecolor],fill=True,display_maximum=False,display_percentages=False,alpha=0.45)

	
	#Legend
	contour_marg.title_label=""
	contour_slice.title_label=""
	contour_marg.labels([descriptors[key] for key in keys])
	contour_slice.labels(None)

	#Save
	fig.savefig("contours_{0}comp.pdf".format(n_components))






if __name__=="__main__":
	contours()


