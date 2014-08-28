from __future__ import print_function,division,with_statement

import os,sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

#TODO: these options are hard coded, make them command line instead in the future

cfht_directory = "cfht_masked_BAD"
feature_string = "minkowski_2--1.0"
feature_label_file = "new_th_minkowski.npy"
feature_label_x = r"$\kappa$"
feature_label_y = r"$V_2$"

def main():

	#Load the observed and the interpolated feature
	feature_label = np.load(os.path.join(cfht_directory,feature_label_file))
	feature_step = (feature_label[1:] - feature_label[:-1]).mean()
	
	observed_feature = np.load(os.path.join(cfht_directory,"observation_")+feature_string+".npy")
	observed_covariance = np.load(os.path.join(cfht_directory,"covariance_"+feature_string+".npy"))
	interpolated_feature = np.load(os.path.join(cfht_directory,"testinterp_")+feature_string+".npy")

	#Build the figure
	fig = plt.figure(figsize=(16,8))
	gs = gridspec.GridSpec(2,1,height_ratios=[3,1])

	#Plot the observed and interpolated features on top
	ax0 = plt.subplot(gs[0])
	ax0.errorbar(feature_label,observed_feature,yerr=np.sqrt(observed_covariance.diagonal()),color="black",label="CFHT_andrea",linestyle="-")
	ax0.plot(feature_label,interpolated_feature,color="red",label="interpolation [0.26,-1.0,0.8] andrea",linestyle="--")
	ax0.set_xlabel(feature_label_x)
	ax0.set_ylabel(feature_label_y)
	ax0.legend(loc="upper right")

	#Plot the fractional difference at the bottom
	ax1 = plt.subplot(gs[1])
	ax1.plot(feature_label,observed_feature - observed_feature,color="black")
	ax1.plot(feature_label,(interpolated_feature - observed_feature)/np.sqrt(observed_covariance.diagonal()),color="red",linestyle="--")
	ax1.set_xlabel(feature_label_x)
	ax1.set_ylabel(r"$\Delta_{perc}$")

	#Save the figure
	fig.tight_layout()
	fig.savefig(os.path.join(cfht_directory,"check_")+feature_string+".png")


if __name__=="__main__":
	main()