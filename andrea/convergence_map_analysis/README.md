To run the analysis you will need the [LensTools](http://www.columbia.edu/~ap3020/LensTools/html) python package. The CFHTLens analysis consists in 4 steps

1. Measure the required cosmological probes, or features (power spectrum, peaks, etc...), measure_features.py
2. Plot the measured features for a sanity check, plot_features.py
3. Compute the parameters likelihood using a multivariate interpolation between simulated cosmological models, likelihood.py
4. Plot the confidence contours in parameter space, contours.py 

Here are the steps in detail:

1. Feature measurements
------------------------

You can run the feature measurements by typing

    python measure_features.py -f options.ini
   
Or, for speedup you can run the measurements on multiple cores

	mpiexec -n 101 measure_features.py -f options.ini

2. Sanity check plots
---------------------

	python plot_features.py -f options.ini

A "plot" directory will appear in your save_path

3. Calculation of the parameter likelihood on a grid
-----------------------------

	python likelihood.py -f options.ini

or, if you want to speed up the $$$\chi^2$$$ calculations using MPI 

	mpiexec -n 101 python likelihood.py -f options.ini

A quick note on the options file: the parameter likelihood profile depends on which features (statistical descriptors) you want to use in the analysis. You can specify those in the options.ini file inputing a string in the _feature_types_ options. There is a particular syntax for the string formatting:

1. Different descriptors should be separated by a *
2. You must specify the smoothing scales after each descriptor type by a :
3. Different smoothing scales are to be separated by a ,
4. Current statistical features implemented are to be selected in (power_spectrum,moments,peaks,minkowski_012)

Suppose I want to compute the parameter likelihood using the power spectrum with 0.5 arcmin smoothing combined with the peak counts with 0.5 and 1.0 arcmin smoothing. Then the corresponding line in options.ini should be

	feature_types = power_spectrum:0.5 * peaks:0.5,1.0 

4. Plot the confidence contours
-------------------------------

	python contours.py likelihood.npy -f options.ini



