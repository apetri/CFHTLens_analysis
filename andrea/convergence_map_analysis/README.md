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

or 

	mpiexec -n 101 python likelihood.py -f options.ini

4. Plot the confidence contours
-------------------------------

	python contours.py likelihood.npy -f options.ini



