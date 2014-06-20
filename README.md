Weak gravitational lensing CFHTLens analysis
=================

Analysis pipeline for CFTHLens: there are two directories, andrea and jia, read below for documentation regarding the code hereby contained. 


Note
----

If you need to use python on Yeti, with the associated packages (numpy, astropy, etc...), you need to use this one

    /vega/astro/users/jl3509/tarball/anacondaa/bin/python

and set your PYTHONPATH environment variable to

    /vega/astro/users/jl3509/tarball/anacondaa/lib/python2.7/site-packages

in order to use the hereby installed packages

Andrea
------

There is a python script in the "andrea" directory, which computes the two point angular correlation function of the galaxy shear by summing over galaxy pairs (this algorithm doesn't require pixelization, but it's not very efficient, even with heavy vectorization). They way you use it is 

    python catalog_ps.py  <catalog_filename>  <output_filename_corr.npy>  <output_filename_power.npy>

The output format is a numpy array with two columns: one with the angle in arcmin, the other with the corresponding level of correlation

Jia
---

Weak Lensing Emulator
===========

Weak lensing emulator for non linear statistics

MCMC:
Uses python code emcee. Input needed: prior, likelihood function (bridges to the Interpolator), observations(peaks, MF, etc.) to be fitted to.

