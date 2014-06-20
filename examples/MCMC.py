import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import emcee
from emcee.utils import MPIPool

import triangle

import GaussianProcess
import GaussianProcess.equations as equations

#Initialize MPI pool used for parallelization
pool = MPIPool()

if not pool.is_master():

	pool.wait()
	sys.exit(0)


#Interpolate the function using a Gaussian process (find the optimum and confidence contours with MCMC)
cosmoParams = np.array([1.0,2.0,3.0,4.0,5.0])
simData = np.array([1.5,2.0,-0.5,4.0,5.0])

if pool.is_master():
	process = GaussianProcess.Process(cosmoParams[:,np.newaxis],simData)
	process.hyperOptimumFind(bounds=((0.1,3.9),(0.1,3.9),(0.1,3.9),(0.1,3.9)),Nguesses=15)

#Give it a shot with MCMC

#4 parameters
ndim=4

#32 walkers
nwalkers=32

#Initialize positions of the walkers
p0 = np.ones((nwalkers,ndim))
p0[:,0] = np.random.uniform(low=0.1,high=3.9,size=nwalkers)
p0[:,1] = np.random.uniform(low=0.1,high=3.9,size=nwalkers)
p0[:,2] = np.random.uniform(low=0.1,high=3.9,size=nwalkers)
p0[:,3] = np.random.uniform(low=0.1,high=3.9,size=nwalkers)

#Sampler initialization
sampler = emcee.EnsembleSampler(nwalkers,ndim,equations.logHyperLikelihood,args=[cosmoParams[:,np.newaxis],simData,False],pool=pool)

#Burn in 

if pool.is_master():
	print "burn in..."

pos,prob,state = sampler.run_mcmc(p0,1000)
sampler.reset()

#Run chain after burn in

if pool.is_master():
	print "running MCMC..."

sampler.run_mcmc(pos,10000)

#Plot result

if pool.is_master():
	
	print "Plotting..."
	p = sampler.flatchain
	fig = triangle.corner(np.array([p[:,3],p[:,2]]).transpose(),plot_datapoints=False,labels=[r"$r_1$",r"$\theta_3$"])
	fig.axes[2].scatter(process.allHyperParameters[:,3],process.allHyperParameters[:,2],marker="x",color="red")
	plt.savefig("mcmc_likelihood.png")

#Quit processes
pool.close()