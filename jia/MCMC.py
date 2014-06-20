# !python
# This code is the last step in WL emulator
# It takes in any likelihood function, output the parameter probability distribution.
# Last modification by Jia (05/08/2014)

import numpy as np
from scipy import *
import scipy.optimize as op
import emcee

# params
steps = 100 # MCMC steps, total steps will be steps*nwalkers
dp = array([0.03, 0.2, 0.05]) # dp between hi-cosmo and fidu-cosmo
fidu_params = array([0.26, -1.0, 0.8])

# Build test model (linear interpolation)
cosmo_mat=(genfromtxt('cosmo_mat_13subfields_1000R_021bins')).reshape(4,1000,-1)
cov_mat = np.cov(cosmo_mat[0], rowvar = 0)# rowvar is the row contaning observations
cov_inv = np.mat(cov_mat).I # covariance matrix inverted
fidu_avg, him_avg, hiw_avg, his_avg = mean(cosmo_mat, axis = 1)

dNdm = (him_avg - fidu_avg)/dp[0]
dNdw =(hiw_avg - fidu_avg)/dp[1] 
dNds = (his_avg - fidu_avg)/dp[2]
X = np.mat([dNdm, dNdw, dNds])

# fit using analytical solution
def cosmo_fit (obs):
	Y = np.mat(obs-fidu_avg)
	del_p = ((X*cov_inv*X.T).I)*(X*cov_inv*Y.T)
	m, w, s = np.squeeze(np.array(del_p.T))+fidu_params
	del_N = Y-del_p.T*X
	chisq = float(del_N*cov_inv*del_N.T)
	return chisq, m, w, s
fitSIM = array(map(cosmo_fit,cosmo_mat[0]))

#### MCMC using emcee ##########

obs = fidu_avg # use fiducial cosmology as observation, to be fitted parameters.

# prior
def lnprior(params):
	'''This gives the flat prior.
	Returns:
	0:	if the params are in the regions of interest.
	-inf:	otherwise.'''
	m, w, s = params
	if -0.23 < m < 0.78 and -5.36 < w < 3.15 and -0.1 < s < 1.49:
		return 0.0
	else:
		return -np.inf

# likelihood function
def lnlike (params, obs):
	'''This gives the likelihood function, assuming Gaussian distribution.
	Returns: -log(chi-sq)
	'''
	model = fidu_avg + mat(array(params)-fidu_params)*X
	Y = np.mat(model - obs)
	del_p = ((X*cov_inv*X.T).I)*(X*cov_inv*Y.T)
	del_N = Y-del_p.T*X
	ichisq = del_N*cov_inv*del_N.T
	Ln = -log(ichisq) #likelihood, is the log of chisq
	return float(Ln)

def lnprob(params, obs):
	lp = lnprior(params)
	if not np.isfinite(lp):
		return -np.inf
	else:
		return lp + lnlike(params, obs)

# find the best fit point, using minimize, or can simply be fiducial value
# this is the starting position of the chain

nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, fidu_params*1.1, args=(obs,))

ndim, nwalkers = 3, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

print 'run sampler'
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(obs,))

print 'run mcmc'
sampler.run_mcmc(pos, steps)
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

errors = array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84],axis=0))))
print 'Result (best fits, lower error, upper error)'
print 'Omega_m:\t', errors[0]
print 'w:\t\t', errors[1]
print 'sigma_8:\t', errors[2]