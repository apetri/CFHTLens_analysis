import numpy as np
from scipy.misc import logsumexp

#################################################################################
##################These are useful to compute the covariance function############
#################################################################################

def logCovTemplate(x,w):
	return -0.5*x**2/w**2


def logCovFunction(cosmoParameters,hyperParameters):

	if(cosmoParameters.shape[1] != len(hyperParameters[3:])):
		raise IndexError("Number of hyperparameters must be number of cosmological parameters + 3!")

	#Tensorize
	cosmoTensor = cosmoParameters[np.newaxis,:,:] - cosmoParameters[:,np.newaxis,:]

	#Compute products
	return logCovTemplate(cosmoTensor,hyperParameters[np.newaxis,np.newaxis,3:]).sum(axis=2)

def fullLogCov(cosmoParameters,hyperParameters):

	if(len(hyperParameters) != cosmoParameters.shape[1]+3):
		raise IndexError("Number of hyperparameters must be number of cosmological parameters + 3!")

	simpleCov = logCovFunction(cosmoParameters,hyperParameters)

	#Combine with the theta hyperparameters
	denseAddOn = np.zeros(simpleCov.shape)
	
	noiseAddOn = -np.ones(simpleCov.shape) * np.inf
	noiseAddOn[np.diag_indices(simpleCov.shape[0])] = 0.0

	#Perform the logsumexp
	return logsumexp(np.array([simpleCov,denseAddOn,noiseAddOn]),axis=0,b=hyperParameters[:3,np.newaxis,np.newaxis])

#####################################################################################################
################Construction of the likelihood for the hyperparameters###############################
################useful for MCMC optimum hyperparameters searches#####################################
#####################################################################################################

#Here symData is just a vector with as many elements as the number of simulations
def logSimLikelihood(parLogCov,simData):

	if(parLogCov.shape[0]!=len(simData)):
		raise IndexError("parameter covariance and simData must be of the same size!!")

	covInv = np.linalg.inv(np.exp(parLogCov))
	return -0.5*np.dot(simData,np.dot(covInv,simData))

#Go with flat priors to start
def logHyperPrior(hyperParameters):

	for i in range(len(hyperParameters)):
		if hyperParameters[i]<0.1 or hyperParameters[i]>4.0:
			return -np.inf

	return 0.0

#Build full hyperparameter likelyhood (remember to add prior in the return if this becomes more compliated!!!!!)
def logHyperLikelihood(hyperParameters,cosmoParameters,simData,negative=False):

	if logHyperPrior(hyperParameters) == -np.inf:
		return -np.inf

	parLogCov = fullLogCov(cosmoParameters,hyperParameters)

	if(negative):
		return -logSimLikelihood(parLogCov,simData) 
	else:
		return logSimLikelihood(parLogCov,simData) 

##########################################################################################################
#####################Calculate the interpolated function at a particular point############################
#####################once you know the hyperparameters####################################################
##########################################################################################################

def logCvector(newPoint,hyperParameters,cosmoParameters,simData):

	bareLogCov = logCovTemplate(newPoint[:,np.newaxis,:]-cosmoParameters[np.newaxis,:,:],hyperParameters[3:]).sum(axis=2)
	return logsumexp(np.array([bareLogCov,np.zeros(bareLogCov.shape)]),axis=0,b=hyperParameters[:2,np.newaxis,np.newaxis])

#This is only a simpler version
def Cnumber(newPoint,hyperParameters):

	return np.ones(newPoint.shape[0])*(hyperParameters[0] + hyperParameters[1])

#Compute interpolated value
def interpolatedValue(cVector,covInv,simData):

	return np.dot(np.dot(cVector,covInv),simData) 

#Compute errorbar
def errorBar(cVector,cNumber,covInv):

	return cNumber * np.eye(len(cNumber)) - np.dot(cVector,np.dot(covInv,cVector.transpose()))




