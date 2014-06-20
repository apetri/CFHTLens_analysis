from .equations import *

import numpy as np
from scipy.optimize import minimize

class Process:
	"""
	This is an object wrapper for the equations module, which actually performs all the linear algebra;
	the goal is to interpolate a scalar function that depends on N parameters knowing its values on 
	S different simulation points (each of these points is a vector of size N).
	The constructor takes the following parameters:

	:param cosmoParams:
		S x N numpy array with the S choices of N parameters (these are the points on which we run the simulations)

	:param simData:
		S sized numpy array with the values of the particular descriptor bin on each of the simulation points

	"""

	def __init__(self,cosmoParams,simData):

		#Check validity of initialization
		if len(simData.shape) != 1:
			raise ValueError("simData array must be one dimensional! (scalar functions only for now :( )")

		if simData.shape[0] != cosmoParams.shape[0]:
			raise ValueError("there should be an entry of simData for each simulation point!!")

		#Initialize
		self.cosmoParams = cosmoParams
		self.simData = simData

	def interpolate(self,hyperParameters=None):
		"""
		This method returns an interpolation function handler similarly to the one scipy.interp1d does; the interpolation
		needs the value of the hyperparameters in order to continue, so either you supply it as a 1d numpy array of size N+3
		or an exception is raised. You can also try to find the optimum hyperparameter values by running hyperOptimumFind.

		:param hyperParameters:
			numpy array of size N + 3 with the hyperparameters used to perform the interpolation
		"""

		#Check for presence of hyperparameters, if none can be found throw an exception
		if hyperParameters == None:
			
			if hasattr(self,"hyperParameters"):
				hyperParameters = self.hyperParameters
			else:
				raise ValueError("I don't know which values for the hyperparameters to use!")

		self.covFunction = np.exp(fullLogCov(self.cosmoParams,hyperParameters))
		self.covinvFunction = np.linalg.inv(self.covFunction)

		#Build the function handler to return
		def interpolator(newPoint):

			#Check validity of input
			if(newPoint.shape[1] != self.cosmoParams.shape[1]):
				raise ValueError("new points should live in the same dimension as the input points!!")

			Cvec = np.exp(logCvector(newPoint,hyperParameters,self.cosmoParams,self.simData))
			Cnum = Cnumber(newPoint,hyperParameters)

			return interpolatedValue(Cvec,self.covinvFunction,self.simData),errorBar(Cvec,Cnum,self.covinvFunction).diagonal()

		#Return the handler
		return interpolator

	def hyperOptimumFind(self,bounds,Nguesses=1):
		"""
		Find the values of the hyperparameters that maximize the likelihood; sets the hyperParameters attributes to the best
		combination found and the allHyperParameters attribute to all the optima that are found

		:param bounds:
			tuple of couples (min,max) for each one of the hyperparameters

		:param Nguesses:
			how many guesses to make (useful if you want to be sure to find the actual maximum)

		After function call, the Process instance will have the following attributes:

		:attr logLikelihood:
			best value for the likelihood found after Nguesses
		
		:attr allLogLikelihood:
			all Nguesses likelihood values found in the optimum search

		:attr hyperParameters:
			hyperparameter values that maximize the likelihood

		:attr allHyperParameters:
			all optimal hyperparameter values found in the optimum search
		"""

		#Check sanity of input
		if len(bounds) != self.cosmoParams.shape[1] + 3:
			raise ValueError("You must specify bounds for all the hyperparameters!")

		Nhyper = self.cosmoParams.shape[1] + 3

		self.logLikelihood = -np.inf
		self.allLogLikelihood = []
		self.hyperParameters = None
		self.allHyperParameters = None 

		##############################
		##Proceed to find the optima##
		##############################

		#First produce the guesses
		hyperGuess = np.zeros((Nguesses,Nhyper))
		for i in range(Nhyper):
			
			hyperGuess[:,i] = np.random.uniform(low=bounds[i][0],high=bounds[i][1],size=Nguesses)

		#For each guess use the scipy minimize utility to find the maximum
		for j in range(Nguesses):

			hyperFound = minimize(logHyperLikelihood,x0=hyperGuess[j],args=(self.cosmoParams,self.simData,True),bounds=bounds)

			print "Trial %d"%(j+1),hyperFound.success,hyperFound.message

			if hyperFound.success :
				
				self.allLogLikelihood.append(-hyperFound.fun)
				
				if self.allHyperParameters == None:
					self.allHyperParameters = hyperFound.x
				else:
					self.allHyperParameters = np.vstack((self.allHyperParameters,hyperFound.x))

				if (-hyperFound.fun) > self.logLikelihood:
					self.logLikelihood = -hyperFound.fun
					self.hyperParameters = hyperFound.x 





