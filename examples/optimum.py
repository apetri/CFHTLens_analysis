import GaussianProcess

import numpy as np
from scipy.optimize import minimize

#Interpolate the function using a Gaussian process (find the optimum)
cosmoParams = np.array([1.0,2.0,3.0,4.0,5.0])
simData = np.array([1.5,2.0,-0.5,4.0,5.0])

#Build process object
process = GaussianProcess.Process(cosmoParams[:,np.newaxis],simData)

#Find maximul likelihood hyperparameters
process.hyperOptimumFind(bounds=((0.01,4.9),(0.01,4.9),(0.01,4.9),(0.01,4.9)),Nguesses=5)