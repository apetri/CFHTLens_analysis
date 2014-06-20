import numpy as np
import matplotlib.pyplot as plt

import GaussianProcess


#Interpolate the function using a Gaussian process 
cosmoParams = np.array([1.0,2.0,3.0,4.0,5.0])
simData = np.array([1.5,2.0,-0.5,4.0,5.0])

#Build the gaussian process object knowing the simulation data at the chosen points
process = GaussianProcess.Process(cosmoParams[:,np.newaxis],simData)

plt.plot(cosmoParams,simData,linestyle='none',marker='o',label='simulated data')

#Choose new points to interpolate
newPoint = np.arange(0.0,6.0,0.2)[:,np.newaxis]

#Try how it looks like using one set of hyperparameters

hyperparams = np.array([1.0,0.5,0.0,0.95])

#Build interpolator object
siminterp = process.interpolate(hyperparams) 

#Compute interpolated values and errorbars
intData,errData = siminterp(newPoint)

#Plot
plt.errorbar(newPoint,intData,yerr=np.sqrt(errData),color='green',label='noiseless interpolation')

#Try how it looks like using various another set of hyperparameters
hyperparams = np.array([1.0,0.5,3.0,3.5])

#Build interpolator object
siminterp = process.interpolate(hyperparams) 

#Compute interpolated values and errorbars
intData,errData = siminterp(newPoint)

#Plot
plt.errorbar(newPoint,intData,yerr=np.sqrt(errData),color='red',label='noisy interpolation')

plt.xlim(0,6)
plt.ylim(-3,7)
plt.xlabel(r'$x$',fontsize=18)
plt.ylabel(r'$f(x)$',fontsize=18)
plt.legend(loc='upper left')

plt.savefig("gaussInterp.png")