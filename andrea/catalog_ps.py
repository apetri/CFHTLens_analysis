import sys

import numpy as np
from scipy.spatial import cKDTree

#Prompt user for correct usage
if(len(sys.argv)<3):
	print "Usage python %s <catalog_file> <output_file>"%sys.argv[0]
	exit(1)

#Load info from catalog: columns (0,1,9,10,11,16,17)=(x,y,w,e1,e2,m,c2)
x,y,w,e1,e2,m,c2 = np.loadtxt(sys.argv[1],usecols=[0,1,9,10,11,16,17]).transpose()

#Print size of catalog
print "This catalog contains %d galaxies"%len(x)

#Build outer products
e1_ij = np.outer(e1,e1)
e2_ij = np.outer(e2,e2)
w_ij = np.outer(w,w)

#Decide binning of the 2pt function (theta is the midpoint of each bin)
Nbins = 10
step = (x.max()-x.min())/Nbins
theta = np.arange(step/2,x.max()-x.min()-step/2,step)
corr = np.zeros(Nbins)
weight = np.zeros(Nbins)

#Group x and y into an array to build the KD tree
X = np.array([x,y]).transpose()

#Build the KD tree
kdt = cKDTree(X)

#Compute the 2pt function by summing over galaxy pairs
for i in range(Nbins):
	
	#Select pairs separated by theta[i] querying the tree
	print "Computing correlation bin %d: theta=%.5f"%(i+1,theta[i])
	p = kdt.query_pairs(theta[i]+step/2)

	#Vectorize
	I = np.array(list(p)).transpose()

	#Sum over the pairs
	corr[i] = ((e1_ij[I[0],I[1]] + e2_ij[I[0],I[1]])*w_ij[I[0],I[1]]).sum()
	weight[i] = w_ij[I[0],I[1]].sum()

#What we computed is a cumulative correlation function, let's recover the differential one
corr[1:] = corr[1:] - corr[:Nbins-1]
weight[1:] = weight[1:] - weight[:Nbins-1]

#Output the correlation function
np.save(sys.argv[2],np.array([theta,corr/weight]))

print "Done!"