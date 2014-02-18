import sys

from mpi4py import MPI

import numpy as np
from scipy.spatial import cKDTree

#Initialize MPI communicator
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#Prompt user for correct usage
if(len(sys.argv)<3 and rank==0):
	print "Usage python %s <catalog_file> <output_file>"%sys.argv[0]
	exit(1)

#Read total number of galaxies from catalog file
if(rank==0):
	print "Determining catalog size..."

x = np.loadtxt(sys.argv[1])[:,0]
numGal = len(x)
numGalPerTask = numGal/size

if(rank==0):
	print "Catalog contains %d galaxies"%numGal

#Decide binning of the 2pt function (theta is the midpoint of each bin)
step = (x.max()-x.min())/10
theta = np.arange(step/2,x.max()-x.min()-step/2,step)
Nbins = len(theta)

#Load info from catalog: columns (0,1,9,10,11,16,17)=(x,y,w,e1,e2,m,c2)
if(rank==0):
	print "Reading catalog..."
x,y,w,e1,e2,m,c2 = np.loadtxt(sys.argv[1],usecols=[0,1,9,10,11,16,17])[rank*numGalPerTask:(rank+1)*numGalPerTask,:].transpose()

#Build outer products
e1_ij = np.outer(e1,e1)
e2_ij = np.outer(e2,e2)
w_ij = np.outer(w,w)

#Define containers for binned 2pt function
corrLoc = np.zeros(Nbins)
corrGlob = np.zeros(Nbins)
weightLoc = np.zeros(Nbins)
weightGlob = np.zeros(Nbins)

#Group x and y into an array to build the KD tree
X = np.array([x,y]).transpose()

#Build the KD tree
kdt = cKDTree(X)

#Compute the 2pt function by summing over galaxy pairs
for i in range(Nbins):
	
	#Select pairs separated by theta[i] querying the tree
	if(rank==0):
		print "Computing correlation bin %d"%(i+1)
	
	pUse = kdt.query_pairs(theta[i]+step/2)
	pRem = kdt.query_pairs(theta[i]-step/2)
	pUse.difference_update(pRem)

	#Vectorize
	I = np.array(list(pUse)).transpose()

	if(len(I.shape)==2):
		#Sum over the pairs
		corrLoc[i] = ((e1_ij[I[0],I[1]] + e2_ij[I[0],I[1]])*w_ij[I[0],I[1]]).sum()
		weightLoc[i] = w_ij[I[0],I[1]].sum()

#Reduce results from all tasks
if(rank==0):
	print "Reducing from all tasks..."

comm.Barrier()
print rank
comm.Reduce([corrLoc,MPI.DOUBLE],[corrGlob,MPI.DOUBLE],op=MPI.SUM,root=0)
comm.Reduce([weightLoc,MPI.DOUBLE],[weightGlob,MPI.DOUBLE],op=MPI.SUM,root=0)

#Output results
comm.Barrier()
if(rank==0):
	np.save(sys.argv[2],np.array([theta,corrGlob/weightGlob]))
	print "Done!!"