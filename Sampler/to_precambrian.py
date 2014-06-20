#######################################################
###This is a simple script that will translate#########
###the (Om,w,si8) values into a file that could########
###be read by Precambrian to set up the runs in########
##############the IG Pipeline##########################
import sys
import numpy as np

########################################################
#######Values of fixed cosmological parameters##########
########################################################

obh2 = 0.0227
wa = 0.00
ns = 0.960
As = 2.41e-9
h = 0.700

if(len(sys.argv)<3):
	print "Usage: python %s <(Om,w,si8) points file> <Precambrian output file>"%sys.argv[0]
	exit(1)

print "%s -----> %s"%(sys.argv[1],sys.argv[2])

points = np.loadtxt(sys.argv[1])

outFile = file(sys.argv[2],"w")

outFile.write("#Obh2   Om    Ol     w0     wa   ns     As     si8    h\n\n")

for i in range(points.shape[0]):
	outFile.write("%.4f %.3f %.3f %.3f %.3f %.3f %.2e %.3f %.3f\n"%(obh2,points[i,0],1-points[i,0],points[i,1],wa,ns,As,points[i,2],h))

outFile.close()

print "DONE!\n"