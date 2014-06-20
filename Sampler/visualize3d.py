import os,sys,glob

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##############################################
########Tunable options#######################
##############################################

frames_directory = "frames"
keep_frames = False
frames_per_second = 10
plot_projections = True

scale_to_cosmological = True
parameter_labels = [r"$\Omega_m$",r"$w$",r"$\sigma_8$"]
parameter_low = np.array([0.07,-3.0,0.1])
parameter_high = np.array([1.0,0.0,1.5])

#Prompt user for correct usage
if(len(sys.argv)<3):
	print "Usage python %s <points_filename> <output_movie_filename>"
	exit(1)

#Create directory for frames
try:
	os.mkdir(frames_directory)
except OSError:
	pass

#Load txt file with points in 3D space
points = np.loadtxt(sys.argv[1])

if(points.shape[1] != len(parameter_labels)):
	raise ValueError("parameter_labels and number of parameters must be the same!")

#Maybe scale to cosmological parameter values?
if(scale_to_cosmological):
	points = parameter_low[np.newaxis,:] + points * (parameter_high[np.newaxis,:] - parameter_low[np.newaxis,:])

	#Save the actual cosmological parameters values to a txt file
	rootName = sys.argv[1].split(".")[0]
	np.savetxt(rootName+"_cosmological.txt",points) 


#Plot the 2D projections first if option is provided
if(plot_projections):

	plt.scatter(points[:,0],points[:,1],color="red")
	
	if(scale_to_cosmological):
		plt.xlabel(parameter_labels[0],fontsize=16)
		plt.ylabel(parameter_labels[1],fontsize=16)
		plt.xlim(parameter_low[0],parameter_high[0])
		plt.ylim(parameter_low[1],parameter_high[1])
	else:
		plt.xlabel(r"$x_1$",fontsize=16)
		plt.ylabel(r"$x_2$",fontsize=16)
		plt.xlim(0,1)
		plt.ylim(0,1)

	plt.savefig("projection1.png")
	plt.clf()

	if(points.shape[1]>2):

		plt.scatter(points[:,1],points[:,2],color="red")
	
		if(scale_to_cosmological):
			plt.xlabel(parameter_labels[1],fontsize=16)
			plt.ylabel(parameter_labels[2],fontsize=16)
			plt.xlim(parameter_low[1],parameter_high[1])
			plt.ylim(parameter_low[2],parameter_high[2])
		else:
			plt.xlabel(r"$x_2$",fontsize=16)
			plt.ylabel(r"$x_3$",fontsize=16)
			plt.xlim(0,1)
			plt.ylim(0,1)

		plt.savefig("projection2.png")
		plt.clf()

		plt.scatter(points[:,2],points[:,0],color="red")
	
		if(scale_to_cosmological):
			plt.xlabel(parameter_labels[2],fontsize=16)
			plt.ylabel(parameter_labels[0],fontsize=16)
			plt.xlim(parameter_low[2],parameter_high[2])
			plt.ylim(parameter_low[0],parameter_high[0])
		else:
			plt.xlabel(r"$x_3$",fontsize=16)
			plt.ylabel(r"$x_1$",fontsize=16)
			plt.xlim(0,1)
			plt.ylim(0,1)

		plt.savefig("projection3.png")
		plt.clf()


#########################################################################
########The 3D part is performed only if there are enough dimensions#####
#########################################################################

if(points.shape[1]>2):

	#Setup 3D plot
	fig = plt.figure()
	ax = fig.add_subplot(111,projection="3d")


	#Scatter the points in the space
	ax.scatter(points[:,0],points[:,1],points[:,2],color="red")

	#Label the axes
	if(scale_to_cosmological):
		ax.set_xlabel(parameter_labels[0],fontsize=16)
		ax.set_ylabel(parameter_labels[1],fontsize=16)
		ax.set_zlabel(parameter_labels[2],fontsize=16)
	else:
		ax.set_xlabel(r"$x_1$",fontsize=16)
		ax.set_ylabel(r"$x_2$",fontsize=16)
		ax.set_zlabel(r"$x_3$",fontsize=16)

	#Set axes limits
	if(scale_to_cosmological):
		ax.set_xlim(parameter_low[0],parameter_high[0])
		ax.set_ylim(parameter_low[1],parameter_high[1])
		ax.set_zlim(parameter_low[2],parameter_high[2])
	else:
		ax.set_xlim(0,1)
		ax.set_ylim(0,1)
		ax.set_zlim(0,1)

	#Rotate the camera and save corresponding frames
	i=0
	theta = 10

	for phi in xrange(0,180,1):
		print "Swiping azimuth: %d degrees"%phi
		ax.view_init(elev=10.,azim=phi)
		plt.savefig("%s/frame_%d.png"%(frames_directory,i))
		i+=1

	for theta in xrange(10,180,1):
		print "Swiping elevation: %d degrees"%theta
		ax.view_init(elev=theta,azim=phi)
		plt.savefig("%s/frame_%d.png"%(frames_directory,i))
		i+=1

	for phi in xrange(180,360,1):
		print "Swiping azimuth: %d degrees"%phi
		ax.view_init(elev=theta,azim=phi)
		plt.savefig("%s/frame_%d.png"%(frames_directory,i))
		i+=1

	#Make the video with ffmpeg
	os.system("ffmpeg -f image2 -r %d -i %s/frame_%%d.png -vcodec mpeg4 -y %s"%(frames_per_second,frames_directory,sys.argv[2]))

	#Remove the frames if option is provided
	if(not(keep_frames)):

		print "Cleaning up frames..."

		frameFiles = glob.glob("%s/*"%frames_directory)
		for frame in frameFiles:
			os.remove(frame)

		os.rmdir(frames_directory)

print "DONE!\n\n"