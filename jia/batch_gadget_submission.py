#!python
import glob
from scipy import *
def write_gadget_submission(ic):
	f = open('/work/02977/jialiu/lenstools_home/Jobs/gadget_5ic{0}.sh'.format(ic), 'w')
	content = '''#!/bin/bash

################################
######Allocation ID#############
################################

#SBATCH -A TG-AST140041


##########################################
#############Directives###################
##########################################

#SBATCH -J Gadget2_3ic{0}

#SBATCH -o /work/02977/jialiu/lenstools_home/Logs/gadget_5ic{0}_%j.err
#SBATCH -e /work/02977/jialiu/lenstools_home/Logs/gadget_5ic{0}_%j.err


#SBATCH -p normal
#SBATCH -t 4:00:00

#SBATCH --mail-user=jia@astro.columbia.edu
#SBATCH --mail-type=all


##########################################
#############Resources####################
##########################################

#SBATCH -n 1280
#SBATCH -N 80

###################################################
#################Execution#########################
###################################################

ibrun -n 256 -o 0 /work/02977/jialiu/IG_Pipeline_0.1/Gadget2/Gadget2 1 256 /work/02977/jialiu/lenstools_home/Om0.300_Ol0.700/512b240/ic{0}/gadget2.param &

ibrun -n 256 -o 256 /work/02977/jialiu/IG_Pipeline_0.1/Gadget2/Gadget2 1 256 /work/02977/jialiu/lenstools_home/Om0.300_Ol0.700/512b240/ic{1}/gadget2.param &

ibrun -n 256 -o 512 /work/02977/jialiu/IG_Pipeline_0.1/Gadget2/Gadget2 1 256 /work/02977/jialiu/lenstools_home/Om0.300_Ol0.700/512b240/ic{2}/gadget2.param &

ibrun -n 256 -o 768 /work/02977/jialiu/IG_Pipeline_0.1/Gadget2/Gadget2 1 256 /work/02977/jialiu/lenstools_home/Om0.300_Ol0.700/512b240/ic{3}/gadget2.param &

ibrun -n 256 -o 1024 /work/02977/jialiu/IG_Pipeline_0.1/Gadget2/Gadget2 1 256 /work/02977/jialiu/lenstools_home/Om0.300_Ol0.700/512b240/ic{4}/gadget2.param &

wait'''.format(ic, ic+1, ic+2, ic+3, ic+4)
	f.write(content)
	f.close()


#map(write_gadget_submission, range(5,501)[::5])

#########################################
################# N-GenIC ###############
#########################################

def write_ngenic_submission():
	f = open('/work/02977/jialiu/lenstools_home/Jobs/ngenic500.sh', 'w')
	content = '''#!/bin/bash

################################
######Allocation ID#############
################################

#SBATCH -A TG-AST140041


##########################################
#############Directives###################
##########################################

#SBATCH -J NGenIC

#SBATCH -o /work/02977/jialiu/lenstools_home/Logs/ngenic.%j.err
#SBATCH -e /work/02977/jialiu/lenstools_home/Logs/ngenic.%j.err


#SBATCH -p development
#SBATCH -t 02:00:00

#SBATCH --mail-user=jia@astro.columbia.edu
#SBATCH --mail-type=all


##########################################
#############Resources####################
##########################################

#SBATCH -n 128
#SBATCH -N 16

###################################################
#################Execution#########################
###################################################'''
	f.write(content)
	f.close()
	f = open('/work/02977/jialiu/lenstools_home/Jobs/ngenic500.sh', 'a')
	f.write('\n')
	for i in range(14,501)[::8]:
		for j in range(8):
			newline = 'ibrun -n 16 -o %s /work/02977/jialiu/IG_Pipeline_0.1/N-GenIC/N-GenIC /work/02977/jialiu/lenstools_home/Om0.300_Ol0.700/512b240/ic%s/ngenic.param  &\n'%(j*16, i+j)
			f.write(newline)
		f.write('wait\n')	
	f.close()

#write_ngenic_submission()

##################################################
############## camb ##############################
##################################################
def write_camb_CMB91_submission():
	fn = '/work/02977/jialiu/CMB_batch/Jobs/camb.sh'
	f = open(fn, 'w')
	content ='''#!/bin/bash

################################
######Allocation ID#############
################################

#SBATCH -A TG-AST140041


##########################################
#############Directives###################
##########################################

#SBATCH -J CAMB

#SBATCH -o /work/02977/jialiu/CMB_batch/Logs/camb%j.err
#SBATCH -e /work/02977/jialiu/CMB_batch/Logs/camb%j.err


#SBATCH -p normal
#SBATCH -t 10:00:00

#SBATCH --mail-user=jia@astro.columbia.edu
#SBATCH --mail-type=all


##########################################
#############Resources####################
##########################################

#SBATCH -n 91
#SBATCH -N 91

###################################################
#################Execution#########################
###################################################

cd /work/02977/jialiu/IG_Pipeline/camb

'''
	f.write(content)
	f.close()
	f = open(fn,'a')
	j = 0
	for icosmo in glob.glob('/home1/02977/jialiu/work/CMB_batch/O*'):
		newline = 'ibrun -n 1 -o %s /work/02977/jialiu/IG_Pipeline_0.1/camb/camb %s/1024b600/camb.param &\n'%(j, icosmo)
		print newline
		j+=1
		f.write(newline)
		
	f.write('wait\n')
	f.close()

write_camb_CMB91_submission()

###########################################
######## camb development #################
###########################################
cosmo_arr = glob.glob('/home1/02977/jialiu/work/CMB_batch/O*')
def write_camb_CMB91dev_submission(n_cosmo):
	'''use dev nodes, run 16x2 cosmos each time, n_cosmo = one of [0,  32,  64 ]
	'''
	fn = '/work/02977/jialiu/CMB_batch/Jobs/camb_dev_%s-%s.sh'%(n_cosmo,amin(n_cosmo+31,90))	
	f = open(fn, 'w')
	content ='''#!/bin/bash

################################
######Allocation ID#############
################################

#SBATCH -A TG-AST140041


##########################################
#############Directives###################
##########################################

#SBATCH -J CAMB

#SBATCH -o /work/02977/jialiu/CMB_batch/Logs/camb%j.err
#SBATCH -e /work/02977/jialiu/CMB_batch/Logs/camb%j.err


#SBATCH -p development
#SBATCH -t 2:00:00

#SBATCH --mail-user=jia@astro.columbia.edu
#SBATCH --mail-type=all


##########################################
#############Resources####################
##########################################

#SBATCH -n 16
#SBATCH -N 16

###################################################
#################Execution#########################
###################################################

cd /work/02977/jialiu/IG_Pipeline/camb

'''
	f.write(content)
	f.close()
	f = open(fn,'a')
	j = 0
	for j in range(n_cosmo, n_cosmo+16):
		newline = 'ibrun -n 1 -o %s /work/02977/jialiu/IG_Pipeline_0.1/camb/camb %s/1024b600/camb.param &\n'%(j, cosmo_arr[n_cosmo])
		print newline
		j+=1
		f.write(newline)	
	f.write('wait\n')
	
	j = 0
	for j in range(n_cosmo+16, min(n_cosmo+32, 91)):
		newline = 'ibrun -n 1 -o %s /work/02977/jialiu/IG_Pipeline_0.1/camb/camb %s/1024b600/camb.param &\n'%(j, cosmo_arr[n_cosmo])
		#print newline
		j+=1
		f.write(newline)	
	f.write('wait\n')	
	f.close()
map(write_camb_CMB91dev_submission, (0,  32,  64))