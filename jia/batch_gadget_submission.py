#!python
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
#SBATCH -t 10:00:00

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


map(write_gadget_submission, range(5,501)[::5])

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