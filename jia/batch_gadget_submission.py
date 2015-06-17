#!python
def write_file(ic):
	f = open('/work/02977/jialiu/lenstools_home/Jobs/gadget_ic{0}.sh'.format(ic), 'w')
	content = '''#!/bin/bash

################################
######Allocation ID#############
################################

#SBATCH -A TG-AST140041


##########################################
#############Directives###################
##########################################

#SBATCH -J Gadget2_ic{0}

#SBATCH -o /work/02977/jialiu/lenstools_home/Logs/gadget_ic{0}_%j.err
#SBATCH -e /work/02977/jialiu/lenstools_home/Logs/gadget_ic{0}_%j.err


#SBATCH -p normal
#SBATCH -t 02:00:00

#SBATCH --mail-user=jia@astro.columbia.edu
#SBATCH --mail-type=all


##########################################
#############Resources####################
##########################################

#SBATCH -n 256
#SBATCH -N 16

###################################################
#################Execution#########################
###################################################

ibrun -n 256 -o 0 /work/02977/jialiu/IG_Pipeline_0.1/Gadget2/Gadget2 1 256 /work/02977/jialiu/lenstools_home/Om0.300_Ol0.700/512b240/ic{0}/gadget2.param'''.format(ic)
	f.write(content)
	f.close()


map(write_file, [2]+range(334,430))