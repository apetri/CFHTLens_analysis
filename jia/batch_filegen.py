#!python
############# edite this block ###########
############# maybe also want to edit run time below #####
errfn = 'massSIM' # error file name, will append variable automatically
program_name = 'stampede_massSIM.py' #program name
N = 2 # number of nodes

##################

n = N*16
def write_file(errfn, program_name, variable):
	f = open('/home1/02977/jialiu/batch/batch_{0}_{1}'.format(errfn, variable), 'w')
	content = '''#!/bin/bash
#SBATCH -J {0}{2}    # Job name
#SBATCH -o {0}{2}.e%j # Name of stdout output file
#SBATCH -e {0}{2}.e%j # Name of stderr output file
#SBATCH -p normal       # Submit to the 'normal' or 'development' queue
#SBATCH -N {3}           # Total number of nodes requested (16 cores/node)
#SBATCH -n {4}          # Total number of mpi tasks requested
#SBATCH -t 0:30:00      # Run time (hh:mm:ss)
#SBATCH -A TG-AST140041

ibrun python /home1/02977/jialiu/CFHTLens_analysis/jia/{1} {2}'''.format(errfn, program_name, variable, N, n)
	f.write(content)
	f.close()

for variable in range(1,14):
	write_file(errfn, program_name, variable)
