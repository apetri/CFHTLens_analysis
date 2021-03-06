#!python
############# edite this block ###########
############# maybe also want to edit run time below #####
errfn = 'kappaPDF' # error file name, will append variable automatically
program_name = 'stampede_noiseless_PDF.py' #program name
N = 4 # number of nodes
hrs = 2 # number of hours requested
variable_arr = (0,1)
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
#SBATCH -t {5}:00:00      # Run time (hh:mm:ss)
#SBATCH -A TG-AST140041

ibrun python /home1/02977/jialiu/CFHTLens_analysis/jia/{1} {2}'''.format(errfn, program_name, variable, N, n, hrs)
	f.write(content)
	f.close()

for variable in variable_arr:
	write_file(errfn, program_name, variable)
