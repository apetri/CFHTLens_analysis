N = 4
n = 16*4
def submission(i):
	f=open('/home1/02977/jialiu/batch/KS_subfield%i_N%s_n%s'%(i, N, n),'w')
	
	string = '''#!/bin/bash
	#SBATCH -J KS_sf%i     # Job name
	#SBATCH -o KS_sf%i.o%j # Name of stdout output file(%j expands to jobId)
	#SBATCH -e KS_sf%i.e%j # Name of stderr output file(%j expands to jobId)
	#SBATCH -p normal          # Submit to the 'normal' or 'development' queue
	#SBATCH -N %i                   # Total number of nodes requested (16 cores/node)
	#SBATCH -n %i                  # Total number of mpi tasks requested
	#SBATCH -t 2:00:00             # Run time (hh:mm:ss)

	ibrun python-mpi /home1/02977/jialiu/CFHTLens_analysis/jia/stampede_massSIM.py %i
	'''%(i, N, n, i)

	f.write(string)
	f.close()
