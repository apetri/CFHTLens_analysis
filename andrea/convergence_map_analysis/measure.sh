#!/bin/bash

#SBATCH -A TG-AST140041

#SBATCH -J measureFeatures
#SBATCH -o measureFeatures.out
#SBATCH -e measureFeatures.err

#SBATCH -n 112
#SBATCH -p development
#SBATCH -t 02:00:00

#SBATCH --mail-user=apetri@phys.columbia.edu
#SBATCH --mail-type=all

ibrun -n 112 -o 0 python /home1/02918/apetri/CFHTLens_analysis/andrea/convergence_map_analysis/measure_features.py -f /home1/02918/apetri/CFHTLens_analysis/andrea/convergence_map_analysis/options_stampede.ini
