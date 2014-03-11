#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "ini.h"
#include "options.h"

#include "convergence.h"
#include "inout.h"
#include "differentials.h"

#define MASTER 0

//Function prototypes
void real_in_task(int N_realizations,int Num_tasks, int taskid, int *first, int *last);

//main
int main(int argc,char **argv){

	//Initialize MPI environment
	int numtasks,taskid,first_in_task,last_in_task,i;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);

	//Check for correct number of arguments
	if(argc<2){
		fprintf(stderr,"Usage: %s <ini_options_file>\n",*argv);
		exit(1);
	}

	//Parse options from ini file
	sys_options options;
	if(ini_parse(argv[1],handler,&options)<0){
		fprintf(stderr,"%s not found!\n",argv[1]);
		exit(1);
	}

	//Choose thresholds for Minkowski functionals binning
	int num_mf_bins = options.num_mf_bins;
	float lowest_threshold = options.lowest_threshold;
	float highest_threshold = options.highest_threshold;
	char *bin_spacing = options.bin_spacing;
	float MF_thresholds[num_mf_bins+1];

	//Decide the thresholds to use
	if(strcmp(bin_spacing,"lin")==0){
		bin_interval_linear(MF_thresholds,lowest_threshold,highest_threshold,num_mf_bins+1);
	} else if(strcmp(bin_spacing,"log")==0){
		bin_interval_log(MF_thresholds,lowest_threshold,highest_threshold,num_mf_bins+1);
	} else{
		fprintf(stderr,"Only lin or log bin spacing allowed!\n");
		MPI_Finalize();
		exit(1);
	}

	//Set support variables for gradients
	float *gx,*gy,*hxx,*hxy,*hyy;

	//Set containers for MFs and moments
	float MFs[3][num_mf_bins];
	float moments[9]; 

	//Now decide which maps this task will take care of
	real_in_task(options.num_realizations,numtasks,taskid,&first_in_task,&last_in_task);

	//Finalize MPI
	MPI_Finalize();

	return 0;
}

//Function implementation
void real_in_task(int N_realizations,int Num_tasks, int taskid, int *first, int *last){
	
	int maps_per_slot,leftover_maps,task_break;
	
	maps_per_slot = N_realizations/Num_tasks;
	leftover_maps = N_realizations%Num_tasks;
	task_break = Num_tasks - leftover_maps;
	
	if(taskid+1 <= task_break){
		
		*first = taskid*maps_per_slot + 1;
		*last = *first + maps_per_slot - 1;
	}
	else{
		
		*first = maps_per_slot*task_break + (taskid - task_break)*(maps_per_slot + 1) + 1;
		*last = *first + maps_per_slot;
		
	}
	
	
}