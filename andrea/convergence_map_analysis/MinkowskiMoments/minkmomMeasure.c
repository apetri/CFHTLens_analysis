#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "ini.h"
#include "options.h"

//Function prototypes
void real_in_task(int N_realizations,int Num_tasks, int taskid, int *first, int *last);

//main
int main(int argc,char **argv){

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

	print_options(&options);

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