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
void realization_id(int num_realization,char *id);

//main
int main(int argc,char **argv){

	//Initialize MPI environment
	int numtasks,taskid,first_in_task,last_in_task,i,j;
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

	//Throw warning about map name format
	if(taskid==MASTER){
		fprintf(stderr,"\nWarning: the realization label in the map name must be of type xxxxr, e.g. 0001r for realization 1!!\n\n");
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

	//Save all the thresholds in a fits file (same for all realizations)
	if(taskid==MASTER){
		
		char outname_thresholds[512];
		int outdim_thresholds[1];
		outdim_thresholds[0] = num_mf_bins+1;
		
		sprintf(outname_thresholds,"%s/%s.fit",options.output_path,options.output_threshold_root);
		fprintf(stderr,"Saving MF thresholds to %s\n",outname_thresholds);
		save_array_fits(outname_thresholds,MF_thresholds,1,outdim_thresholds);

	}

	//Print information about current analysis
	if(taskid==MASTER){
		fprintf(stderr,"\nAnalyzing %d maps, divided between %d tasks\n\n",options.num_realizations,numtasks);
		fprintf(stderr, "Lowest threshold=%e, Highest threshold=%e, Number of bins=%d, Bin spacing= %s\n",lowest_threshold,highest_threshold,num_mf_bins,bin_spacing);
	}

	//Set support variables for map and its gradients
	float *map,*gx,*gy,*hxx,*hxy,*hyy;
	int size = options.num_pixels_size;

	map = (float *)malloc(sizeof(float)*size*size);
	gx = (float *)malloc(sizeof(float)*size*size);
	gy = (float *)malloc(sizeof(float)*size*size);
	hxx = (float *)malloc(sizeof(float)*size*size);
	hxy = (float *)malloc(sizeof(float)*size*size);
	hyy = (float *)malloc(sizeof(float)*size*size);

	//Set containers for MFs and moments
	float MFs[num_mf_bins*3];
	float mean,fifth;
	float moments_values[9]; 

	//Output filenames and fits array dimensions
	char map_name[512],realid[5],outname_minkowski[512],outname_moments[512];
	int outdim_minkowski[2],outdim_moments[1];

	outdim_minkowski[0] = num_mf_bins;
	outdim_minkowski[1] = 3;
	outdim_moments[0] = 9;

	//Now decide which maps this task will take care of
	real_in_task(options.num_realizations,numtasks,taskid,&first_in_task,&last_in_task);

	//Cycle over maps in this taks
	for(i=first_in_task;i<=last_in_task;i++){
		
		realization_id(i,realid);
		sprintf(map_name,"%s/%s_%sr_%s",options.map_path,options.name_prefix,realid,options.name_suffix);
		sprintf(outname_minkowski,"%s/%s_%sr.fit",options.output_path,options.output_mf_root,realid);
		sprintf(outname_moments,"%s/%s_%sr.fit",options.output_path,options.output_moments_root,realid);

		/*This part of the code reads in the map from the fits file*/
		get_map(map_name,map,size);

		/*Decide if subtracting the average from the maps*/
		if(options.subtract_average){
			average_subtract(map,size);
		}
		
		/*This part of the code performs the gradient and second derivatives measurements*/
		gradient_xy(map,gx,gy,size);
		hessian(map,hxx,hyy,hxy,size);

		/*Zero out Minkowski functionals and moments containers before measuring*/
		for(j=0;j<num_mf_bins*3;j++){
			MFs[j] = 0.0;
		}

		for(j=0;j<9;j++){
			moments_values[j] = 0.0;
		}

		mean = 0.0;
		fifth = 0.0;

		/*Perform the Minkowski functionals measurements*/
		minkovski_functionals(map,size,1.0,gx,gy,hxx,hyy,hxy,num_mf_bins+1,MF_thresholds,MFs,MFs+num_mf_bins,MFs+2*num_mf_bins);

		/*Perform the moments measurements*/
		moments(map,size,gx,gy,hxx,hyy,hxy,&mean,moments_values,moments_values+2,moments_values+5,&fifth);

		/*Measurements done, output results to fits file*/
		save_array_fits(outname_minkowski,MFs,2,outdim_minkowski);
		save_array_fits(outname_moments,moments_values,1,outdim_moments);

		/*Log progress*/
		if(taskid==MASTER){
			fprintf(stderr,"Approximate progress: %d%%\n",i*100/(options.num_realizations/numtasks));
		}

	}

	//Free memory
	free(map);
	free(gx);
	free(gy);
	free(hxx);
	free(hxy);
	free(hyy);

	//End of line
	MPI_Barrier(MPI_COMM_WORLD);

	if(taskid==MASTER){
		fprintf(stderr,"\nDONE!!\n\n");
	}

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

void realization_id(int num_realization,char *id){

	int idint[4];

	idint[0] = num_realization/1000;
	idint[1] = (num_realization - 1000*idint[0])/100;
	idint[2] = (num_realization - 1000*idint[0] - 100*idint[1])/10;
	idint[3] = (num_realization - 1000*idint[0] - 100*idint[1] - 10*idint[2]);

	if(idint[3]>9){
		fprintf(stderr,"Realization is assumed to be 4 digits!!\n");
		exit(1);
	}

	sprintf(id,"%d%d%d%d",idint[0],idint[1],idint[2],idint[3]);

}