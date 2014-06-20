#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_permutation.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include "ini.h"
#include "options.h"
#include "sampler.h"

int save_points(char *filename,double **data,int Npoints,int D);

int main(int argc, char **argv){

	//Check for correct number of arguments
	if(argc<2){
		fprintf(stderr,"Usage: %s <ini_options_file> <random_seed(optional)>\n",*argv);
		exit(1);
	}

	sys_options options;

	//parse options from INI options file provided from command line
	if(ini_parse(argv[1],handler,&options)<0){
		fprintf(stderr,"Couldn't read %s, quitting...\n",argv[1]);
		exit(1);
	}

	//options are parsed, fill in the parameters

	int D = options.dimensions, Npoints = options.number_of_points;
	float p = options.p, lambda = options.lambda;

	//Log parsed options
	fprintf(stdout,"%d points in %d dimensions\n",Npoints,D);
	fprintf(stdout,"p=%.1f, lambda=%.1f\n",p,lambda);
	
	int i,d,i1,i2,iterCount;
	
	const gsl_rng_type *T;
	gsl_rng *r;
	gsl_permutation *perm = gsl_permutation_alloc(Npoints);

	double **data,currentCost,deltaCost;

	FILE *costFile;

	//Allocate resources for data points
	if((data = (double **)malloc(sizeof(double *)*Npoints)) == NULL){
		perror("malloc() failed");
		exit(1);
	}

	for(i=0;i<Npoints;i++){
		if((data[i] = (double *)malloc(sizeof(double)*D)) == NULL){
			perror("malloc() failed");
			exit(1);
		}
	}

	//Initialize random number generator
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);

	//Initialize permutation
	gsl_permutation_init(perm);

	//Initialize coordinates
	for(i=0;i<Npoints;i++){
		for(d=0;d<D;d++){
			data[i][d] = ((double)perm->data[i])/(Npoints-1);
		}
	}

	//Save the diagonal configuration to file if prompted by options
	if(options.save_initial_step){

		if(save_points(options.diagonal_file,data,Npoints,D)!=0){
			perror("save_points() failed");
			exit(1);
		}
	
	}

	//Set random seed: give the user the option to override random seed from command line
	if(argc>2) options.seed = atoi(argv[2]);
	gsl_rng_set(r,options.seed);

	//Log cost function value for diagonal latin hypercube
	fprintf(stdout,"Diagonal latin hypercube cost function: %.3lf\n",diagonalCost(Npoints,lambda));

	//Initialize the point coordinates in data with random permutations of (1..Npoints) to enforce latin hypercube structure
	for(d=0;d<D;d++){

		//Shuffle the numbers
		gsl_ran_shuffle(r,perm->data,Npoints,sizeof(size_t));

		//Permute coordinates
		for(i=0;i<Npoints;i++){
			data[i][d] = ((double)perm->data[i])/(Npoints-1);
		}

	}

	//Save the result of the first shuffle to file if prompted by options
	if(options.save_initial_step){

		if(save_points(options.first_shuffle_file,data,Npoints,D)!=0){
			perror("save_points() failed");
			exit(1);
		}

	}

	/*The loop does the following: it swaps a random coordinate of a random pair,
	checks if the cost is lower. If so, it keeps the configuration, otherwise it
	reverses it and tries a new one.*/

	iterCount = 0;
	currentCost = cost(data,Npoints,D,p,lambda);

	//Log initial value of cost function
	fprintf(stdout,"Initial value of cost function: %.3lf\n",currentCost);

	//Open file for saving cost function values
	if(options.save_cost){
		costFile = fopen(options.cost_file,"w");
		if(costFile==NULL){
			perror("fopen() cost file failed");
			exit(1);
		}
	}


	while(1){

		//Decide which coordinate to swap of which pair

		i1 = gsl_rng_uniform_int(r,Npoints);
		while((i2=gsl_rng_uniform_int(r,Npoints))==i1);
		d = gsl_rng_uniform_int(r,D);

		//Compute the change in the cost function
		deltaCost = swap(data,Npoints,D,p,lambda,i1,i2,d);

		/*Check if gain in cost is positive or negative: if positive, revert the swap, if negative keep it;
		anyway, log the result*/
		if(deltaCost>=0){
			swapBack(data,i1,i2,d);
		} else{
			currentCost += deltaCost;
		}

		//Log the result
		if(options.save_cost){
			fprintf(costFile,"%le\n",currentCost);
		}

		//Criterion to break the loop
		if(++iterCount == 100000) break;
	
	}

	//Close cost file
	if(options.save_cost){
		fclose(costFile);
	}

	//Save coordinates to external file
	if(save_points(options.points_file,data,Npoints,D)!=0){
		perror("save_points() failed");
		exit(1);
	}

	//Release resources for data points
	for(i=0;i<Npoints;i++){
		free(data[i]);
	}

	free(data);

	//Release resources for random number generator and permutations
	gsl_rng_free(r);
	gsl_permutation_free(perm);

	//Log final value of cost function
	fprintf(stdout,"Final value of cost function: %.3lf\n",currentCost);

	return 0;
}

/*This function saves the computed points to an external file in 
text format*/

int save_points(char *filename,double **data,int Npoints,int D){

	FILE *pointsFile;
	int i,d;

	if((pointsFile=fopen(filename,"w")) == NULL){
		return 1;
	}

	for(i=0;i<Npoints;i++){
		for(d=0;d<D;d++){
			fprintf(pointsFile,"%.3lf ",data[i][d]);
		}
		fprintf(pointsFile,"\n");
	}

	fclose(pointsFile);

	return 0;

}