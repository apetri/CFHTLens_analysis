#include <math.h>
#include "sampler.h"

/*This function computes the p-distance between 2 points in D dimensions:
the exponent p is tunable*/

double distance(double *x,double *y,int D,float p){

	double dist = 0.0;
	int i;

	for(i=0;i<D;i++){
		dist += pow(fabs(x[i]-y[i]),p);
	}

	return pow(dist,1.0/p);

}

/*This is the cost function of the problem: it is a sum of all the reciprocal of 
the pairs distances, with some tunable exponent lambda; note that the final
1/lambda exponentiation is not performed here!*/

double cost(double **data,int Npoints,int D,float p,float lambda){

	double sum = 0.0;
	int i,j;

	for(i=0;i<Npoints;i++){
		for(j=i+1;j<Npoints;j++){

			//Add the contribution of pair (i,j) to the cost function
			sum += pow(pow(D,1.0/p)/distance(data[i],data[j],D,p),lambda); 

		}
	}

	return (2.0/(Npoints*(Npoints-1)))*sum;

}

/*This computes the cost function in the particular case in which all the points are 
equally spaced on the diagonal of the hypercube*/

double diagonalCost(int Npoints,float lambda){

	double sum = 0.0;
	int i,j;

	for(i=0;i<Npoints;i++){
		for(j=i+1;j<Npoints;j++){

			//Add the contribution of pair (i,j) to the cost function
			sum += pow((Npoints-1)*1.0/(j-i),lambda); 

		}
	}

	return (2.0/(Npoints*(Npoints-1)))*sum;

}

/*This function computes the variation of the cost when a pair of coordinates is exchanged;
this function also performs an in-place swap of the coordinates. More specifically:
this function swaps coordinate d of points i1 and i2 and returns the variation of the cost
function due to this swap*/

double swap(double **data,int Npoints,int D,float p,float lambda,int i1,int i2, int d){

	double costBefore,costAfter,temp;
	int i;

	//initialize to 0
	costBefore = costAfter = 0.0;

	/*compute the contribution of points i1 and i2 to the cost function, before swapping;
	sum over all the particles except i1 and i2*/
	for(i=0;i<Npoints;i++){
		
		if(i!=i1 && i!=i2){
			costBefore += pow(pow(D,1.0/p)/distance(data[i],data[i1],D,p),lambda) + pow(pow(D,1.0/p)/distance(data[i],data[i2],D,p),lambda);
		}
	
	}

	//perform the coordinate swap
	temp = data[i1][d];
	data[i1][d] = data[i2][d];
	data[i2][d] = temp;

	/*compute the contribution of points i1 and i2 to the cost function, after swapping;
	sum over all the particles except i1 and i2*/
	for(i=0;i<Npoints;i++){
		
		if(i!=i1 && i!=i2){
			costAfter += pow(pow(D,1.0/p)/distance(data[i],data[i1],D,p),lambda) + pow(pow(D,1.0/p)/distance(data[i],data[i2],D,p),lambda);
		}
	
	}

	//return the cost difference
	return (2.0/(Npoints*(Npoints-1)))*(costAfter - costBefore);

}

/*This function swaps the particle pair back: needed if the original swap didn't improve the cost function*/

void swapBack(double **data,int i1,int i2,int d){

	double temp;

	temp = data[i1][d];
	data[i1][d] = data[i2][d];
	data[i2][d] = temp;

}