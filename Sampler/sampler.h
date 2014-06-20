#ifndef __SAMPLER_H
#define __SAMPLER_H

//function prototypes
double distance(double *x,double *y,int D,float p);
double cost(double **data,int Npoints,int D,float p,float lambda);
double diagonalCost(int Npoints,float lambda);
double swap(double **data,int Npoints,int D,float p,float lambda,int i1,int i2, int d);
void swapBack(double **data,int i1,int i2,int d);

#endif