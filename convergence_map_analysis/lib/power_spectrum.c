#include <stdlib.h>
#include <math.h>
#include <fftw3.h>

#include "coordinates.h"

/*Compute a 2d map power spectrum using FFTW*/

void power_spectrum(float *map,long size,float map_angle,int Nvalues,float *lvalues,double *power){
	
	fftw_plan transform;
	fftw_complex *fourier_map;
	float lpix,lx,ly,l;
	double *map_double;
	long i,j,*hits;
	int Nbins=Nvalues-1,k;
	
	//define l resolution
	lpix = 360.0/map_angle;
	
	//allocate memory
	fourier_map = fftw_malloc(sizeof(fftw_complex)*size*(size/2+1));
	hits = malloc(sizeof(long)*Nbins);
	for(i=0;i<Nbins;i++){
		hits[i]=0;
	}
	map_double = malloc(sizeof(double)*size*size);
	for(i=0;i<size*size;i++){
		map_double[i] = map[i] * (map_angle*M_PI/180)/(size*size);
	}
	
	//define FFTW plan
	transform = fftw_plan_dft_r2c_2d(size,size,map_double,fourier_map,FFTW_ESTIMATE);
	
	//execute fourier transform
	fftw_execute(transform);
	fftw_destroy_plan(transform);
	
	//compute and bin power spectrum
	for(i=0;i<size;i++){
		
		lx = min_long(i,size-i) * lpix;
		
		for(j=0;j<size/2+1;j++){
			
			ly = j*lpix;
			l=sqrt(lx*lx + ly*ly);
			
			//decide in which l bin this pixel falls into
			for(k=0;k<Nbins;k++){
				
				if(l>lvalues[k] && l<=lvalues[k+1]){
					power[k] += pow(fourier_map[fourier_coordinate(i,j,size)][0],2) + pow(fourier_map[fourier_coordinate(i,j,size)][1],2);
					hits[k]++;
				}
				
			}
			
		}
	}
	
	for(k=0;k<Nbins;k++){
		if(hits[k]>0){
			power[k] = power[k]/hits[k];
		}
	}
	
	//free memory
	free(map_double);
	fftw_free(fourier_map);
	free(hits);
	
	
}