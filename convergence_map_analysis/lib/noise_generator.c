#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>

#include "coordinates.h"

#define TINY 1.0e-20

/*Generate gaussian random numbers starting from a uniform distribution*/
float generate_gaussian(float uniform1,float uniform2,float rms){
	
	float gauss;
	
	if(uniform1 > 1.0){
		printf("WARNING NEGATIVE SQRT!!\n");
	}
	
	if(uniform1 == 0.0){
		printf("WARNING ZERO LOG!!\n");
	}
	
	gauss = sqrt(-2.0*log(uniform1)) * cos(2.0*M_PI*uniform2) * rms;
	return gauss;
	
}

/*Given power spectral shape, generate a two dimensional noise map in Fourier space,with random phases*/
void noise_map_fourier(float(*template)(float,float *),float lpix, float *parameters,fftw_complex *map, long map_size,int seed){
	
	long x,y;
	float l,lx,ly,rms,pixel_noise,uniform1,uniform2,phase;
	
	//initialize random seed
	srand(seed);
	//generate noise map pixel by pixel
	for(x=0;x<map_size;x++){
		
		lx = min_long(x,map_size-x)*lpix;
		
		for(y=0;y<map_size/2+1;y++){
			
			ly = y*lpix;
			l = sqrt(lx*lx + ly*ly);
			
			//Noise characteristics rms and phase
			rms = sqrt((*template)(l,parameters));
			uniform1 = rand()/(float)(RAND_MAX);
			if(uniform1==0.0){
				uniform1 = TINY;
			}
			uniform2 = rand()/(float)(RAND_MAX);
			pixel_noise = generate_gaussian(uniform1,uniform2,rms);
			phase = (2.0*M_PI)*rand()/(float)(RAND_MAX);
			
			//Add the noise to the map
			map[fourier_coordinate(x,y,map_size)][0] = (lpix/(2*M_PI)) * pixel_noise * cos(phase);
			map[fourier_coordinate(x,y,map_size)][1] = (lpix/(2*M_PI)) * pixel_noise * sin(phase);
		
			
		}
	}
	
	
}

/*Given power spectral shape, generate a two dimensional noise map in real space*/
void noise_map_real(float(*template)(float,float *),float *parameters,float angpix,long map_size,int seed, double *map){
	
	float lpix=2*M_PI/(map_size*angpix);
	fftw_complex *fourier_map;
	fftw_plan invert;
	
	fourier_map = fftw_malloc(sizeof(fftw_complex)*map_size*(map_size/2 + 1));
	noise_map_fourier(template,lpix,parameters,fourier_map,map_size,seed);
	
	invert = fftw_plan_dft_c2r_2d(map_size,map_size,fourier_map,map,FFTW_ESTIMATE);
	fftw_execute(invert);
	
	fftw_destroy_plan(invert);
	fftw_free(fourier_map);
	
}

/*Given power spectral shape, add a two dimensional noise map to the signal map, in real space*/
void add_noise_to_map(float(*template)(float,float *),float *parameters,float angpix,long map_size,int seed, float *map){
	
	double *noise_map;
	long k;
	
	noise_map = malloc(sizeof(double)*map_size*map_size);
	noise_map_real(template,parameters,angpix,map_size,seed,noise_map);
	
	for(k=0;k<map_size*map_size;k++){
		map[k] += noise_map[k];
	}
	
	free(noise_map);
	
}

/*Add white noise component to a map*/
void add_white_noise_to_map(float *map,long map_size,float rms,int seed){
	
	long k;
	float uniform1,uniform2,noise;
	
	srand(seed);
	
	for(k=0;k<map_size*map_size;k++){
		
		//generate two random numbers in (0,1)
		uniform1 = rand()/(float)RAND_MAX;
		uniform2 = rand()/(float)RAND_MAX;
		//set first to TINY if it is 0.0
		if(uniform1==0.0){
			uniform1=TINY;
		}
		//generate gaussian with rms variance
		noise=generate_gaussian(uniform1,uniform2,rms);
		//add to pixel
		map[k] += noise;
		
	}
	
	
}


