#include <stdlib.h>
#include <math.h>
#include <fftw3.h>

extern "C"{
#include "coordinates.h"
#include "systematics.h"
}

#include "bilateral.hpp"

float smoothing_kernel(int i,int j,long map_size,float pix_filter_size){
	
	float kx,ky,ker;
	
	kx=(float)min_int(i,map_size - i)/map_size;
	ky=(float)min_int(j,map_size - j)/map_size;
	
	ker=exp(-0.5*pow(pix_filter_size,2)*pow(2.0*M_PI,2)*(pow(kx,2)+pow(ky,2)));
	
	return ker;
	
}

//Gaussian map smoothing using fftw
void smooth_map_gaussian(float *map,long map_size,float pix_filter_size){
	
	long i,j;
	long k;
	float kernel;
	fftw_complex *in,*out;
	fftw_plan plan_forward,plan_backward;
	
	in=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*map_size*map_size);
	
	//initialize input
	for(k=0;k<map_size*map_size;k++){
		in[k][0]=map[k];
		in[k][1]=0.0;
	}
	
	out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*map_size*map_size);
	
	//fourier transform the map
	
	plan_forward = fftw_plan_dft_2d((int)map_size,(int)map_size,in,out,FFTW_FORWARD,FFTW_ESTIMATE);
	fftw_execute(plan_forward);
	fftw_destroy_plan(plan_forward);
	
	//multiply by gaussian smoothing kernel
  	
	for(i=0;i<map_size;i++){
		for(j=0;j<map_size;j++){
			
			k = coordinate(i,j,map_size);
			kernel = smoothing_kernel(i,j,map_size,pix_filter_size);
			out[k][0]=out[k][0]*kernel;
			out[k][1]=out[k][1]*kernel;
			
		}
	}
	
	//Now invert the fourier transform to get the smoothed map
	
	plan_backward = fftw_plan_dft_2d((int)map_size,(int)map_size,out,in,FFTW_BACKWARD,FFTW_ESTIMATE);
	fftw_execute(plan_backward);
	fftw_destroy_plan(plan_backward);
	
	//Replace the old map with the smoothed one
	
	for(k=0;k<map_size*map_size;k++){
		map[k]=in[k][0]/(float)(map_size*map_size);
	}
	
	fftw_free(in);
	fftw_free(out);
	
	
}

//Bilateral smoothing using opencv
void smooth_map_bilateral(float *map,long map_size,float pix_filter_size,float sigma_color){

	map[0] = map[0];
	map_size++;
	pix_filter_size++;
	sigma_color++;

}