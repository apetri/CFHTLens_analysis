#ifndef __SYSTEMATICS_H
#define __SYSTEMATICS_H

#include <fftw3.h>

/*prototypes*/

//smooth
float smoothing_kernel(int,int,long,float);
void smooth_map_gaussian(float *,long,float);

//binning
void bin_interval_linear(float *,float,float,int);
void bin_interval_log(float *,float,float,int);

//power spectrum
void power_spectrum(float *,long,float,int,float *,double *);

//minowski functionals
void minkovski_functionals(float *,long,float,float *,float *, float *, float *, float *,int, float *,float *,float *,float *);

//moments
void moments(float *,long,float *,float *,float *,float *,float *,float *,float *,float *, float *, float *);

//peaks
void peak_count(float *,long,float,int,float *,float *);

//noise
float generate_gaussian(float,float,float);
void noise_map_fourier(float(*)(float,float *),float,float *,fftw_complex *,long,int);
void noise_map_real(float(*)(float,float *),float *,float,long,int,double *);
void add_noise_to_map(float(*)(float,float *),float *,float,long,int,float *);
void add_white_noise_to_map(float *,long,float,int);

//noise templates
float log_linear_power(float,float *);
float white_power(float,float *);
float exp_power(float,float *);
float external_power(float,float *);
float *read_template_from_file(char *);

#endif