#ifndef __SMOOTH_HPP
#define __SMOOTH_HPP

#ifdef __SMOOTH_IS_CPP

extern "C" float smoothing_kernel(int,int,long,float);
extern "C" void smooth_map_gaussian(float *,long,float);
extern "C" void smooth_map_bilateral(float *,long,double,double);

#else

float smoothing_kernel(int,int,long,float);
void smooth_map_gaussian(float *,long,float);
void smooth_map_bilateral(float *,long,double,double);

#endif

#endif