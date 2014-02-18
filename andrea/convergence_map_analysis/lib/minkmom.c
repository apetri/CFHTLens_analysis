#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void bin_interval_linear(float *value,float left_lim,float right_lim,int N_values){
	
	int i;
	float step;
	
	step=(right_lim-left_lim)/(N_values-1);
	for(i=0;i<N_values;i++){
		
		value[i]=left_lim + i*step;
	}
	
}

void bin_interval_log(float *value,float left_lim,float right_lim,int N_values){
	
	int i;
	
	for(i=0;i<N_values;i++){
		
		value[i] = left_lim*pow((right_lim/left_lim),(i*1.0)/(N_values-1));
	}
	
}

//Minkovski functional calculations
float mink_1_integrand(float gx,float gy){
	
	return 0.25*sqrt(pow(gx,2) + pow(gy,2));
	
}

float mink_2_integrand(float gx,float gy,float hxx,float hyy,float hxy){
	
	if(pow(gx,2)+pow(gy,2)==0.0){
		printf("WARNING: DIVIDE by 0: taking limit\n");
		printf("hxx=%e hyy=%e hxy=%e\n",hxx,hyy,hxy);
		return (2*hxy - hxx - hyy)/(4.0*M_PI);
	}
	else{
		return ((2*gx*gy*hxy-pow(gx,2)*hyy-pow(gy,2)*hxx)/(pow(gx,2)+pow(gy,2)))/(2.0*M_PI);
	}
}

void minkovski_functionals(float *map,long map_size,float sigma,float *gx,float *gy, float *hxx, float *hyy, float *hxy, int Nvalues, float *values,float *mink_0,float *mink_1,float *mink_2){
	
	int i,Nbins = Nvalues-1;
	long k;
	float integrand1,integrand2;
	
	for(k=0;k<map_size*map_size;k++){
		
		//calculate the minkovski functionals
		
		integrand1=mink_1_integrand(gx[k],gy[k]);
		integrand2=mink_2_integrand(gx[k],gy[k],hxx[k],hyy[k],hxy[k]);
		
		for(i=0;i<Nbins;i++){
			
			if(map[k]>=(values[i]+values[i+1])*sigma/2){
				
				mink_0[i] += 1.0/(map_size*map_size);
				
			}
			
			if(map[k]>=values[i]*sigma && map[k]<values[i+1]*sigma){
				
				mink_1[i] += integrand1/((map_size*map_size)*(values[i+1]-values[i]));
				mink_2[i] += integrand2/((map_size*map_size)*(values[i+1]-values[i]));
			
			}
		}
	
	}
}

void moments(float *map,long map_size, float *gx, float *gy, float *hxx, float *hyy, float *hxy, float *mean,float *variance, float *skeweness, float *kurtosis, float *fifth){
	
	long k;
	
	for(k=0;k<map_size*map_size;k++){
		
		//Do some map statistics here
		
		//Mean
		*mean += map[k]/(map_size*map_size);
		
		//Variance
		variance[0] += pow(map[k],2)/(map_size*map_size);
		variance[1] += (pow(gx[k],2) + pow(gy[k],2))/(map_size*map_size);
		
		//Skeweness
		skeweness[0] += pow(map[k],3)/(map_size*map_size);
		skeweness[1] += (pow(map[k],2)*(hxx[k]+hyy[k]))/(map_size*map_size);
		skeweness[2] += ((pow(gx[k],2) + pow(gy[k],2))*(hxx[k]+hyy[k]))/(map_size*map_size);
		
		//Kurtosis
		kurtosis[0] += pow(map[k],4)/(map_size*map_size);
		kurtosis[1] += (pow(map[k],3)*(hxx[k]+hyy[k]))/(map_size*map_size);
		kurtosis[2] += (map[k]*(pow(gx[k],2) + pow(gy[k],2))*(hxx[k]+hyy[k]))/(map_size*map_size);
		kurtosis[3] += pow((pow(gx[k],2) + pow(gy[k],2)),2)/(map_size*map_size);
		
		//Fifth moment
		*fifth += pow(map[k],5)/(map_size*map_size);
		
	}
	
}



