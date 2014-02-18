#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/*Amara's log linear noise power spectrum*/
/*Supply in order (A,n,l0) as noise parameters*/

float log_linear_power(float l,float *parameters){
	
	float A,n,l0;
	
	A = parameters[0];
	n = parameters[1];
	l0 = parameters[2];
	
	if(l==0.0){
		
		return 0.0;
	}
	else{
		
		return fabs(A*(n*log10(l/l0)+1.0)/(l*(l+1.0)));
	
	}
	
}

/*White noise for check*/

float white_power(float l,float *parameters){
	
	if(l==0.0){
		return 0.0;
	}
	else{
		return parameters[0];
	}
	
}

/*exponential noise*/

float exp_power(float l,float *parameters){
	
	float N,sigma;
	
	N=parameters[0];
	sigma=parameters[1];
	
	if(l==0.0){
		return 0.0;
	}
	else{
		return N*exp(-pow(l/sigma,2));
	}
	
}

/*linear interpolation from external file*/

float external_power(float l, float *parameters){
	
	int Nbins=(int)parameters[0],i;
	float result=0.0;
	
	for(i=1;i<Nbins;i++){
		
		if(l>=parameters[i] && l<parameters[i+1]){
			result = parameters[Nbins + i] + ((parameters[Nbins + i + 1]-parameters[Nbins + i])/(parameters[i+1]-parameters[i]))*(l-parameters[i]);
		}
		
	}
	
	return result;
	
}

/*read from file and store in a linear array*/

float* read_template_from_file(char *filename){
	
	FILE *fp;
	int Nbins=0,i;
	float x,y,*parameters;
	
	fp = fopen(filename,"r");
	while(fscanf(fp,"%e %e",&x,&y)!=EOF){
		Nbins++;
	}
	
	fclose(fp);
	
	parameters = malloc(sizeof(float)*(2*Nbins+1));
	parameters[0] = (float)Nbins;
	
	fp = fopen(filename,"r");
	
	for(i=1;i<=Nbins;i++){
		fscanf(fp,"%e %e",&parameters[i],&parameters[Nbins + i]);
	}
	
	return parameters;
	
	
}





