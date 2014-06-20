
void function(double *parameters,int Nparams,double *descriptor,int Nbins){

	int i,j;

	for(j=0;j<Nbins;j++){
		descriptor[j] = 1.0 + j;
		for(i=0;i<Nparams;i++){
			descriptor[j] += parameters[i];
		}
	}

}