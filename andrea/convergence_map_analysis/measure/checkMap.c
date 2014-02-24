#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "convergence.h"
#include "inout.h"

int main(int argc,char **argv){

	//check for correct number of arguments
	if(argc<3){
		fprintf(stderr, "Usage: %s <fits_map_file> <output_root>\n",*argv);
		exit(1);
	}


	return 0;
}