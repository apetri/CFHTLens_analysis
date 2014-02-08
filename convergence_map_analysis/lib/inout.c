#include <fitsio.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "inout.h"

//Get the size of a 2D square map
long map_size(char *map_file_name){
	
	fitsfile *map_file;
	int status;
	long naxes[2];
	
	status=0;
	fits_open_file(&map_file,map_file_name,0,&status);
	fits_get_img_size(map_file,2,naxes,&status);
	fits_close_file(map_file,&status);
	
	return naxes[0];
	
}

//Save a 2D fits map into an allocatable 1D array of length size*size
void get_map(char *map_file_name,float *map,long map_size){
	
	fitsfile *map_file;
	int status,anynul;
	long fpixel[2];
	
	status=0;
	fpixel[0]=1;
	fpixel[1]=1;
	
	fits_open_file(&map_file,map_file_name,0,&status);
	fits_read_pix(map_file,TFLOAT,fpixel,map_size*map_size,0,map,&anynul,&status);
	fits_close_file(map_file,&status);
	
}

//Get the value of a header float keyword
void get_key_float(char *map_file_name, char *keyword, float *value){
	
	fitsfile *map_file;
	int status;
	
	status=0;
	
	fits_open_file(&map_file,map_file_name,0,&status);
	fits_read_key(map_file,TFLOAT,keyword,value,NULL,&status);
	fits_close_file(map_file,&status);
	
}