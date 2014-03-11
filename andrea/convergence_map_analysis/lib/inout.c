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

//Save a 1D float array into a square 2D fits map
void save_map(char *map_file_name,float *map,long map_size){
	
	fitsfile *map_file;
	int status,naxis;
	long naxes[2],fpixel[2];
	
	status=0;
	naxis=2;
	naxes[0]=map_size;
	naxes[1]=map_size;
	fpixel[0]=1;
	fpixel[1]=1;
	
	fits_create_file(&map_file,map_file_name,&status);
	fits_create_img(map_file,FLOAT_IMG,naxis,naxes,&status);
	fits_write_pix(map_file,TFLOAT,fpixel,map_size*map_size,map,&status);
	fits_close_file(map_file,&status);
	
}

void save_array_fits(char *filename,float *array,int numaxes,int *dim){

	fitsfile *outfile;
	int i,status,totalelements;
	long naxes[numaxes],fpixel[numaxes];

	status=0;
	totalelements=1;
	for(i=0;i<numaxes;i++){
		naxes[i] = dim[i];
		fpixel[i] = 1;
		totalelements*=dim[i];
	}

	fits_create_file(&outfile,filename,&status);
	fits_create_img(outfile,FLOAT_IMG,numaxes,naxes,&status);
	fits_write_pix(outfile,TFLOAT,fpixel,totalelements,array,&status);
	fits_close_file(outfile,&status);

}