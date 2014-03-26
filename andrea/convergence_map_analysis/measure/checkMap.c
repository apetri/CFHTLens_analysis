#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "differentials.h"
#include "convergence.h"
#include "inout.h"

#define POW_NBINS 1000
#define PEAK_NBINS 100

int main(int argc,char **argv){

	//check for correct number of arguments
	if(argc<3){
		fprintf(stderr, "Usage: %s <fits_map_file> <output_root>\n",*argv);
		exit(1);
	}

	int i;

	//read in the map
	long mapsize = map_size(argv[1]);
	float *map = (float*)malloc(sizeof(float)*mapsize*mapsize);
	get_map(argv[1],map,mapsize);
	float map_angle = 3.4;
	//get_key_float(argv[1],"ANGLE",&map_angle);
	float sigma = sqrt(variance_map(map,mapsize));

	//set power spectrum and peaks bin edges
	float *power_bin_edges,*peak_bin_edges,*peaks;
	double *power;

	power_bin_edges = (float*)malloc(sizeof(float)*(POW_NBINS+1));
	power = (double*)malloc(sizeof(double)*POW_NBINS);
	peak_bin_edges = (float*)malloc(sizeof(float)*(PEAK_NBINS+1));
	peaks = (float*)malloc(sizeof(float)*PEAK_NBINS);

	//initialize to 0
	for(i=0;i<POW_NBINS;i++){
		power[i] = 0.0;
	}

	for(i=0;i<PEAK_NBINS;i++){
		peaks[i] = 0.0;
	}

	//set bin values
	bin_interval_linear(power_bin_edges,0.0,1.063e+05,POW_NBINS+1);
	bin_interval_linear(peak_bin_edges,-2.0,5.0,PEAK_NBINS+1);

	//measure power spectrum and peaks
	fprintf(stderr,"Measuring power spectrum with FFTW...\n");
	power_spectrum(map,mapsize,map_angle,POW_NBINS+1,power_bin_edges,power);
	fprintf(stderr,"Counting peaks...\n");
	peak_count(map,mapsize,sigma,PEAK_NBINS+1,peak_bin_edges,peaks);

	//set filenames for output
	char power_filename[1024],peak_edge_filename[1024],peak_hist_filename[1024];
	sprintf(power_filename,"%s%s",argv[2],"powspec.txt");
	sprintf(peak_edge_filename,"%s%s",argv[2],"peak_bin_edges.txt");
	sprintf(peak_hist_filename,"%s%s",argv[2],"peak_histogram_snr.txt");

	//output results to files

	FILE *powout = fopen(power_filename,"w");
	if(powout==NULL){
		perror(power_filename);
		exit(1);
	}

	fprintf(stderr,"Writing power spectrum to %s\n",power_filename);

	for(i=0;i<POW_NBINS;i++){
		fprintf(powout,"%e %le\n",power_bin_edges[i+1],power_bin_edges[i+1]*(power_bin_edges[i+1]+1)*power[i]/(2*M_PI));
	}

	fclose(powout);

	FILE *peakout_edge = fopen(peak_edge_filename,"w");
	FILE *peakout_hist = fopen(peak_hist_filename,"w");
	if(peakout_edge==NULL || peakout_hist==NULL){
		perror(argv[2]);
		exit(1);
	}

	fprintf(stderr,"Writing peak edges to %s\n",peak_edge_filename);
	fprintf(stderr,"Writing peak histogram to %s\n",peak_hist_filename);

	for(i=0;i<=PEAK_NBINS;i++){
		fprintf(peakout_edge,"%e\n",peak_bin_edges[i]);
	}
	for(i=0;i<PEAK_NBINS;i++){
		fprintf(peakout_hist,"%e\n",peaks[i]);
	}

	fclose(peakout_edge);
	fclose(peakout_hist);

	//free memory
	free(map);
	free(power_bin_edges);
	free(power);
	free(peak_bin_edges);
	free(peaks);

	fprintf(stderr,"Done!\n");

	return 0;
}