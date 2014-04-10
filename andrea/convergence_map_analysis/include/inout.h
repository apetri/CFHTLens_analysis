#ifndef __INOUT_H
#define __INOUT_H

long map_size(char *map_file_name);
void get_map(char *map_file_name,float *map,long map_size);
int get_key_float(char *map_file_name, char *keyword, float *value);
void save_map(char *map_file_name,float *map,long map_size);
void save_array_fits(char *,float *,int,int *);

#endif