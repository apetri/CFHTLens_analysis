#ifndef __INOUT_H
#define __INOUT_H

long map_size(char *map_file_name);
void get_map(char *map_file_name,float *map,long map_size);
void get_key_float(char *map_file_name, char *keyword, float *value);

#endif