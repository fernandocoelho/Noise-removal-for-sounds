#ifndef _WAVEIO_
    #define _WAVEIO_
#include <stdio.h>
#include <stdlib.h>
#define HEADER_LENGTH 44
#define DEBUG

typedef struct {
    char riff[4];
    int file_length;
    char wave[4];
    char fmt[4] ;
    int fmt_length;
    short format;
    short channels;
    int sample_rate;
    int bytes_per_second;
    short block_align;
    short bits_sample;
    char data_init[4];
    int data_length;
    short* data;
} wave_t;

wave_t* wave_read(const char* file_name);
void wave_write(const char* file_name, wave_t* wave);
void wave_print(wave_t* wave);

void print_chars(const char* title, const char* chars, int length);

#endif
