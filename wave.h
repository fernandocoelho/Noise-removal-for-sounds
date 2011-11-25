#ifndef _WAVE_
    #define _WAVE_
#include <math.h>
#include "waveio.h"

void wave_duplicate(wave_t* wave);
void wave_encrypt(wave_t* wave);
void wave_reduce(wave_t* wave);
void wave_enlarge(wave_t* wave);
void wave_trim(wave_t* wave, int start, int end);
void wave_no_silence(wave_t* wave);
void wave_reverse(wave_t* wave);
void wave_echo(wave_t* wave);
int wave_is_equal(wave_t* wave1, wave_t* wave2);
void wave_free(wave_t *wave);
void wave_reload(wave_t *wave, const char* file_name);

void wave_filter_highpass(wave_t *wave, double dt, double rc);
void wave_filter_mean(wave_t* wave, int range);
void wave_filter_median(wave_t* wave, int range);
void wave_filter_gaussian(wave_t* wave, int range);

void quick_sort(short *array, int begin, int end);
void merge_sort(short *array, int size);

#endif

