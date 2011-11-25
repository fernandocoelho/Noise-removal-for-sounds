#ifndef _WAVEHU_
	#define _WAVEHU_

__global__ void wave_encrypt(short* data, size_t size);
__global__ void wave_filter_median(short *data, int size, short *buffer);
__global__ void wave_filter_gaussian(short *data, int size, short *buffer);
__global__ void wave_filter_mean(short *data, int size, short *buffer);
#endif
