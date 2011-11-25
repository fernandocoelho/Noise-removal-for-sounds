#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>
#include "waveio.h"
#include "wave.cuh"
#include <sys/time.h>

void wave_duplicate(wave_t* wave) {
    int i;

    wave->data = (short*) realloc(wave->data, sizeof(short) * 2 * wave->data_length);

    for (i = 0; i < wave->data_length; i++) {
        wave->data[i + wave->data_length] = wave->data[i];
    }

    wave->file_length += 2 * wave->data_length;
    wave->data_length *= 2;
}

double file_size(int array_size) {
    return((double) (sizeof(short) * array_size)) / (1024.0 * 1024.0);
}

int main(int argc, const char *argv[]) {
    for (int i = 0; i < 8; i++) {
        unsigned long long micros;
        struct timeval t1, t2;

        wave_t *sound;

        /* loads the wave from file */
        sound = wave_read(argv[1]);
        printf("Cuda:\n");

        for (int j = 0; j < i; j++)
            wave_duplicate(sound);

        short *d_data, *d_buffer;
        size_t size = sizeof(short) * sound->data_length;

        cudaMalloc((void**)&d_data, size);
        cudaMalloc((void**)&d_buffer, size);

        gettimeofday(&t1, NULL);
        cudaMemcpy(d_data, sound->data, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_buffer, d_data, size, cudaMemcpyDeviceToDevice);

        int nBlocks = 32768;
        int blockSize = 512;
        /* edit here to change filter */
        wave_filter_mean <<< nBlocks, blockSize >>>(d_data, sound->data_length, d_buffer);
        // wave_filter_median <<< nBlocks, blockSize >>>(d_data, sound->data_length, d_buffer);
        // wave_filter_gaussian <<< nBlocks, blockSize >>>(d_data, sound->data_length, d_buffer);
        // wave_filter_mean <<< nBlocks, blockSize >>>(d_data, sound->data_length, d_buffer);

        cudaMemcpy(sound->data, d_data, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_buffer, d_data, 1, cudaMemcpyDeviceToDevice);
        gettimeofday(&t2, NULL);

        micros = (t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec);
        printf("    %d: %.3f\n", i, micros / 1000.0); //milisegundos
        //printf("    %d: %.3f\n", i, file_size(sound->data_length));

        cudaFree(d_data);
        cudaFree(d_buffer);
    }

    // wave_write(argv[2], sound);

    return 0;
}

