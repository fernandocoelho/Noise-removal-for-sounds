#include "wave.cuh"

/* ------------------------------------------------------------------------- */
__device__ void __half_selection_sort(short *array, int size) {
    int i, j;
    for (i = 0; i < size / 2; i++) {
        int min = i;
        for (j = i; j < size; j++) {
            if (array[j] < array[min])
                min = j;
        }
        int tmp = array[i];
        array[i] = array[min];
        array[min] = tmp;
    }
}

/* ------------------------------------------------------------------------- */
__device__ inline int myabs(int x) {
    return (x > 0) ? x : -x;
}

/* ------------------------------------------------------------------------- */
__global__ void wave_encrypt(short *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        data[idx] = 0;
}

/* ------------------------------------------------------------------------- */
__global__ void wave_filter_median(short *data, int size, short *buffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    int range = 10;

    for (; i < size; i += step) {
        short sorted[21];
        for(int j = -range; j <= range; j++) {
            sorted[j + range] = 0;

            if(((i + (j * 2)) < 0) || ((i + (j * 2)) >= size))
                continue;
            sorted[j + range] = buffer[i + (j * 2)];
        }

        __half_selection_sort(sorted, ((range * 2) + 1));
        data[i] = sorted[range / 2];
    }
}

/* ------------------------------------------------------------------------- */
__global__ void wave_filter_mean(short *data, int size, short *buffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    int range = 10;

    for (; i < size; i += step) {
        int sum = 0;
        for (int j = -range; j <= range; j++) {
            if (((i + (j * 2)) < 0) || ((i + (j * 2)) >= size))
                continue;      
            sum += buffer[i + (j * 2)];
        }

        data[i] = (short) (sum / range);
    }
}

/* ------------------------------------------------------------------------- */
__global__ void wave_filter_gaussian(short *data, int size, short *buffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    int range = 10;
    float gaussian[] = { 0.065756, 0.065088, 0.063126, 0.059986,
        0.055851, 0.050950, 0.045541, 0.039883, 0.034223, 0.028772,
        0.023702 };

    for (; i < size; i += step) {
        int sum = 0;
        for (int j = -range; j <= range; j++) {
            if (((i + (j * 2)) < 0) || ((i + (j * 2)) >= size))
                continue;      
            sum += (short) (gaussian[myabs(j)] * (float) buffer[i + (j * 2)]);
        }

        data[i] = sum;
    }
}
