#include "wave.h"

/* ------------------------------------------------------------------------- */
void wave_encrypt(wave_t* wave) {
    int i;

    #pragma omp parallel for private(i)
    for (i = 0; i < wave->data_length; i++) {
        if (wave->data[i] > 0) wave->data[i] /= 10;
        else wave->data[i] >> 2;
    }
}

/* ------------------------------------------------------------------------- */
void wave_duplicate(wave_t* wave) {
    int i;

    wave->data = (short*) realloc(wave->data, sizeof(short) * 2 * wave->data_length);

    #pragma omp parallel for private(i)
    for (i = 0; i < wave->data_length; i++) {
        wave->data[i + wave->data_length] = wave->data[i];
    }

    wave->file_length += 2 * wave->data_length;
    wave->data_length *= 2;
}

/* ------------------------------------------------------------------------- */
void wave_reduce(wave_t* wave) {
    int i;
    short buffer[wave->data_length / 2];

    #pragma omp parallel private(i)
    {
        #pragma omp for
        for (i = 0; i < wave->data_length / 2; i++) {
            buffer[i] = wave->data[2 * i];
        }

        #pragma omp single
        {
            wave->data_length /= 2;
            wave->file_length -= 2 * wave->data_length;
            wave->data = (short*) realloc(wave->data, sizeof(short) * wave->data_length);
        }

        #pragma omp for
        for (i = 0; i < wave->data_length; i++) {
            wave->data[i] = buffer[i];
        }
    }
}

/* ------------------------------------------------------------------------- */
void wave_filter_mean(wave_t* wave, int range) {
    if (!wave || range <= 0) return;

    short *src = (short *) malloc(wave->data_length * sizeof(short));

    #pragma omp parallel
    {
        int i;
        #pragma omp for schedule(guided)
        for(i = 0; i < wave->data_length; i++) {
            src[i] = wave->data[i];
        }

        #pragma omp for
        for(i = 0; i < wave->data_length; i++) {
            int sum = 0;
            int j;

            for(j = -range; j <= range; j++) {
                if(((i + (j * 2)) < 0) || ((i + (j * 2)) >= wave->data_length))
                    continue;
                sum += src[i + (j * 2)];
            }
            wave->data[i] = (short) (sum / range);
        }
    }

    free(src);
}

/* ------------------------------------------------------------------------- */
void __half_selection_sort(short *array, int size) {
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
void wave_filter_median(wave_t* wave, int range) {
    if (!wave || range <= 0) return;

    short *src = (short *) malloc(wave->data_length * sizeof(short));

    #pragma omp parallel
    {
        int i, j;
        #pragma omp for schedule(guided)
        for(i = 0; i < wave->data_length; i++) {
            src[i] = wave->data[i];
        }

        #pragma omp for
        for(i = 0; i < wave->data_length; i++) {
            short sorted[(range * 2) + 1];
            for(j = -range; j <= range; j++) {
                sorted[j + range] = 0;
            }

            for(j = -range; j <= range; j++) {
                if(((i + (j * 2)) < 0) || ((i + (j * 2)) >= wave->data_length))
                    continue;
                sorted[j + range] = src[i + (j * 2)];
            }

            __half_selection_sort(sorted, ((range * 2) + 1));
            wave->data[i] = sorted[range / 2];
        }
    }

    free(src);
}

/* ------------------------------------------------------------------------- */
void wave_filter_gaussian(wave_t* wave, int range) {
    if (!wave || range <= 0) return;

    double sum;

    short *src = (short *) malloc (wave->data_length * sizeof(short));
    double *gaussian = (double *) malloc((range* 2 + 1) * sizeof(double));
    int alpha = alpha = 0.7 * range;


    #pragma omp parallel shared(gaussian, alpha)
    {
        int i;
        #pragma omp for schedule(guided)
        for (i = -range; i <= range; i++) {
            gaussian[i + range] = 1 / (alpha * sqrt(2 * M_PI)) * 
                pow(M_E, -(i * i / (2 * alpha * alpha)));
        }

        #pragma omp for reduction(+:sum)
        for(i = -range; i <= range; i++) {
            sum += gaussian[i + range];
        }

        #pragma omp for schedule(guided)
        for(i = -range; i <= range; i++) {
            gaussian[i + range] /= sum;
        }

        #pragma omp for schedule(guided)
        for(i = 0; i < wave->data_length; i++) {
            src[i] = wave->data[i];
        }

        #pragma omp for
        for(i = 0; i < wave->data_length; i++) {
            int sum = 0;
            int j;
            for(j = -range; j <= range; j++) {
                if(((i + (j * 2)) < 0) || ((i + (j * 2)) >= wave->data_length))
                    continue;
                sum += gaussian[j + range] * (double) src[i + (j * 2)];
            }
            wave->data[i] = (short) sum;
        }
    }

    free(src);
    free(gaussian);
}

/* ------------------------------------------------------------------------- */
void wave_filter_highpass(wave_t *wave, double dt, double rc) {
    if (!wave) return;

    short *buffer = malloc(wave->data_length * sizeof(short));
    double alpha = rc / (rc + dt);

    #pragma omp parallel shared(alpha, buffer)
    {
        int i;
        #pragma omp for
        for (i = 0; i < wave->data_length; i++) {
            buffer[i] = wave->data[i];
        }

        #pragma omp single
        {
            wave->data[0] = buffer[0];
            wave->data[1] = buffer[1];
        }

        #pragma omp for ordered
        for (i = 2; i < wave->data_length; i+= 2) {
            wave->data[i] = alpha * (buffer[i] - buffer[i - 2]);
            wave->data[i + 1] = alpha * (buffer[i + 1] - buffer[i - 1]);

            #pragma omp ordered
            {
                wave->data[i] += alpha * wave->data[i - 2];
                wave->data[i + 1] += alpha * wave->data[i - 1];
            }
        }
    }

    free(buffer);
}

/* ------------------------------------------------------------------------- */
void wave_enlarge(wave_t* wave) {
    int i, j;
    short buffer[wave->data_length * 2];

    #pragma omp parallel private(i, j)
    {
        i = 0;
        #pragma omp for
        for (j = 0; j < wave->data_length; j++) {
            buffer[i] = wave->data[j];
            buffer[i + 1] = wave->data[j];
            i += 2;
        }

        #pragma omp single
        {
            wave->file_length += 2 * wave->data_length;
            wave->data_length *= 2;
            wave->data = (short*) realloc(wave->data, sizeof(short) * wave->data_length);
        }

        #pragma omp for
        for (i = 0; i < wave->data_length; i++) {
            wave->data[i] = buffer[i];
        }
    }
}

/* ------------------------------------------------------------------------- */
void wave_trim(wave_t* wave, int start, int end) {
    int i;

    #pragma omp parallel for private(i)
    for (i = 0; i < wave->data_length - start - end; i++) {
        wave->data[i] = wave->data[i + start];
    }

    wave->file_length -= 2 * (start + end);
    wave->data_length -= (start + end);
    wave->data = (short*) realloc(wave->data, sizeof(short) * wave->data_length);
}

/* ------------------------------------------------------------------------- */
void wave_no_silence(wave_t* wave) {
    int i, j, diff, cut = 32;
    for (i = 0, diff = 0; i < wave->data_length - 1 && diff < cut; i++) {
        if (abs(wave->data[i] - wave->data[i + 1]) > cut * cut) diff++;
        else diff = 0;
    }
    i -= diff;

    for (j = 0, diff = 0; j < wave->data_length - 1 && diff < cut; j++) {
        int in = wave->data_length - j - 1;
        if (abs(wave->data[in] - wave->data[in + 1]) > cut * cut) diff++;
        else diff = 0;
    }
    j -= diff;

    wave_trim(wave, i, j);
}

/* ------------------------------------------------------------------------- */
void wave_reverse(wave_t* wave) {
    int i;
    short buffer[wave->data_length];

    #pragma omp parallel private(i)
    {
        #pragma omp for
        for (i = 0; i < wave->data_length; i++) {
            buffer[i] = wave->data[wave->data_length - i - 1];
        }

        #pragma omp for
        for (i = 0; i < wave->data_length; i++) {
            wave->data[i] = buffer[i];
        }
    }
}

/* ------------------------------------------------------------------------- */
void wave_echo(wave_t* wave) {
    int i, echo = 56234;

    wave->data_length += echo * 3;
    wave->file_length += 2 * echo * 3;
    wave->data = (short*) realloc(wave->data, sizeof(short) * wave->data_length);

    #pragma omp parallel private(i)
    {
        #pragma omp for
        for (i = wave->data_length - echo; i < wave->data_length; i++) {
            wave->data[i] = 0;
        }

        #pragma omp for
        for (i = 0; i < wave->data_length - echo; i += 2) {
            wave->data[i + echo] += abs(wave->data[i]) / 2;
            wave->data[i + 1 + echo] -= abs(wave->data[i + 1]) / 2;
        }
    }
}


/* ------------------------------------------------------------------------- */
    int wave_is_equal(wave_t* wave1, wave_t* wave2) {
        if (wave1->data_length != wave2->data_length)
            return 0;

        int i;
        for (i = 0; i < wave1->data_length; i++) {
            if (wave1->data[i] != wave2->data[i])
                return 0;
        }
        return 1;
    }

/* ------------------------------------------------------------------------- */
void wave_fill(wave_t* wave, int frequency) {
    int i, mult = 1;
    #pragma omp parallel for private(i)
    for (i = 0; i < wave->data_length; i++) {
        wave->data[i] = 23000 * mult;
        if ((i % (wave->sample_rate / (frequency))) == 1) 
            mult *= -1;     
    }
}

/* ------------------------------------------------------------------------- */
void swap(short *a, short *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

/* ------------------------------------------------------------------------- */
void merge(short *array, short size) {
    int mid;
    int i, j, k;
    short *tmp = (short *) malloc(size * sizeof(short));

    mid = size / 2;

    i = 0;
    j = mid;
    k = 0;
    while (i < mid && j < size) {
        if (array[i] < array[j]) {
            tmp[k] = array[i];
            ++i;
        }
        else {
            tmp[k] = array[j];
            ++j;
        }
        ++k;
    }

    if (i == mid) {
        while (j < size) {
            tmp[k] = array[j];
            ++j;
            ++k;
        }
    }
    else {
        while (i < mid) {
            tmp[k] = array[i];
            ++i;
            ++k;
        }
    }

    for (i = 0; i < size; ++i) {
        array[i] = tmp[i];
    }

    free(tmp);
}

/* ------------------------------------------------------------------------- */ 
void merge_sort(short *array, int size) {
    int mid;

    if (size > 1) {
        mid = size / 2;

        merge_sort(array, mid);
        merge_sort(array + mid, size - mid);
        merge(array, size);
    }
}

