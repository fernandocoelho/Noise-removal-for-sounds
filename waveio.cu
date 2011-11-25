#include "waveio.h"

/* ------------------------------------------------------------------------- */
wave_t* wave_read(const char* file_name) {
    FILE* fp = fopen(file_name, "r");
    wave_t* wave = (wave_t*) malloc(sizeof(wave_t));

    fread(wave, HEADER_LENGTH, 1, fp);
    wave->data_length /= 2;
    wave->data = (short*) malloc(sizeof(short) * wave->data_length);
    fread(wave->data, 2 * wave->data_length, 1, fp);

    fclose(fp);

    return wave;
}

/* ------------------------------------------------------------------------- */
void wave_free(wave_t *wave) {
    if (!wave) return;

    wave->data_length = 0;
    free(wave->data);
    free(wave);
}

/* ------------------------------------------------------------------------- */
void wave_reload(wave_t *wave, const char* file_name) {
    wave_free(wave);
    wave = wave_read(file_name);
}

/* ------------------------------------------------------------------------- */
void wave_write(const char* file_name, wave_t* wave) {
    FILE* fp = fopen(file_name, "w+");

    wave->data_length *= 2;
    fwrite(wave, HEADER_LENGTH, 1, fp);
    fwrite(wave->data, wave->data_length, 1, fp);
    wave->data_length /= 2;
    fclose(fp);
}

/* ------------------------------------------------------------------------- */
void wave_print(wave_t* wave) {
    print_chars("riff", wave->riff, 4);
    printf("file_length: %d\n", wave->file_length);
    print_chars("wave", wave->wave, 4);
    print_chars("fmt", wave->fmt, 4);
    printf("fmt_length: %d\n", wave->fmt_length);
    printf("format: %d\n", wave->format);
    printf("channels: %d\n", wave->channels);
    printf("sample_rate: %d\n", wave->sample_rate);
    printf("bytes_per_second: %d\n", wave->bytes_per_second);
    printf("block_align: %d\n", wave->block_align);
    printf("bits_sample: %d\n", wave->bits_sample);
    print_chars("data_init", wave->data_init, 4);
    printf("data_length: %d\n", 2 * wave->data_length);

#ifdef DEBUG
    int i;
    for (i = 0; i < wave->data_length; i++) {
        if (i % 15 == 0) printf("\n");
        printf("%6hd ", wave->data[i]);
    }
    printf("\n");
#endif

}

/* ------------------------------------------------------------------------- */
void print_chars(const char* title, const char* chars, int length) {
    int i;
    printf("%s: ", title);
    for (i = 0; i < length; i++) {
        printf("%c", chars[i]);
    }
    printf("\n");
}

