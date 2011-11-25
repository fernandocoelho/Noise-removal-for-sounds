#include "wave.h"

int main(int argc, char const* argv[]) {
    wave_t* sound_cool;
    sound_cool = wave_read(argv[1]);

    int i;
    for (i = 0; i < sound_cool->data_length; i++) {
        printf("%hd ", sound_cool->data[i]);
    }
    printf("\n");
    if (argc == 2) return 0;
    sound_cool = wave_read(argv[2]);

    for (i = 0; i < sound_cool->data_length; i++) {
        printf("%hd ", sound_cool->data[i]);
    }
    return 0;
}
