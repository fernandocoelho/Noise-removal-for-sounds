#include "wave.h"
#include<sys/time.h>
#include<omp.h>

int main (int argc, char *argv[]) {
    int option;
    char src[256], dest[256];
    printf("Choose your destiny:\n");
    printf("\t1) Mean filter\n");
    printf("\t2) Median filter\n");
    printf("\t3) Gaussian filter\n");
    printf("\t4) Highpass filter\n");
    printf(" > ");
    scanf("%d", &option);

    if (option < 1|| option > 4) {
        printf("-.-'\n");
        return 1;
    }

    printf("Source wave file: ");
    scanf("%s", src);
    
    wave_t *sound;
    sound = wave_read(src);

    printf("Destination wave file: ");
    scanf("%s", dest);

    switch (option) {
        case 1:
            wave_filter_mean(sound, 10);
            break;
        case 2:
            wave_filter_median(sound, 10);
            break;
        case 3:
            wave_filter_gaussian(sound, 10);
            break;
        case 4:
            wave_filter_highpass(sound, 25, 10);
            break;
    }

    wave_write(dest, sound);

    return 0;
}
