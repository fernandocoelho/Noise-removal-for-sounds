#include "wave.h"
#include<sys/time.h>
#include<omp.h>

void print_maroto(wave_t *sound, int params, char **argv) {
    unsigned long long micros;
    struct timeval t1,t2;
    int i;

    int media = atoi(argv[3]); 
    for (i = 1; i <= atoi(argv[2]); i++) {
        omp_set_num_threads(i);
        int j;
        micros = 0;
        for (j = 0; j < media; j++) {
            wave_reload(sound, argv[1]);
            gettimeofday(&t1,NULL);	
            if (params == 0)	
                wave_filter_median(sound, 10);
            else if (params == 1)
                wave_filter_gaussian(sound, 10);
            else if (params == 2)
                wave_filter_mean(sound, 10);
            else if (params == 3)
                wave_filter_highpass(sound, 25, 10);
            gettimeofday(&t2,NULL);
            micros += (t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec);
        }
        micros /= media;
        printf("        %d: %.3f\n", i, micros / 1000.0);
    }	
}

int main (int argc, char *argv[]) {
    wave_t *sound;

    if (argc < 5) {
        printf("You should pass at least 4 params:\n");
        printf("\t%s <filename> <num_procs> <num_tests> <machine_name>\n", argv[0]);

        exit(1);
    }

    sound = wave_read(argv[1]);
    printf("%s:\n", argv[4]);

    printf("    Median:\n");
    print_maroto(sound, 0, argv);

    printf("    Gaussian:\n");
    print_maroto(sound, 1, argv);

    printf("    Mean:\n");	
    print_maroto(sound, 2, argv);

    printf("    Highpass:\n");
    print_maroto(sound, 3, argv);

    return 0;
}
