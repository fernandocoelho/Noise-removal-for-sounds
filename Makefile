CC=gcc
CCC=nvcc
FLAGS=-Wall -lm
EXEC=bin/wave
OMPFILES=main.c
IFILES=interactive.c
CFILES=wave.c waveio.c
CUFILES=waveio.cu wave.cu main.cu
CUFLAGS=

all: interactive omp cuda

interactive: ${IFILES} ${CFILES}
	${CC} ${FLAGS} -o ${EXEC}_interactive ${IFILES} ${CFILES} -fopenmp

omp: ${OMPFILES} ${CFILES}
	${CC} ${FLAGS} -o ${EXEC}_omp ${OMPFILES} ${CFILES} -fopenmp

pwave: print_wave.o wave.o spline.o
	${CC} ${FLAGS} -o bin/pwave print_wave.o wave.o 

cuda:
	${CCC} ${CUFLAGS} -o ${EXEC}_cuda ${CUFILES}

clean:
	rm -f *.o *.swp ${EXEC}* bin/pwave
