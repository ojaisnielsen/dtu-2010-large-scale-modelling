#ifndef UTIL
#define UTIL

#include <mpi.h>
#include <omp.h>
#include <stdarg.h>
#include <math.h>
#include "common.h"
#include "matrix.h"

void processSays(char *format, ...);
int initRectComm(int verbose);
void getMaxBlockDims(int *maxDims);
void getBlockRange(int *coords, int *row0, int *col0, int *rows, int *cols);
void delayedFree(void *buffer);
void delayedFreeMatrix(Matrix matrix);
void performDelayedFree();
void timerStart();
void timerStop();

#endif