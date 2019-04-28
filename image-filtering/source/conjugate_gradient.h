#ifndef CONJUGATE_GRADIENT
#define CONJUGATE_GRADIENT

#include <mpi.h>
#include "common.h"
#include "matrix.h"
#include "util.h"

void initConjugateGradient(Matrix invM, Matrix b, Matrix x, Matrix res, Matrix p, Matrix z, float *totalRho);
void finalizeConjugateGradient(Matrix p);
int iterConjugateGradient(Matrix a, Matrix m, Matrix invM, Matrix x, Matrix res, Matrix p, Matrix q, Matrix z, float *totalRho, float eps);
void sendEdges(Matrix matrix);
void receiveEdges(int rows, int cols);
void setNeumannBoundaries(float val, Matrix matrix);


#endif