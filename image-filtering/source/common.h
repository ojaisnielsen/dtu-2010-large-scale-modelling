#ifndef COMMON
#define COMMON

#include <stdlib.h>
#include <stdio.h>

#define ABS(x) (((x) > 0) ? (x) : (-(x))) // Absolute value macro.
#define MAX(x,y) (((x) > (y)) ? (x) : (y)) // Maximum macro.
#define MIN(x,y) (((x) < (y)) ? (x) : (y)) // minimum macro.

#define K1 10.0f // First linear diffusion parameter.
#define EPS 0.1f // Residual's norm value under which a conjugate gradient algorithm is considered converged.
#define MAX_ITERS 100 // Maximum number of iterations for each conjugate gradient resolution.
#define K2 10.0f // Second linear diffusion parameter.
#define LAMBDA 0.0001f // Perona-Malik coefficient parameter.

#define INPUT "input.pgm" // Name of the file containing the input data.
#define OUTPUT "result.pgm" // Name of the file where the output is stored.
#define LOG "log.csv" // Name of the file to which the log information is added.

#endif