#include <mpi.h>
#include <omp.h>
#include "common.h"
#include "util.h"
#include "conjugate_gradient.h"

int initialize();
void compute();
void finalize();

void *buffers[100];
int nBuffers, threadSupport, rank, cartRank, size, totalInputDims[2], maxDims[2], gridDims[2], gridCoords[2], neighboursRanks[4], nNeighbours, nIters;
float *recvBuffers[4], *sendBuffers[2];
MPI_Comm cartComm;
MPI_Status statuses[4]; 
MPI_Request sendRequests[4], recvRequests[4];
Matrix input, x;