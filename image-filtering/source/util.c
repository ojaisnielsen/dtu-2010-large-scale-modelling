#include "util.h"

extern MPI_Comm cartComm;
extern int gridCoords[2], gridDims[2], totalInputDims[2], nBuffers, size, rank, nIters;
double timer;
void *buffers[100];

void timerStart()
{
	/*
	One process starts the timer after synchronizing all the processes.
	*/
	
	MPI_Barrier(cartComm);
	if (rank == 0)
	{
		timer = MPI_Wtime();
	}
}

void timerStop()
{
	/*
	The process that has the timer runing stops it after synchronizing all the processes. It adds informations about the computation in a CSV file.
	*/
	
	FILE *logFile;
	int nThreads;

	MPI_Barrier(cartComm);
	if (rank == 0)
	{
		timer = MPI_Wtime() - timer;
		#pragma omp parallel
		{
			#pragma omp critical
			nThreads = omp_get_num_threads();	
		}
		logFile = fopen(LOG, "a");
		fprintf(logFile, "%d, %d, %d, %d, %f\n", totalInputDims[0] * totalInputDims[1], size, nThreads, nIters, timer); // The size of the global input, the number of processes, the number of threads per process, the total number of iterations and the time of computation is stored.
		fclose(logFile);
	}
}

void processSays(char *format, ...)
{
	/*
	Displays a formated text with the coordinates of the process indicated.
	*/
    va_list args;

    va_start(args, format);
    printf("Process (%d, %d) says:\n	", gridCoords[0], gridCoords[1]);
    vprintf(format, args);
    va_end(args);
}


int initRectComm( int verbose )
{
	/*
	Intitializes a cartesian grid. Returns 1 if it fails, 0 otherwise.
	*/
	
	int rank, size, periods[2];

	MPI_Comm_size(MPI_COMM_WORLD, &size);

	
	gridDims[0] = MAX(1, (int)sqrt(size * totalInputDims[0] / (float)totalInputDims[1]));
	gridDims[1] = size / gridDims[0];

	if (size != gridDims[0] * gridDims[1])
	{
		if (verbose)
		{
			fprintf(stderr, "Invalid number of processes. Try with %d processes.\n", gridDims[0] * gridDims[1]);
		}
		return 1;
	}

	periods[0] = periods[1] = 0;
	MPI_Cart_create(MPI_COMM_WORLD, 2, gridDims, periods, 1, &cartComm);
	MPI_Comm_rank(cartComm, &rank);
	MPI_Cart_coords(cartComm, rank, 2, gridCoords);
	return 0;
}

void getBlockRange( int *coords, int *row0, int *col0, int *rows, int *cols )
{
	/*
	Computes the range of the block associated to the calling process.
	*/
	
	int a, b;
	a = totalInputDims[0] % gridDims[0] == 0 ? totalInputDims[0] / gridDims[0] : 1 + (totalInputDims[0] / gridDims[0]);
	b = totalInputDims[1] % gridDims[1] == 0 ? totalInputDims[1] / gridDims[1] : 1 + (totalInputDims[1] / gridDims[1]);
	*row0 = coords[0] * a;
	*col0 = coords[1] * b;
	*rows = MIN((coords[0] + 1) * a, totalInputDims[0]) - *row0;
	*cols = MIN((coords[1] + 1) * b, totalInputDims[1]) - *col0;
}

void getMaxBlockDims( int *maxDims )
{
	/*
	Computes the maximum dimensions among all the blocks.
	*/
	
	maxDims[0] = totalInputDims[0] % gridDims[0] == 0 ? totalInputDims[0] / gridDims[0] : 1 + (totalInputDims[0] / gridDims[0]);
	maxDims[1] = totalInputDims[1] % gridDims[1] == 0 ? totalInputDims[1] / gridDims[1] : 1 + (totalInputDims[1] / gridDims[1]);
}

void delayedFree(void *buffer)
{
	/*
	Adds a buffer to be freed later.
	*/
	
	buffers[nBuffers] = buffer;
	nBuffers++;
}

void delayedFreeMatrix(Matrix matrix)
{
	/*
	Adds a matrix to be freed later.
	*/
	
	buffers[nBuffers] = *(matrix->values);
	nBuffers++;
	buffers[nBuffers] = matrix->values;
	nBuffers++;
	buffers[nBuffers] = matrix;
	nBuffers++;
}

void performDelayedFree()
{
	/*
	Frees the buffers and matrices set to be freed later.
	*/
	
	int i;

	for (i = 0; i < nBuffers; i++)
	{
		free(buffers[i]);
	}
}
