#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 1000
#define MIN(x, y) ((x)<(y) ? (x):(y))
#define VEC(i) i - 1
#define MAT(i, j) (i - 1) + N * (j - 1)

int main(int argc, char** argv)
{
	int rank;
	int size;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int n = (int) sqrt((double) size);
	if (size != n * n)
	{
		if (rank == 0)
		{
   			printf ("Number of processes must be square.");
		}
    	exit(1);
	}

	int dims[2] = {n, n};
	int periods[2] = {0, 0};
	MPI_Comm cartesianComm;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cartesianComm);

	MPI_Comm colComm;
	int remainDims[2] = {1, 0};
	MPI_Cart_sub(cartesianComm, remainDims, &colComm);

	int coords[2];
	MPI_Comm_rank(cartesianComm, &rank);
	MPI_Cart_coords(cartesianComm, rank, 2, coords);


	int dn = N % n == 0 ? N / n : (N / n) + 1;
	int i0 = coords[0] * dn + 1;
	int i1 = MIN(N, i0 + dn - 1);
	int j0 = coords[1] * dn + 1;
	int j1 = MIN(N, j0 + dn - 1);

	double* A = new double[N * N];
	double* y = new double[N];
	double* x = new double[N];

	for (int j = j0; j <= j1; j++)
	{
		y[VEC(j)] = 0;
		if (coords[0] == coords[1])
		{
			x[VEC(j)] = 5;
		}
		for (int i = i0; i <= i1; i++)
		{
			A[MAT(i, j)] = 4;
		}
	}

	MPI_Bcast(x + j0 - 1, j1 - j0 + 1, MPI_DOUBLE, int root, MPI_Comm comm );

	for (int i = i0; i <= i1; i++)
	{		
		for (int j = j0; j <= j1; j++)
		{
			x[VEC(i)] += A[MAT(i, j)] * y[VEC(j)];
		}
	}

	
	

	MPI_Finalize();
	return 0;
}