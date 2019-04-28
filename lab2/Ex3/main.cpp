#include <mpi.h>
#include <stdio.h>
#define MIN(x, y) ((x)<(y) ? (x):(y))

int main(int argc, char** argv)
{

	int rank;
	int size;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int N;
	double alpha;
	double time;



	if (rank == 0)
	{
		sscanf(argv[1], "%d", &N);
	}

	MPI_Bcast(&N, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

	int n0 = rank * (N /  (size -  1)) + 1;
	int n1 = MIN(N,  (rank + 1) * (N / (size -  1)));

	if (rank == 0)
	{
		sscanf(argv[2], "%f", &alpha);
	}

	MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double *vPart = new double[N];
	double *wPart = new double[N];
	for (int i = n0; i <= n1; i++)
	{
		vPart[i - n0] = i;
		wPart[i - n0] = N - 1 - i;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0)
	{
		time = MPI_Wtime();
	}
	double *rPart = new double[N];
	for (int i = n0; i <= n1; i++)
	{
		rPart[i - n0] = vPart[i - n0] + alpha * wPart[i - n0];
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0)
	{
		time = MPI_Wtime() - time;
		printf("Vector update time for N = %d and %d processes: %fs", N, size, time);
	}
	delete[] rPart;


	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0)
	{
		time = MPI_Wtime();
	}
	double sPart = 0;
	for (int i = n0; i <= n1; i++)
	{
		sPart += vPart[i - n0] * wPart[i - n0];
	}
	double s;
	MPI_Reduce(&sPart, &s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0)
	{
		time = MPI_Wtime() - time;
		printf("\nScalar product time for N = %d and %d processes: %fs", N, size, time);
	}


	delete[] vPart;
	delete[] wPart;

	MPI_Finalize();
	return 0;
}