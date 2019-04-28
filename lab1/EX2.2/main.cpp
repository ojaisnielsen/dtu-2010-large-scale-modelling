#include <mpi.h>
#include <stdio.h>


int main(int argc, char** argv)
{
	int sum = 0;
	int rank;
	int size;
	int received;
	MPI_Status status;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank > 0)
	{
		MPI_Send(&rank, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
		MPI_Recv(&received, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
		sum += received;
	}
	if (rank < size - 1)
	{
		MPI_Send(&rank, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
		MPI_Recv(&received, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &status);
		sum += received;
	}
	printf("Process %d says '%d'\n", rank, sum);

	MPI_Finalize();
	return 0;
}