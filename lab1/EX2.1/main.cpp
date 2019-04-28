#include <mpi.h>
#include <stdio.h>


int main(int argc, char** argv)
{
	int message = 5;

	int rank;
	int size;
	int received;
	MPI_Status status;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0)
	{
		MPI_Send(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		MPI_Recv(&received, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, &status);
		printf("Process %d says '%d'\n", rank, received);
	}
	else 
	{
		MPI_Recv(&received, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
		received++;
		printf("Process %d says '%d'\n", rank, received);
		MPI_Send(&received, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}