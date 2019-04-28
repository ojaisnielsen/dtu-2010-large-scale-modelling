#include <mpi.h>
#include <stdio.h>
#define MIN(x, y) ((x)<(y) ? (x):(y))


int main(int argc, char** argv)
{

	int N = 1000;

	int rank;
	int size;
	float received;
	MPI_Status status;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int n0 = rank * (N /  (size -  1)) + 1;
	int n1 = MIN(N,  (rank + 1) * (N / (size -  1)));

	float sum = 0.0f;
	float h = 1.0f / (float) N;

	for (int i = n0; i <= n1; i++)
	{
		float a = ((float) i - 0.5f) * h;
		sum += 4.0f / (1 + a * a);
	}

	if (rank > 0)
	{
		MPI_Send(&sum, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	}
	else
	{
		for (int i = 1; i < size; i++)
		{
			MPI_Recv(&received, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
			sum += received;
		}
		sum *= h;
		printf("Process %d says '%f'\n", rank, sum);
	}

	MPI_Finalize();
	return 0;
}