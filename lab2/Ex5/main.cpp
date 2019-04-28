#include <mpi.h>
#include <stdio.h>
#define N 1047
#define MIN(x, y) ((x)<(y) ? (x):(y))
#define VEC(i) i - 1
#define MAT(i, j) (i - 1) + N * (j - 1)

int main(int argc, char** argv)
{
	int rank;
	int size;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int dn = N % size == 0 ? N / size : (N / size) + 1;
	int i0 = rank * dn + 1;
	int i1 = MIN(N, i0 + dn - 1);

	// Initialization

	MPI_Request* sendReqs = new MPI_Request[size];
	MPI_Request* receiveReqs = new MPI_Request[size];

	for (int p = 0; p < size; p++)
	{
		sendReqs[p] = MPI_REQUEST_NULL;
		receiveReqs[p] = MPI_REQUEST_NULL;
	}

	double* A = new double[N * N];
	double* y = new double[N];
	double* x = new double[N];

	printf("process %d says i0: %d i1: %d\n", rank, i0, i1);

	for (int i = i0; i <= i1; i++)
	{
		x[VEC(i)] = 0;
		y[VEC(i)] = 0.1;
		for (int j = 1; j <= N; j++)
		{
			A[MAT(i, j)] = 0.2;
		}
	}

	// Send y
	for (int p = 0; p < size; p++)
	{
		
		if (p != rank) 
		{
			printf("process %d says: send %d to %d\n", rank, i1 - i0 + 1, p);
			MPI_Isend(y + p * dn, i1 - i0 + 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, sendReqs + p);
		}
	}

	// Receive y

	for (int p = 0; p < size - 1; p++)
	{
		
		if (p != rank) 
		{
			printf("process %d says: receive %d from %d\n", rank, dn, p);
			MPI_Irecv(y + p * dn, dn, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, receiveReqs + p);
		}
	}

	int p = size - 1;
	if (p != rank) 
	{
		printf("process %d says: receive %d from %d\n", rank, N - (size - 1) * dn, p);
		MPI_Irecv(y + p * dn, N - p * dn, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, receiveReqs + p);
	}


	// Compute

	for (int i = i0; i <= i1; i++)
	{
		for (int j = i0; j <= i1; j++)
		{
			x[VEC(i)] += A[MAT(i, j)] * y[VEC(j)];
		}
	}

	int* received = new int[size];
	int totalReceived = 0;
	int currentTest = 0;
	int currentResult;
	MPI_Status status;

	for (int i = 0; i < size; i++)
	{
		received[i] = 0;
	}

	while (totalReceived < size - 1)
	{
		int testRank = currentTest % size;
		if (received[testRank] == 0 && testRank != rank)
		{						
			MPI_Test(receiveReqs + testRank, &currentResult, &status);			
			if (currentResult == 1)
			{
				printf("process %d says: received from %d successfully\n", rank, testRank);
				received[testRank] = 1;
				totalReceived++;

				int j0 = testRank * dn + 1;
				int j1 = MIN(N - 1, j0 + dn - 1);

				for (int i = i0; i <= i1; i++)
				{
					for (int j = j0; j <= j1; j++)
					{
						x[VEC(i)] += A[MAT(i, j)] * y[VEC(j)];
					}
				}
			}
		}
		currentTest++;
	}

	delete[] A;
	delete[] y;
	delete[] x;
	delete[] sendReqs;
	delete[] receiveReqs;
	

	MPI_Finalize();
	return 0;
}