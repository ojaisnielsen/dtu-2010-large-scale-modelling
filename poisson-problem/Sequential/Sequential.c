#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <mpi.h>
#include "../utils.h"

//// Parameters

int N = 100; // Domain width
int maxIter = 1000; // Maximum number of iterations
float delta = 0.1f; // Space step
float epsilon = 0.1f; // Residual converged maximum value

float sourceDist(int x, int y) // Source distribution
{
	return x >= (N / 2) && x <=  2 * N / 3 && y >= N / 6 && y <= N / 3 ? 200.0f : 0.0f;
}

float limitsDist(int x, int y) // Limits distribution
{
	return x == N - 1 || x == 0 || y == N - 1 ? 20.0f : 0.0f;
}

int main(int argc, char** argv)
{
	int isConverged, nIter, x, y;
	float rSquare;
	float* f;
	float* u;
	float* uNew;
	double time;

	MPI_Init(&argc, &argv);

	//// Load input data and allocate computational buffers

	f = (float*) malloc(N * N * sizeof(float));
	drawDist(f, &sourceDist, N, N);
	//writePgm("source.pgm", totalF, N, N);

	u = (float*) malloc(N * N * sizeof(float));
	drawDist(u, &limitsDist, N, N);
	//writePgm("limits.pgm", totalU, N, N);

	uNew = (float*) malloc(N * N * sizeof(float));

	//// Main loop

	isConverged = 0;
	nIter = 0;

	time = MPI_Wtime();
	while(!isConverged)
	{
		//// Computations

		rSquare = 0;

		for (x = 0; x < N; x++)
		{				
			for (y = 0; y < N; y++)
			{
				float r;

				//// Compute new value of variable
				if (x == 0  || x == N - 1 || y == 0 || y == N - 1)
				{
					uNew[x + y * N] = u[x + y * N];
					continue;
				}					

				uNew[x + y * N] = delta * delta * f[x + y * N];
				uNew[x + y * N] += u[x + (y - 1) * N];
				uNew[x + y * N] += u[x + (y + 1) * N];
				uNew[x + y * N] += u[x - 1 + y * N];
				uNew[x + y * N] += u[x + 1 + y * N];
				uNew[x + y * N] /= 4;


				//// Compute new value of the squared residual
				r = uNew[x + y * N] - u[x + y * N];
				rSquare += r * r;
			}
		}

		//// Update variable
		memcpy(u, uNew, N * N * sizeof(float));

		//// Compute global squared residual
		if (nIter % 100 == 0)
		{
			printf("Iteration: %d; residual: %f\n", nIter, rSquare);
		}

		//// Check if the algorithm is converged
		isConverged = rSquare < epsilon * epsilon || nIter == maxIter;
		nIter++;
	}

	time = MPI_Wtime() - time;
	printf("Computational time: %f\n", time);
	logResults("log.txt", "Sequential", 1, N * N, nIter, time);


	writePgm("resultsS.pgm", u, N, N);		


	free(f);
	free(u);
	free(uNew);

	MPI_Finalize();
	return 0;
}