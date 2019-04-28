#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <mpi.h>
#include "../utils.h"

//// Parameters

float a = 10.0f; // Domain width
int N = 100; // Discretized domain width
int maxIter = 1000; // Maximum number of iterations (-1 for no limit)
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
	int rank, n, x0, xSup, y0, ySup, nNeighbours, coords[2], neighboursDirections[4], neighboursOffsets[4], top, bottom, left, right, inSidesX0[4], inSidesXSup[4], inSidesY0[4], inSidesYSup[4], outSidesX0[4], outSidesXSup[4], outSidesY0[4], outSidesYSup[4], buffersSizes[4], neighbour, isConverged, nIter;
	MPI_Comm cartComm;
	float* totalF;
	float* totalU;
	float* f;
	float* u;
	float* uNew;
	float* receiveBuffers[4];
	float* sendBuffers[4];
	double time;

	//// Initialize MPI

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (initSquareComm(&cartComm, &n, coords))
	{
		if (rank == 0)
		{
			printf("Number of processes must be square.");
		}
		MPI_Finalize();
		exit(1);		
	}


	//// Load input data and allocate computational buffers

	totalF = (float*) malloc(N * N * sizeof(float));
	drawDist(totalF, &sourceDist, N, N);
	//writePgm("source.pgm", totalF, N, N);

	totalU = (float*) malloc(N * N * sizeof(float));
	drawDist(totalU, &limitsDist, N, N);
	//writePgm("limits.pgm", totalU, N, N);

	getBlockRange(coords, N, n, &x0, &xSup, &y0, &ySup);

	f = (float*) malloc((xSup - x0) * (ySup - y0) * sizeof(float));
	cropMatrix(totalF, f, x0, xSup, y0, ySup, N, 1, 1);
	free(totalF);

	u = (float*) malloc((xSup - x0) * (ySup - y0) * sizeof(float));
	cropMatrix(totalU, u, x0, xSup, y0, ySup, N, 1, 1);
	free(totalU);

	uNew = (float*) malloc((xSup - x0) * (ySup - y0) * sizeof(float));

	//// Compute exchanges ranges and allocate exchanges buffers

	top = bottom = left = right = -1;
	getNeighboursSidesRanges(x0, xSup, y0, ySup, N, N, &nNeighbours, neighboursDirections, neighboursOffsets, &top, &bottom, &left, &right, inSidesX0, inSidesXSup, inSidesY0, inSidesYSup, outSidesX0, outSidesXSup, outSidesY0, outSidesYSup);

	for (neighbour = 0; neighbour < nNeighbours; neighbour++)
	{
		buffersSizes[neighbour] = (inSidesXSup[neighbour] - inSidesX0[neighbour]) * (inSidesYSup[neighbour] - inSidesY0[neighbour]);
		receiveBuffers[neighbour] = (float*) malloc(buffersSizes[neighbour] * sizeof(float));
		sendBuffers[neighbour] = (float*) malloc(buffersSizes[neighbour] * sizeof(float));

	}

	//// Main loop

	isConverged = 0;
	nIter = 0;

	MPI_Barrier(cartComm);
	if (rank == 0)
	{
		time = MPI_Wtime();
	}
	while(!isConverged)
	{
		float rSquare, totalRSquare;
		int neighbourRank, dummyRank, neighbour, x, y;
		MPI_Request dummyReq, reqs[4];
		MPI_Status statuses[4];

		//// Data exchanges with the neighbouring blocks

		for (neighbour = 0; neighbour < nNeighbours; neighbour++)
		{
			int relX0, relXSup, relY0, relYSup;

			MPI_Cart_shift(cartComm, neighboursDirections[neighbour], neighboursOffsets[neighbour], &dummyRank, &neighbourRank);

			//// Send inner side to neighbouring block

			relX0 = inSidesX0[neighbour] - x0;
			relXSup = inSidesXSup[neighbour] - x0;
			relY0 = inSidesY0[neighbour] - y0;
			relYSup = inSidesYSup[neighbour] - y0;
			cropMatrix(u, sendBuffers[neighbour], relX0, relXSup, relY0, relYSup, xSup - x0, 1, 1);
			MPI_Isend(sendBuffers[neighbour], buffersSizes[neighbour], MPI_FLOAT, neighbourRank, 0, cartComm, &dummyReq);

			//// Receive outter side from neighbouring block
			MPI_Irecv(receiveBuffers[neighbour], buffersSizes[neighbour], MPI_FLOAT, neighbourRank, 0, cartComm, reqs + neighbour);
		}

		//// Wait for the receives
		MPI_Waitall(nNeighbours, reqs, statuses);	

		//// Computations

		rSquare = 0;

		for (x = 0; x < xSup - x0; x++)
		{				
			for (y = 0; y < ySup - y0; y++)
			{
				float r;

				//// Compute new value of variable
				if ((x == 0  && coords[0] == 0) || (x == xSup - x0 - 1  && coords[0] == n - 1) || (y == 0  && coords[1] == 0) || (y == ySup - y0 - 1  && coords[1] == n - 1))
				{
					uNew[x + y * (xSup - x0)] = u[x + y * (xSup - x0)];
					continue;
				}					

				uNew[x + y * (xSup - x0)] = (a / N) * (a / N) * f[x + y * (xSup - x0)];
				uNew[x + y * (xSup - x0)] += y > 0 ? u[x + (y - 1) * (xSup - x0)] : receiveBuffers[top][x];
				uNew[x + y * (xSup - x0)] += y < ySup - y0 - 1 ? u[x + (y + 1) * (xSup - x0)] : receiveBuffers[bottom][x];
				uNew[x + y * (xSup - x0)] += x > 0 ? u[x - 1 + y * (xSup - x0)] : receiveBuffers[left][y];
				uNew[x + y * (xSup - x0)] += x < xSup - x0 - 1 ? u[x + 1 + y * (xSup - x0)] : receiveBuffers[right][y];
				uNew[x + y * (xSup - x0)] /= 4;


				//// Compute new value of the squared residual
				r = uNew[x + y * (xSup - x0)] - u[x + y * (xSup - x0)];
				rSquare += r * r;
			}
		}

		//// Update variable
		memcpy(u, uNew, (xSup - x0) * (ySup - y0) * sizeof(float));

		//// Compute global squared residual
		MPI_Allreduce(&rSquare, &totalRSquare, 1, MPI_FLOAT, MPI_SUM, cartComm);
		if (rank == 0 && nIter % 100 == 0)
		{
			printf("Iteration: %d; residual: %f\n", nIter, totalRSquare);
		}

		//// Check if the algorithm is converged
		isConverged = totalRSquare < epsilon * epsilon || nIter == maxIter;
		nIter++;
	}

	MPI_Barrier(cartComm);
	if (rank == 0)
	{
		time = MPI_Wtime() - time;
		printf("Computational time: %f\n", time);
		logResults("log.csv", "Jacobi", n * n, N * N, nIter, time);
	}

	//// Gather results

	if (rank > 0)
	{
		MPI_Send(u, (xSup - x0) * (ySup - y0), MPI_FLOAT, 0, 0, cartComm);
	}
	else
	{
		int senderRank;
		totalU = (float*) malloc(N * N * sizeof(float));
		insertBlock(u, totalU, x0, xSup, y0, ySup, N);
		for (senderRank = 1; senderRank < n * n; senderRank++)
		{
			MPI_Status dummyStatus;
			int senderCoords[2];
			MPI_Cart_coords(cartComm, senderRank, 2, senderCoords);
			getBlockRange(senderCoords, N, n, &x0, &xSup, &y0, &ySup);
			u = (float*) malloc((xSup - x0) * (ySup - y0) * sizeof(float));			
			MPI_Recv(u, (xSup - x0) * (ySup - y0), MPI_FLOAT, senderRank, 0, cartComm, &dummyStatus);
			insertBlock(u, totalU, x0, xSup, y0, ySup, N);
		}

		writePgm("resultsJ.pgm", totalU, N, N);		
	}


	free(f);
	free(u);
	free(uNew);
	for (neighbour = 0; neighbour < nNeighbours; neighbour++)
	{
		free(sendBuffers[neighbour]);
		free(receiveBuffers[neighbour]);
	}

	MPI_Finalize();
	return 0;
}