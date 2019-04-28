#include "conjugate_gradient.h"

extern MPI_Comm cartComm;
extern MPI_Status statuses[4];
extern MPI_Request sendRequests[4], recvRequests[4];
extern int gridCoords[2], gridDims[2], neighboursRanks[4], nNeighbours;
extern float *recvBuffers[4], *sendBuffers[2];

void initConjugateGradient(Matrix invM, Matrix b, Matrix x, Matrix res, Matrix p, Matrix z, float *totalRho)
{
	/*
	Initialization of the conjugate-gradient resolution. 
	*/
	
	float rho;

	// The first estimate is set to 0. The other matrices are initialized following that.
	
		setValueMatrix(0, x);
		copyMatrix(b, res);
		getElementMultiplyMatrix(invM, res, z);	//
		rho = dotProductMatrix(z, res);
		MPI_Allreduce(&rho, totalRho, 1, MPI_FLOAT, MPI_SUM, cartComm);
		copyMatrix(z, p);

	// The values of the initial direction previously computed corresponding to the edges of the block are exchanged with the neighbor processes.
	
		receiveEdges(p->rows, p->cols);
		sendEdges(p);
}

void finalizeConjugateGradient(Matrix p)
{
	/*
	Finalization of the conjugate-gradient resolution.
	*/
	
	// The values of the current direction corresponding to the edges of the block are sent to terminate the pending asynchronous receive operations.
		
		sendEdges(p);

	// The function waits for all the pending operations to finish.
	
		MPI_Waitall(nNeighbours, recvRequests, statuses);
		MPI_Waitall(nNeighbours, sendRequests, statuses);
}

int iterConjugateGradient(Matrix a, Matrix m, Matrix invM, Matrix x, Matrix res, Matrix p, Matrix q, Matrix z, float *totalRho, float eps)
{
	/*
	Perform one iteration of the conjugate-gradient resolution. It returns 1 if this makes the algorithm converge, 0 otherwise.
	*/
	
	int i, r, c;
	float pq, zr, totalPq, totalZr, alpha, rho, oldTotalRho;
	
	// The Neumann boundary conditions are enforced.
	
		setNeumannBoundaries(0, p);
	
	// The five-point stencil formula is applied. For pixels situated on the edges of the block, the values received from the neighbors or imposed by the Neumann conditions are used.
	
		MPI_Waitall(nNeighbours, recvRequests, statuses);	

		#pragma omp parallel for private(i) private(r) private(c)
		for (i = 0; i < p->rows * p->cols; i++)
		{
			r = i / p->cols;
			c = i % p->cols;

			q->values[r][c] = a->values[r][5 * c] * (r > 0 ? p->values[r - 1][c] : recvBuffers[0][c]);
			q->values[r][c] += a->values[r][5 * c + 1] * (r < x->rows - 1 ? p->values[r + 1][c] : recvBuffers[1][c]);
			q->values[r][c] += a->values[r][5 * c + 2] * (c > 0 ? p->values[r][c - 1] : recvBuffers[2][r]);
			q->values[r][c] += a->values[r][5 * c + 3] * (c < x->cols - 1 ? p->values[r][c + 1] : recvBuffers[3][r]);
			q->values[r][c] += a->values[r][5 * c + 4] * p->values[r][c];			
		}

	// The values of the current direction that the neighbor processes have to send are set to be received.

		receiveEdges(p->rows, p->cols);
	
	// The other matrices are updated.

		pq = dotProductMatrix(p,q);
		MPI_Allreduce(&pq, &totalPq, 1, MPI_FLOAT, MPI_SUM, cartComm);
		zr = dotProductMatrix(z,res);
		MPI_Allreduce(&zr, &totalZr, 1, MPI_FLOAT, MPI_SUM, cartComm);
		alpha = totalZr / totalPq;	

		multiplyMatrix(alpha, p);
		addMatrix(p, x);
		multiplyMatrix(1 / alpha, p);
		multiplyMatrix(-alpha, q);
		addMatrix(q, res);
		multiplyMatrix(-1 / alpha, q);

		getElementMultiplyMatrix(invM, res, z);

		rho = dotProductMatrix(z, res);
		oldTotalRho = *totalRho;
		MPI_Allreduce(&rho, totalRho, 1, MPI_FLOAT, MPI_SUM, cartComm);

		// The new norm of the global residual si compared to the convergence criterion. 1 is returned if the algorithm has converged.

			if (*totalRho < eps * eps)
			{		
				return 1;
			}

		MPI_Waitall(nNeighbours, sendRequests, statuses);

		multiplyMatrix(*totalRho / oldTotalRho, p);

		addMatrix(z, p);
		
	// The values of the current direction corresponding to the edges of the block are sent to the neighbor processes.

		sendEdges(p);

	return 0;

}

void sendEdges(Matrix matrix)
{
	/*
	Sends the edges of a matrix to the appropriate neighbor processes.
	*/
	
	int n;

	n = 0;
	if (gridCoords[0] > 0) // Top neighbor.
	{
		MPI_Isend(*(matrix->values), matrix->cols, MPI_FLOAT, neighboursRanks[0], 0, cartComm, sendRequests);
		n++;
	}
	if (gridCoords[0] < gridDims[0] - 1) // Bottom neighbor.
	{
		MPI_Isend(*(matrix->values) + (matrix->rows - 1) * matrix->cols, matrix->cols, MPI_FLOAT, neighboursRanks[1], 0, cartComm, sendRequests + n);
		n++;
	}
	if (gridCoords[1] > 0) // Left neighbor.
	{
		copySubMatrixToBuffer(0, 0, matrix->rows, 1, matrix, sendBuffers[0]);
		MPI_Isend(sendBuffers[0], matrix->rows, MPI_FLOAT, neighboursRanks[2], 0, cartComm, sendRequests + n);
		n++;
	}
	if (gridCoords[1] < gridDims[1] - 1) // Right neighbor.
	{
		copySubMatrixToBuffer(0, matrix->cols - 1, matrix->rows, 1, matrix, sendBuffers[1]);		
		MPI_Isend(sendBuffers[1], matrix->rows, MPI_FLOAT, neighboursRanks[3], 0, cartComm, sendRequests + n);
	}
}

void receiveEdges(int rows, int cols)
{
	/*
	Receives the edges of a matrix sent by the appropriate neighbor processes.
	*/
	int n;
	
	n = 0;
	if (gridCoords[0] > 0) // Top neighbor.
	{		
		MPI_Irecv(recvBuffers[0], cols, MPI_FLOAT, neighboursRanks[0], 0, cartComm, recvRequests);
		n++;
	}
	if (gridCoords[0] < gridDims[0] - 1) // Bottom neighbor.
	{
		MPI_Irecv(recvBuffers[1], cols, MPI_FLOAT, neighboursRanks[1], 0, cartComm, recvRequests + n);
		n++;
	}
	if (gridCoords[1] > 0) // Left neighbor.
	{
		MPI_Irecv(recvBuffers[2], rows, MPI_FLOAT, neighboursRanks[2], 0, cartComm, recvRequests + n);
		n++;
	}
	if (gridCoords[1] < gridDims[1] - 1) // Right neighbor.
	{
		MPI_Irecv(recvBuffers[3], rows, MPI_FLOAT, neighboursRanks[3], 0, cartComm, recvRequests + n);
	}
}

void setNeumannBoundaries(float val, Matrix matrix)
{
	/*
	Enforces the Neumann boundary conditions for the appropriate processes.
	*/
	
	int r, c;
	
	if (gridCoords[0] == 0) // Top boundary.
	{
		recvBuffers[0][0] = 0;
		recvBuffers[0][matrix->cols - 1] = 0;
		for (c = (gridCoords[1] == 0); c < matrix->cols - (gridCoords[1] == gridDims[1] - 1); c++)
		{
			recvBuffers[0][c] = matrix->values[1][c] + 2 * val;
		}
	}
	if (gridCoords[0] == gridDims[0] - 1) // Bottom boundary.
	{
		recvBuffers[1][0] = 0;
		recvBuffers[1][matrix->cols - 1] = 0;
		for (c = (gridCoords[1] == 0); c < matrix->cols - (gridCoords[1] == gridDims[1] - 1); c++)
		{
			recvBuffers[1][c] = matrix->values[matrix->rows - 2][c] + 2 * val;
		}
	}
	if (gridCoords[1] == 0) // Left boundary.
	{
		recvBuffers[2][0] = 0;
		recvBuffers[2][matrix->rows - 1] = 0;
		for (r = (gridCoords[0] == 0); r < matrix->rows - (gridCoords[0] == gridDims[0] - 1); r++)
		{
			recvBuffers[2][r] = matrix->values[r][1] + 2 * val;
		}
	}
	if (gridCoords[1] == gridDims[1] - 1) // Right boundary.
	{
		recvBuffers[3][0] = 0;
		recvBuffers[3][matrix->rows - 1] = 0;
		for (r = (gridCoords[0] == 0); r < matrix->rows - (gridCoords[0] == gridDims[0] - 1); r++)
		{
			recvBuffers[3][r] = matrix->values[r][matrix->cols - 2] + 2 * val;
		}
	}	
}
