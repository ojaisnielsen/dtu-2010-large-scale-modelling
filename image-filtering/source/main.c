#include "main.h"

int main(int argc, char **argv)
{
	MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED, &threadSupport); // Initialization of MPI.
	omp_set_nested(1); // Initialization of OpenMP.

	if (!initialize()) // If the initialization of the processes is successful:
	{
		timerStart(); // start the timer,
		compute(); // perform the computation,
		timerStop(); // stop the timer,
		finalize(); // finalize the computation.
	}

	performDelayedFree(); // Free all the remaining buffers.
	MPI_Finalize(); // Finalize MPI.
	return 0;
}

int initialize()
{
	/*
	Initialization of the processes. Returns 1 if it fails, 0 otherwise.
	*/
	
	MPI_Status status;
	MPI_Request request;
	int i, cartRank, coords[2], row0, col0, rows, cols, shiftedRank;
	Matrix totalInput;
	float *sendBuffer;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	// One process tries to read the input data and to broadcast its dimensions.
	
		totalInputDims[0] = -1;
		totalInputDims[1] = -1;
		
		if (rank == 0)
		{
			if (!readPgm(INPUT, &totalInput))
			{
				totalInputDims[0] = totalInput->rows;
				totalInputDims[1] = totalInput->cols;
			}
		}
		
		MPI_Bcast(totalInputDims, 2, MPI_INT, 0, MPI_COMM_WORLD);
	
	// If the input could not be read or if a Cartesian communicator could not be created, the function returns 1.		
	
		if (totalInputDims[0] <= 0 || totalInputDims[1] <= 0 || initRectComm(rank == 0))
		{		
			return 1;
		}
		
		MPI_Comm_rank(cartComm, &cartRank);	
	
	// The process that read the input data dispatches each block to its process.
	
		getMaxBlockDims(maxDims);
		
		if (rank == 0)
		{		
			sendBuffer = (float *) malloc(size * maxDims[0] * maxDims[1] * sizeof(float));
			for (i = 0; i < size; i++)
			{		
				MPI_Cart_coords(cartComm, i, 2, coords);
				getBlockRange(coords, &row0, &col0, &rows, &cols);
				copySubMatrixToBuffer(row0, col0, rows, cols, totalInput, sendBuffer + i * maxDims[0] * maxDims[1]);
				MPI_Isend(sendBuffer + i * maxDims[0] * maxDims[1], cols * rows, MPI_FLOAT, i, 0, cartComm, &request);
			}		
			freeMatrix(totalInput);
		}
	
	// Each process receives its block.
	
		getBlockRange(gridCoords, &row0, &col0, &rows, &cols);
		input = newEmptyMatrix(rows, cols);
		delayedFreeMatrix(input);
		MPI_Recv(*(input->values), cols * rows, MPI_FLOAT, 0, 0, cartComm, &status);
		MPI_Barrier(cartComm);

	if (rank == 0)
	{		
		free(sendBuffer);
	}
	
	// Each process computes the ranks of its neighbor processes.
	
		nNeighbours = (gridCoords[0] > 0) + (gridCoords[0] < gridDims[0] - 1) + (gridCoords[1] > 0) + (gridCoords[1] < gridDims[1] - 1);
		if (gridCoords[0] > 0)
		{
			MPI_Cart_shift(cartComm, 0, -1, &shiftedRank, neighboursRanks);
		}
		if (gridCoords[0] < gridDims[0] - 1)
		{
			MPI_Cart_shift(cartComm, 0, 1, &shiftedRank, neighboursRanks + 1);
		}
		if (gridCoords[1] > 0)
		{
			MPI_Cart_shift(cartComm, 1, -1, &shiftedRank, neighboursRanks + 2);
		}
		if (gridCoords[1] < gridDims[1] - 1)
		{
			MPI_Cart_shift(cartComm, 1, 1, &shiftedRank, neighboursRanks + 3);
		}
	
	return 0;
}

void compute()
{
	/*
	Main computation.
	*/
	
	int i, r, c, shiftR, shiftC, offsets[2];
	Matrix a, b, m, invM, res, p, q, z, s;
	float totalRho, g;
	
	// Initialization of the second member matrix. It is the input block with extra 1-pixel wide zero-value row(s)/column(s) for the processes located on the edges of the grid. That way, the first pass solves the diffusion on a domain larger than the global input by one pixel in each direction.
	
		offsets[0] = (gridCoords[0] == 0);
		offsets[1] = (gridCoords[1] == 0);

		b = newEmptyMatrix(input->rows + (gridCoords[0] == 0) + (gridCoords[0] == gridDims[0] - 1), input->cols + (gridCoords[1] == 0) + (gridCoords[1] == gridDims[1] - 1));
		setValueMatrix(0, b);
		insertMatrix(offsets[0], offsets[1], input, b);
	
	// Initializations of the other matrices and buffers.	
	
		x = newEmptyMatrix(b->rows, b->cols);
		a = newEmptyMatrix(x->rows, 5 * x->cols);
		m = newEmptyMatrix(x->rows, x->cols);
		invM = newEmptyMatrix(x->rows, x->cols);	
		res = newEmptyMatrix(x->rows, x->cols);
		z = newEmptyMatrix(x->rows, x->cols);
		q = newEmptyMatrix(x->rows, x->cols);
		p = newEmptyMatrix(x->rows, x->cols);

		recvBuffers[0] = (float *) malloc(x->cols * sizeof(float));
		recvBuffers[1] = (float *) malloc(x->cols * sizeof(float));
		recvBuffers[2] = (float *) malloc(x->rows * sizeof(float));
		recvBuffers[3] = (float *) malloc(x->rows * sizeof(float));
		sendBuffers[0] = (float *) malloc(x->rows * sizeof(float));
		sendBuffers[1] = (float *) malloc(x->rows * sizeof(float));

	// The five-point stencil coefficients for the first pass are computed as well as the diagonal preconditioner and its inverse.
	
		#pragma omp parallel for private(i) private(r) private(c)
		for (i = 0; i < x->rows * x->cols; i++)
		{
			r = i / x->cols;
			c = i % x->cols;

			a->values[r][5 * c] = a->values[r][5 * c + 1] = a->values[r][5 * c + 2] = a->values[r][5 * c + 3] = -K1;
			a->values[r][5 * c + 4] = 4 * K1 + 1;
			m->values[r][c] = a->values[r][5 * c + 4];
			invM->values[r][c] = 1 / m->values[r][c];
		}

	// The conjugate-gradient resolution is initialized, performed and finalized for the first pass.
	
		initConjugateGradient(invM, b, x, res, p, z, &totalRho);

		for (i = 0; i < MAX_ITERS; i++)
		{
			if (iterConjugateGradient(a, m, invM, x, res, p, q, z, &totalRho, EPS))
			{
				if (rank == 0)
				{
					printf("First pass converged in %d iterations.\n", i + 1);
				}
				break;
			}
		}	
		nIters = i + 1; // The number of iterations is stored.

		finalizeConjugateGradient(p);

	// The Perona-Malik coefficients are computed. They are stored in a matrix block with the same size as the input block. The second diffusion coefficient is included in these coefficients.

		s = newEmptyMatrix(input->rows, input->cols);

		receiveEdges(x->rows, x->cols);
		sendEdges(x);

		MPI_Waitall(nNeighbours, recvRequests, statuses);

		#pragma omp parallel for private(i) private(r) private(c) private(shiftR) private(shiftC) private(g)
		for (i = 0; i < s->rows * s->cols; i++)
		{
			r = i / s->cols;
			c = i % s->cols;
			shiftR = r + offsets[0];
			shiftC = c + offsets[1];

			g = (r > 0 || gridCoords[0] == 0) ? x->values[shiftR - 1][shiftC] : recvBuffers[0][shiftC];
			g -= (r < s->rows - 1 || gridCoords[0] == gridDims[0] - 1) ? x->values[shiftR + 1][shiftC] : recvBuffers[1][shiftC];
			s->values[r][c] = g * g / 4;

			g = (c > 0 || gridCoords[1] == 0) ? x->values[shiftR][shiftC - 1] : recvBuffers[2][shiftR];
			g -= (c < s->cols - 1 || gridCoords[1] == gridDims[1] - 1) ? x->values[shiftR][shiftC + 1] : recvBuffers[3][shiftR];
			s->values[r][c] += g * g / 4;	

			s->values[r][c] = K2 / (1 + (s->values[r][c] / LAMBDA));
		}

		MPI_Waitall(nNeighbours, sendRequests, statuses);

	freeMatrix(m);
	freeMatrix(invM);
	freeMatrix(a);
	freeMatrix(b);
	freeMatrix(res);
	freeMatrix(p);
	freeMatrix(q);
	freeMatrix(z);
	freeMatrix(x);

	// The matrices for the second pass are initialized. This pass solves the diffusion with Perona-Malik coefficients on a domain of the same size as the global input.

	x = newEmptyMatrix(input->rows, input->cols);
	delayedFreeMatrix(x);
	a = newEmptyMatrix(x->rows, 5 * x->cols);
	m = newEmptyMatrix(x->rows, x->cols);
	invM = newEmptyMatrix(x->rows, x->cols);	
	res = newEmptyMatrix(x->rows, x->cols);
	z = newEmptyMatrix(x->rows, x->cols);
	q = newEmptyMatrix(x->rows, x->cols);
	p = newEmptyMatrix(x->rows, x->cols);

	// The five-point stencil coefficients for the second pass are computed as well as the diagonal preconditioner and its inverse.

		receiveEdges(s->rows, s->cols);
		sendEdges(s);
		
		MPI_Waitall(nNeighbours, recvRequests, statuses);

		#pragma omp parallel for private(i) private(r) private(c) private(g)
		for (i = 0; i < x->rows * x->cols; i++)
		{
			r = i / x->cols;
			c = i % x->cols;

			a->values[r][5 * c] = -s->values[r][c];
			a->values[r][5 * c + 1] = -s->values[r][c];
			a->values[r][5 * c + 2] = -s->values[r][c];
			a->values[r][5 * c + 3] = -s->values[r][c];
			a->values[r][5 * c + 4] = 4 * s->values[r][c] + 1;

			m->values[r][c] = a->values[r][5 * c + 4];
			invM->values[r][c] = 1 / m->values[r][c];

			if (!((r  == 0 && gridCoords[0] == 0) || (r  == x->rows - 1 && gridCoords[0] == gridDims[0] - 1)))
			{
				g = ((r > 0 ? s->values[r - 1][c] : recvBuffers[0][c]) - (r < s->rows - 1 ? s->values[r + 1][c] : recvBuffers[1][c])) / 2;
				a->values[r][5 * c] += -g / 2;
				a->values[r][5 * c + 1] += g / 2;
			}
			if (!((c  == 0 && gridCoords[1] == 0) || (c  == x->cols - 1 && gridCoords[1] == gridDims[1] - 1)))
			{
				g = ((c > 0 ? s->values[r][c - 1] : recvBuffers[2][r]) - (c < s->cols - 1 ? s->values[r][c + 1] : recvBuffers[3][r])) / 2;
				a->values[r][5 * c + 2] += -g / 2;
				a->values[r][5 * c + 3] += g / 2;				
			}
		}

		MPI_Waitall(nNeighbours, sendRequests, statuses);
		
	// The conjugate-gradient resolution is initialized, performed and finalized for the first pass.		

		initConjugateGradient(invM, input, x, res, p, z, &totalRho);

		for (i = 0; i < MAX_ITERS; i++)
		{
			if (iterConjugateGradient(a, m, invM, x, res, p, q, z, &totalRho, EPS))
			{
				if (rank == 0)
				{
					printf("Second pass converged in %d iterations.\n", i + 1);
				}
				break;
			}

		}	
		nIters += i + 1; // The number of iterations is added to the total.

		finalizeConjugateGradient(p);
	

	free(recvBuffers[0]);
	free(recvBuffers[1]);
	free(recvBuffers[2]);
	free(recvBuffers[3]);	
	free(sendBuffers[0]);
	free(sendBuffers[1]);
	freeMatrix(m);
	freeMatrix(invM);
	freeMatrix(a);
	freeMatrix(res);
	freeMatrix(p);
	freeMatrix(q);
	freeMatrix(z);
}

void finalize()
{
	/*
	Finalization of the computation.
	*/
	
	int i, coords[2], row0, col0, rows, cols;
	Matrix totalOutput;
	float *sendBuffer, *gatherBuffer;	

	// One process gathers the results from all the processes, assembles them and saves the global result to a file.
	
		sendBuffer = (float *) malloc(maxDims[0] * maxDims[1] * sizeof(float));
		memcpy(sendBuffer, *(x->values), x->rows * x->cols * sizeof(float));
		
		if (rank == 0)
		{
			gatherBuffer = (float *) malloc(maxDims[0] * maxDims[1] * size * sizeof(float));
		}
		
		MPI_Gather(sendBuffer, maxDims[0] * maxDims[1], MPI_FLOAT, gatherBuffer, maxDims[0] * maxDims[1], MPI_FLOAT, 0, cartComm);
		free(sendBuffer);
		
		if (rank == 0)
		{		
			totalOutput = newEmptyMatrix(totalInputDims[0], totalInputDims[1]);

			for (i = 0; i < size; i++)
			{			
				MPI_Cart_coords(cartComm, i, 2, coords);
				getBlockRange(coords, &row0, &col0, &rows, &cols);
				insertBufferIntoMatrix(row0, col0, rows, cols, gatherBuffer + i * maxDims[0] * maxDims[1], totalOutput);
			}
			writePgm(OUTPUT, totalOutput);
			freeMatrix(totalOutput);
			free(gatherBuffer);
		}

	MPI_Comm_free(&cartComm);
}