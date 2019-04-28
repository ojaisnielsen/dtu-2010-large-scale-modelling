#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

void logResults(const char* filename, const char* method, int gridSize, int domainSize, int nIter, double time)
{
	/*
	Adds to the CSV file "filename" the number of iterations "nIter" and the time "time" with the method "method", a processe grid of size "gridSize" and a domain of size "domainSize".
	*/

	FILE* logFile;

	logFile = fopen(filename, "a");
	fprintf(logFile, "%s, %d, %d, %d, %f\n", method, domainSize, gridSize, nIter, time);
	fclose(logFile);
}

void readPgm(const char* filename, float** data, int* width, int* height)
{
	/*
	Reads the PGM image file "filename" writes its content to "data", its width to "width" and its height to "height".
	*/

	unsigned char* rawData;
	FILE* pgm;
	int i;

	pgm = fopen(filename, "rb");
	fscanf(pgm, "P5 %d %d 255\n", width, height);
	rawData = (unsigned char*) malloc((*width) * (*height) * sizeof(char));
	fread(rawData, sizeof(char), (*width) * (*height), pgm);
	fclose(pgm);
	*data = (float*) malloc((*width) * (*height) * sizeof(float));
	for (i = 0; i < (*width) * (*height); i++)
	{
		(*data)[i] = (float) rawData[i];
	}
	free(rawData);
}

void writePgm(const char* filename, const float* data, int width, int height)
{
	/*
	Writes the data in "data" to the PGM image file "filename" with width "width" and height "height".
	*/

	unsigned char* rawData;
	FILE* pgm;
	int i;

	rawData = (unsigned char*) malloc(width * height * sizeof(char));
	for (i = 0; i < width * height; i++)
	{
		rawData[i] = (unsigned char) data[i];
	}
	pgm = fopen(filename, "wb");
	fprintf(pgm, "P5 %d %d 255\n", width, height);
	fwrite(rawData, sizeof(char), width * height, pgm);
	fclose(pgm);
	free(rawData);
}

void cropMatrix(const float* matrix, float* croppedMatrix, int x0, int xSup, int y0, int ySup, int xSize, int xStep, int yStep)
{
	/*
	Extract the data which first matrix coordinate lies between "x0" and "xSup" and wich second lies between "y0" and "ySup", every "xStep" columns and "yStep" lines from matrix "matrix" which width is "xSize" and stores it in "croppedMatrix".
	*/

	int x, y;

	for (x = x0; x < xSup; x += xStep)
	{
		for (y = y0; y < ySup; y += yStep)
		{
			croppedMatrix[((x - x0) / xStep) + ((y - y0) / yStep) * ((xSup - x0) / xStep)] = matrix[x + y * xSize];
		}
	}
}

void insertBlock(const float* block, float* matrix, int x0, int xSup, int y0, int ySup, int xSize)
{
	/*
	Insert the data contained in "block" in matrix "matrix" which  width is "xSize" so their first coordinate lies between "x0" and "xSup" and their second lies between "y0" and "ySup".
	*/

	int x, y;

	for (x = x0; x < xSup; x++)
	{
		for (y = y0; y < ySup; y++)
		{
			matrix[x + y * xSize] = block[x - x0 + (y - y0) * (xSup - x0)];
		}
	}
}

void getBlockRange(const int* gridCoords, int domainSize, int gridSize, int* x0, int* xSup, int* y0, int* ySup)
{
	/*
	Computes the discretized domain coordinates limits "x0", "xSup", "y0" and "ySup" which define the range associated with the process which Cartesian coordinates are "gridCoords".
	*/

	int d;

	d = (domainSize / gridSize) + (domainSize % gridSize);
	*x0 = gridCoords[0] * d;
	*xSup = MIN((gridCoords[0] + 1) * d, domainSize);
	*y0 = gridCoords[1] * d;
	*ySup = MIN((gridCoords[1] + 1) * d, domainSize);
}

void initVectColorBuffers(int vectX0, int vectXSup, int vectY0, int vectYSup, int nColors, int* buffersSizes, float** buffers, int* colorOffsetX, int* colorOffsetY)
{
	/*
	Takes coordinates "vectX0", "vectXSup", "vectY0" and "vectYSup" that define a vector and a number of colors "nColors" and returns:
	- bufferSizes: for each color, the number of occurences in the vector;
	- buffers: for each color, a buffer with the size previously determined;
	- colorOffsetX: for each color, the first x-coordinate where the color apears in the vector;
	- colorOffsetY: for each color, the first y-coordinate where the color apears in the vector.
	*/

	int firstColor, size, color;

	firstColor = getColor(vectX0, vectY0, nColors);
	size = (vectXSup - vectX0) * (vectYSup - vectY0);

	for (color = 0; color < nColors; color++)
	{
		int currentColor;
		if (colorOffsetX != NULL && colorOffsetY != NULL)
		{
			colorOffsetX[color] = (vectYSup - vectY0) == 1 ? (firstColor + nColors - color) % nColors : 0;
			colorOffsetY[color] = (vectXSup - vectX0) == 1 ? (firstColor + nColors - color) % nColors : 0;
		}

		currentColor = (firstColor + color) % nColors;
		buffersSizes[currentColor] = (size + nColors - color - 1) / nColors;
		buffers[currentColor] = (float*) malloc(buffersSizes[currentColor] * sizeof(float));
	}
}

int getColor(int x, int y, int nColors)
{	
	/*
	Takes coordinates "x" and "y" and number of colors "nColors" and returns the color associated with the coordinates.
	*/

	return ((y % nColors) + (x % nColors)) % nColors;
}

int getFirstComputedColor(const int* gridCoords, int nColors)
{
	/*
	Takes process cartesian coordinates  "gridCoords" and number of colors "nColors" and returns the color that the process should start computing the values for at each iteration.
	*/

	return ((gridCoords[0] + gridCoords[1]) % nColors);
}


void getNeighboursSidesRanges(int x0, int xSup, int y0, int ySup, int xSize, int ySize, int* nNeighbours, int* neighboursDirections, int* neighboursOffsets, int* top, int* bottom, int* left, int* right, int* inSidesX0, int* inSidesXSup, int* inSidesY0, int* inSidesYSup, int* outSidesX0, int* outSidesXSup, int* outSidesY0, int* outSidesYSup)
{
	/*
	Takes coordinates "x0", "xSup", "y0" and "ySup" that define a block, "xSize" and "ySize" the width and height of the discretized domain and returns:
	- nNeighbours: the number of neighbouring blocks to the block;
	- neighboursDirections: for each neighbouring block, the direction in which it lies: 0 for horizontal and 1 for vertical;
	- neighboursOffsets: for each neighbouring block: -1 if it lies before the block or 1 if it lies after the block, along the previously determined direction; 
	- top, bottom, left, right: the index of the neighbouring block that is respectively on above, beyond, to the left and to the right of th block, if it exists;
	- inSidesX0, inSidesXSup, inSidesY0, inSidesYSup: for each neighbour, the coordinates that define the vector of the block that is touches neighbouring blobk;
	- outSidesX0, outSidesXSup, outSidesY0, outSidesYSup: for each neighbour, the coordinates that define the vector of the neighbouring block that touches the blobk.
	*/

	int neighbour;

	(*nNeighbours) = 0;
	if (y0 > 0)
	{
		neighboursDirections[(*nNeighbours)] = 1;
		neighboursOffsets[(*nNeighbours)] = -1;
		(*top) = (*nNeighbours);
		(*nNeighbours)++;
	}
	if (ySup < ySize - 1)
	{
		neighboursDirections[(*nNeighbours)] = 1;
		neighboursOffsets[(*nNeighbours)] = 1;
		(*bottom) = (*nNeighbours);
		(*nNeighbours)++;
	}
	if (x0 > 0)
	{
		neighboursDirections[(*nNeighbours)] = 0;
		neighboursOffsets[(*nNeighbours)] = -1;
		(*left) = (*nNeighbours);
		(*nNeighbours)++;
	}
	if (x0 < xSize - 1)
	{
		neighboursDirections[(*nNeighbours)] = 0;
		neighboursOffsets[(*nNeighbours)] = 1;
		(*right) = (*nNeighbours);
		(*nNeighbours)++;
	}

	for (neighbour = 0; neighbour < (*nNeighbours); neighbour++)
	{
		inSidesX0[neighbour] = neighbour == (*right) ? xSup - 1 : x0;
		inSidesXSup[neighbour] = neighbour == (*left) ? x0 + 1 : xSup;
		inSidesY0[neighbour] = neighbour == (*bottom) ? ySup - 1 : y0;
		inSidesYSup[neighbour] = neighbour == (*top) ? y0 + 1 : ySup;
		
		outSidesX0[neighbour] = neighboursDirections[neighbour] == 0 ? inSidesX0[neighbour] + neighboursOffsets[neighbour] : inSidesX0[neighbour];
		outSidesXSup[neighbour] = neighboursDirections[neighbour] == 0 ? inSidesXSup[neighbour] + neighboursOffsets[neighbour] : inSidesXSup[neighbour];
		outSidesY0[neighbour] = neighboursDirections[neighbour] == 1 ? inSidesY0[neighbour] + neighboursOffsets[neighbour] : inSidesY0[neighbour];
		outSidesYSup[neighbour] = neighboursDirections[neighbour] == 1 ? inSidesYSup[neighbour] + neighboursOffsets[neighbour] : inSidesYSup[neighbour];
	}
}


int getColumnOffsetColor(int x, int y0, int currentColor, int nColors)
{
	/*
	Takes a point coordinates "x" and "y0" and returns the distance to the next (or same) point along the column that is associated to the color "currentColor" given "nColors" colors;
	*/

	return (getColor(x, y0, nColors) + nColors - currentColor) %nColors;
}


void drawDist(float* data, float (*dist)(int, int), int xSize, int ySize)
{
	/*
	Stores the values of the function "dist" on the domain of dimensions "xSize" and "ySize" in "data".
	*/

	int x, y;

	for (x = 0; x < xSize; x++)
	{
		for (y = 0; y < ySize; y++)
		{
			data[x + y * xSize] = dist(x, y);
		}
	}
}

int initSquareComm(MPI_Comm* cartComm, int* gridWidth, int* coords)
{
	/*
	If possible, creates a square cartesian communicator "cartComm"  and returns its width "gridWidth" and the coordinates of the current process "coords".
	*/

	int rank, size, dims[2], periods[2];

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	*gridWidth = (int) sqrt((float) size);
	if (size != (*gridWidth) * (*gridWidth))
	{
		return 1;
	}

	dims[0] = dims[1] = *gridWidth;
	periods[0] = periods[1] = 0;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, cartComm);
	MPI_Comm_rank(*cartComm, &rank);
	MPI_Cart_coords(*cartComm, rank, 2, coords);

	return 0;
}

void fillArray(float* data, float value, int size)
{
	/*
	Fills the array "data" of size "size" with the value "value".
	*/

	int i;

	for (i = 0; i < size; i++)
	{
		data[i] = value;
	}
}