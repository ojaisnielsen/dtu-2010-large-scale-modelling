#ifndef UTILS
#define UTILS

#include <mpi.h>

#define MAX(a, b) a > b ? a : b
#define MIN(a, b) a < b ? a : b

void logResults(const char* filename, const char* method, int gridSize, int domainSize, int nIter, double time);
int initSquareComm(MPI_Comm* cartComm, int* gridWidth, int* coords);
void readPgm(const char* filename, float** data, int* width, int* height);
void writePgm(const char* filename, const float* data, int width, int height);
void cropMatrix(const float* matrix, float* croppedMatrix, int x0, int xSup, int y0, int ySup, int xSize, int xStep, int yStep);
void insertBlock(const float* block, float* matrix, int x0, int xSup, int y0, int ySup, int xSize);
void getBlockRange(const int* gridCoords, int domainSize, int gridSize, int* x0, int* xSup, int* y0, int* ySup);
void initVectColorBuffers(int vectX0, int vectXSup, int vectY0, int vectYSup, int nColors, int* buffersSizes, float** buffers, int* colorOffsetX, int* colorOffsetY);
int getColor(int x, int y, int nColor);
int getFirstComputedColor(const int* gridCoords, int nColors);
void getBlockSidesRanges(int blockX0, int blockXSup, int blockY0, int blockYSup, int* sidesX0, int* sidesXSup, int* sidesY0, int* sidesYSup);
void getNeighboursSidesRanges(int x0, int xSup, int y0, int ySup, int xSize, int ySize, int* nNeighbours, int* neighboursDirections, int* neighboursOffsets, int* top, int* bottom, int* left, int* right, int* inSidesX0, int* inSidesXSup, int* inSidesY0, int* inSidesYSup, int* outSidesX0, int* outSidesXSup, int* outSidesY0, int* outSidesYSup);
int getColumnOffsetColor(int x, int y0, int currentColor, int nColors);
void drawDist(float* data, float (*dist)(int, int), int xSize, int ySize);
void fillArray(float* data, float value, int size);

#endif