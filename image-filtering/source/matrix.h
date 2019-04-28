#ifndef MATRIX
#define MATRIX

#include <string.h>
#include "common.h"

typedef struct MatrixSt {
	int rows, cols; 
	float **values;
} *Matrix;


Matrix newEmptyMatrix(int rows, int cols);
void copySubMatrixToBuffer(int row0, int col0, int rows, int cols, Matrix matrix, float *buffer);
int readPgm(const char *filename, Matrix *image);
int writePgm(const char *filename, Matrix image);
void freeMatrix(Matrix matrix);
void getCropMatrix(int row0, int col0, int rows, int cols, Matrix inMatrix, Matrix outMatrix);
void copyMatrix(Matrix inMatrix, Matrix outMatrix);
void insertMatrix(int row0, int col0, Matrix matrix, Matrix targetMatrix);
void insertSubMatrix(int row0, int col0, int rows, int cols, int targetRow0, int targetCol0, Matrix matrix, Matrix targetMatrix);
float dotProductMatrix(Matrix matrix0, Matrix matrix1);
void multiplyMatrix(float coef, Matrix matrix);
void getMultiplyMatrix(float coef, Matrix inMatrix, Matrix outMatrix);
void addMatrix(Matrix inMatrix, Matrix outMatrix);
void getAddMatrix(Matrix inMatrix0, Matrix inMatrix1, Matrix outMatrix);
void setValueMatrix(float val, Matrix matrix);
void insertBufferIntoMatrix(int row0, int col0, int rows, int cols, float *buffer, Matrix matrix);
float maxMatrix(Matrix matrix);
float minMatrix(Matrix matrix);
void getElementMultiplyMatrix(Matrix inMatrix0, Matrix inMatrix1, Matrix outMatrix);
void elementMultiplyMatrix( Matrix inMatrix, Matrix outMatrix );
void removeComments(FILE *file);

#endif