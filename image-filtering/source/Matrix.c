#include "matrix.h"

Matrix newEmptyMatrix(int rows, int cols)
{
	/*
	Creates a new un-initialized matrix.
	*/
	
	int r;
	Matrix mat;
	float *v;

	mat = (Matrix) malloc(sizeof(struct MatrixSt));
	mat->rows = rows;
	mat->cols = cols;
	mat->values = (float **) malloc(rows * sizeof(float *));
	v = (float *) malloc(rows * cols * sizeof(float));
	for (r = 0; r < rows; r++) 
	{
		mat->values[r] = v + r * cols;
	}
	return mat;
}

void insertBufferIntoMatrix(int row0, int col0, int rows, int cols, float *buffer, Matrix matrix)
{
	/*
	Inserts the content of a buffer into a matrix.
	*/
	
	int i, r, c;

	#pragma omp parallel for private(i) private(r) private(c)
	for (i = 0; i < rows * cols; i++)
	{
		r = i / cols;
		c = i % cols;
		matrix->values[r + row0][c + col0] = buffer[i];
	}		
}

void freeMatrix(Matrix matrix)
{
	/*
	Frees a matrix.
	*/
	
	free(*(matrix->values));
	free(matrix->values);
	free(matrix);
}

void getCropMatrix( int row0, int col0, int rows, int cols, Matrix inMatrix, Matrix outMatrix )
{
	/*
	Copies a part of a matrix to another matrix.
	*/
	
	int i, r, c;

	#pragma omp parallel for private(i) private(r) private(c)
	for (i = 0; i < rows * cols; i++)
	{
		r = i / cols;
		c = i % cols;
		outMatrix->values[r][c] = inMatrix->values[r + row0][c + col0];
	}
}

void copyMatrix( Matrix inMatrix, Matrix outMatrix )
{
	/*
	Copies the content of a matrix to another matrix.
	*/
	
	memcpy(*(outMatrix->values), *(inMatrix->values), inMatrix->rows * inMatrix->cols * sizeof(float));
}

void insertMatrix( int row0, int col0, Matrix matrix, Matrix targetMatrix )
{
	/*
	Inserts the content of a matrix into another matrix.
	*/
	
	insertSubMatrix(0, 0, matrix->rows, matrix->cols, row0, col0, matrix, targetMatrix);
}

void insertSubMatrix( int row0, int col0, int rows, int cols, int targetRow0, int targetCol0, Matrix matrix, Matrix targetMatrix )
{
	/*
	Inserts a part of a matrix into another matrix.
	*/
	
	int i, r, c;

	#pragma omp parallel for private(i) private(r) private(c)
	for (i = 0; i < rows * cols; i++)
	{
		r = i / cols;
		c = i % cols;
		targetMatrix->values[r + targetRow0][c + targetCol0] = matrix->values[r + row0][c + col0];
	}
}

float dotProductMatrix(Matrix matrix0, Matrix matrix1)
{
	/*
	Returns the dot product of two matrices.
	*/
	
	float res;
	int i, r, c;

	res = 0;
	#pragma omp parallel for private(i) private(r) private(c) reduction(+:res)
	for (i = 0; i < matrix0->rows * matrix0->cols; i++)
	{
		r = i / matrix0->cols;
		c = i % matrix0->cols;
		res += matrix0->values[r][c] * matrix1->values[r][c];
	}
	return res;
}

void elementMultiplyMatrix( Matrix inMatrix, Matrix outMatrix )
{
	/*
	Multiplies element wise two matrices, storing the result in the second one.
	*/
	
	int i, r, c;

	#pragma omp parallel for private(i) private(r) private(c)
	for (i = 0; i < inMatrix->rows * inMatrix->cols; i++)
	{
		r = i / inMatrix->cols;
		c = i % inMatrix->cols;
		outMatrix->values[r][c] *= inMatrix->values[r][c];		
	}
}


void getElementMultiplyMatrix( Matrix inMatrix0, Matrix inMatrix1, Matrix outMatrix )
{
	/*
	Multiplies element wise two matrices, storing the result in a third one.
	*/
	
	int i, r, c;

	#pragma omp parallel for private(i) private(r) private(c)
	for (i = 0; i < inMatrix0->rows * inMatrix0->cols; i++)
	{
		r = i / inMatrix0->cols;
		c = i % inMatrix0->cols;
		outMatrix->values[r][c] = inMatrix1->values[r][c] * inMatrix0->values[r][c];
	}
}

void multiplyMatrix(float coef, Matrix matrix)
{
	/*
	Multiplies a matrix by a number.
	*/
	
	int i, r, c;

	#pragma omp parallel for private(i) private(r) private(c)
	for (i = 0; i < matrix->rows * matrix->cols; i++)
	{
		r = i / matrix->cols;
		c = i % matrix->cols;
		matrix->values[r][c] *= coef;
	}	
}

void getMultiplyMatrix( float coef, Matrix inMatrix, Matrix outMatrix )
{
	/*
	Multiplies a matrix by a number, storing the result in another matrix.
	*/
	
	int i, r, c;

	#pragma omp parallel for private(i) private(r) private(c)
	for (i = 0; i < inMatrix->rows * inMatrix->cols; i++)
	{
		r = i / inMatrix->cols;
		c = i % inMatrix->cols;
		outMatrix->values[r][c] = inMatrix->values[r][c] * coef;
	}	
}

void addMatrix( Matrix inMatrix, Matrix outMatrix )
{
	/*
	Adds two matrices, storing the result in the second one.
	*/
	
	int i, r, c;

	#pragma omp parallel for private(i) private(r) private(c)
	for (i = 0; i < inMatrix->rows * inMatrix->cols; i++)
	{
		r = i / inMatrix->cols;
		c = i % inMatrix->cols;
		outMatrix->values[r][c] += inMatrix->values[r][c];
	}	
}

void getAddMatrix( Matrix inMatrix0, Matrix inMatrix1, Matrix outMatrix )
{
	/*
	Adds two matrices, storing the result in a third one.
	*/
	
	int i, r, c;

	#pragma omp parallel for private(i) private(r) private(c)
	for (i = 0; i < inMatrix0->rows * inMatrix0->cols; i++)
	{
		r = i / inMatrix0->cols;
		c = i % inMatrix0->cols;
		outMatrix->values[r][c] = inMatrix0->values[r][c] + inMatrix1->values[r][c];
	}	
}

void setValueMatrix(float val, Matrix matrix)
{
	/*
	Sets a specific value to all the elements of a matrix.
	*/
	
	int i, r, c;

	#pragma omp parallel for private(i) private(r) private(c)
	for (i = 0; i < matrix->rows * matrix->cols; i++)
	{
		r = i / matrix->cols;
		c = i % matrix->cols;
		matrix->values[r][c] = val;
	}	
}

void copySubMatrixToBuffer(int row0, int col0, int rows, int cols, Matrix matrix, float *buffer)
{
	/*
	Copies a part of a matrix to a buffer.
	*/
	
	int i, r, c;

	#pragma omp parallel for private(i) private(r) private(c)
	for (i = 0; i < rows * cols; i++)
	{
		r = i / cols;
		c = i % cols;
		buffer[i] = matrix->values[r + row0][c + col0];
	}
}

float maxMatrix(Matrix matrix)
{
	/*
	Returns the maximum value of a matrix.
	*/
	
	float m, localM;	
	int i, r, c;

	m = matrix->values[0][0];
	#pragma omp parallel
	{
		localM = matrix->values[0][0];
		#pragma omp for nowait
		for (i = 0; i < matrix->rows * matrix->cols; i++)
		{
			r = i / matrix->cols;
			c = i % matrix->cols;
			localM = MAX(localM, matrix->values[r][c]);
		}
		
		#pragma omp critical
		m = MAX(m, localM);
	}
	return m;
}

float minMatrix(Matrix matrix)
{
	/*
	Returns the minimum value of a matrix.
	*/
	float m, localM;	
	int i, r, c;

	m = matrix->values[0][0];
	#pragma omp parallel
	{
		localM = matrix->values[0][0];
		#pragma omp for nowait
		for (i = 0; i < matrix->rows * matrix->cols; i++)
		{
			r = i / matrix->cols;
			c = i % matrix->cols;
			localM = MIN(localM, matrix->values[r][c]);
		}
		
		#pragma omp critical
		m = MIN(m, localM);
	}
	return m;
}

int readPgm(const char *filename, Matrix *image)
{
	/*
	Reads the content of a PGM file, divides it by 255 and stores it into an unallocated matrix. 
	*/
	
	FILE *file;
	int char1, char2, width, height, max, c1, c2, c3, r, c;

	file = fopen (filename, "rb");
	if (!file)
	{
		fprintf(stderr, "Could not open file \"%s\".\n", filename);
		return 1;
	}

	char1 = fgetc(file);
	char2 = fgetc(file);
	removeComments(file);
	c1 = fscanf(file, "%d", &width);
	removeComments(file);
	c2 = fscanf(file, "%d", &height);
	removeComments(file);
	c3 = fscanf(file, "%d", &max);

	if (char1 != 'P' || char2 != '5' || c1 != 1 || c2 != 1 || c3 != 1 || max > 255)
	{
		fprintf(stderr, "Input is not a standard raw 8-bit PGM file.\n");
		return 1;
	}

	fgetc(file);


	*image = newEmptyMatrix(height, width);
	for (r = 0; r < height; r++)
	{
		for (c = 0; c < width; c++)
		{
			(*image)->values[r][c] = ((float) fgetc(file)) / 255.0f;
		}
	}
	fclose(file);
	return 0;
}

int writePgm(const char *filename, Matrix image)
{
	/*
	Writes the content of a matrix multiplied by 255 to a PGM file.
	*/
	
	FILE *file;
	int r, c, val;

	file = fopen (filename, "wb");
	if (!file)
	{
		fprintf(stderr, "Could not open file \"%s\" for writing.\n", filename);
		return 1;
	}

	fprintf(file, "P5\n%d %d\n255\n", image->cols, image->rows);

	for (r = 0; r < image->rows; r++)
	{
		for (c = 0; c < image->cols; c++) 
		{
			val = (int) (255.0 * image->values[r][c]);
			fputc(MAX(0, MIN(255, val)), file);
		}
	}
	fclose(file);
	return 0;
}



void removeComments(FILE *file)
{
	/*
	Removes the lines starting with "#" from a file.
	*/
	int ch;

	fscanf(file," "); 
	while ((ch = fgetc(file)) == '#') 
	{
		while ((ch = fgetc(file)) != '\n'  &&  ch != EOF)
		{
		}
		fscanf(file," ");
	}
	ungetc(ch, file);
}