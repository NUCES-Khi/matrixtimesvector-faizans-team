#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define TILE_SIZE 32

void matVecMult(double **matrix, double *vector, double *result, int rows, int cols) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i += TILE_SIZE) {
        for (int j = 0; j < cols; j += TILE_SIZE) {
            for (int ii = i; ii < i + TILE_SIZE; ii++) {
                result[ii] = 0.0;
                for (int jj = j; jj < j + TILE_SIZE; jj++) {
                    result[ii] += matrix[ii][jj] * vector[jj];
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s matrix_size\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    if (size <= 0) {
        printf("Invalid matrix size\n");
        return 1;
    }

    // Allocate memory for matrix, vector, and result
    double **matrix = (double **)malloc(size * sizeof(double *));
    double *vector = (double *)malloc(size * sizeof(double));
    double *result = (double *)malloc(size * sizeof(double));

    // Initialize matrix and vector with random values
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double *)malloc(size * sizeof(double));
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX; // Random value between 0 and 1
        }
        vector[i] = (double)rand() / RAND_MAX;
    }

    // Perform matrix-vector multiplication
    double start_time = omp_get_wtime();
    matVecMult(matrix, vector, result, size, size);
    double end_time = omp_get_wtime();

    // Calculate execution time
    double execution_time = end_time - start_time;

    // Print result and execution time
    printf("Result:\n");
    for (int i = 0; i < size; i++) {
        printf("%f\n", result[i]);
    }
    printf("Execution time: %f seconds\n", execution_time);

    // Free allocated memory
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
    free(vector);
    free(result);

    return 0;
}
