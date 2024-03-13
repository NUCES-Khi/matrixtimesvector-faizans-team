



#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void matVecMult(double **matrix, double *vector, double *result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i][j] * vector[j];
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

    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Ensure number of processes matches matrix size
    if (size % num_procs != 0) {
        if (rank == 0)
            printf("Matrix size must be divisible by the number of processes\n");
        MPI_Finalize();
        return 1;
    }

    int local_size = size / num_procs;

    // Allocate memory for matrix, vector, and local result
    double **matrix = (double **)malloc(local_size * sizeof(double *));
    double *vector = (double *)malloc(size * sizeof(double));
    double *local_result = (double *)malloc(local_size * sizeof(double));

    // Initialize matrix and vector with random values
    srand(time(NULL));
    for (int i = 0; i < local_size; i++) {
        matrix[i] = (double *)malloc(size * sizeof(double));
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX; // Random value between 0 and 1
        }
        vector[i] = (double)rand() / RAND_MAX;
    }

    // Scatter matrix rows and broadcast vector
    MPI_Bcast(vector, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(matrix[0], local_size * size, MPI_DOUBLE, matrix[0], local_size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform local matrix-vector multiplication
    matVecMult(matrix, vector, local_result, local_size, size);

    // Gather local results to root process
    double *result = NULL;
    if (rank == 0) {
        result = (double *)malloc(size * sizeof(double));
    }
    MPI_Gather(local_result, local_size, MPI_DOUBLE, result, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    if (rank == 0) {
        // Print result
        printf("Result:\n");
        for (int i = 0; i < size; i++) {
            printf("%f\n", result[i]);
        }
        free(result);
    }

    // Free allocated memory
    for (int i = 0; i < local_size; i++) {
        free(matrix[i]);
    }
    free(matrix);
    free(vector);
    free(local_result);

    return 0;
}
