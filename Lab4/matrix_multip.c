//
// Created by artur on 22.11.18.
//
#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <time.h>
#include <stdlib.h>
#include <time.h>

// A is N1 x N3, B is N3xN2, result is N1xN2
void naive_multiplication(double **A, double **B, double **C, int N1, int N2, int N3)
{

    for (int i = 0; i < N1; i++)
    {
        for (int j = 0; j < N2; j++)
        {
            for (int k = 0; k < N3; k++)
            {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }
}

// A is N1 x N3, B is N3xN2, result is N1xN2
// Now we use different iteration to make less "jumps" in memory
void better_multiplication(double **A, double **B, double **C, int N1, int N2, int N3)
{

    for (int i = 0; i < N1; i++)
    {
        for (int k = 0; k < N2; k++)
        {
            for (int j = 0; j < N3; j++)
            {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }
}

void gsl_multiplication(double *A_ptr, double *B_ptr, double *C_ptr, int N1, int N2, int N3)
{

    gsl_matrix_view A = gsl_matrix_view_array(A_ptr, N1, N3);
    gsl_matrix_view B = gsl_matrix_view_array(B_ptr, N3, N2);
    gsl_matrix_view C = gsl_matrix_view_array(C_ptr, N1, N2);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &A.matrix, &B.matrix, 0.0, &C.matrix);
}

int main(int argc, char const *argv[])
{
    clock_t start_time, end_time;
    srand(time(NULL));
    printf("Creatin CSV results file ...\n");
    FILE *fp=fopen("multiplication.csv","w+");
    printf("File created\n Making measurements ...\n");
    fprintf(fp,"size,naive,better,blas\n");

    for(int i = 100; i <= 1000; i = i + 100)
    {
        printf("Size: %dx%d\n", i, i);
        fprintf(fp,"%d", i);

        double **A = calloc(i, sizeof(double *));
        double **B = calloc(i, sizeof(double *));
        double **C = calloc(i, sizeof(double *));

        double *A_ptr = calloc(i*i, sizeof(double));
        double *B_ptr = calloc(i*i, sizeof(double));
        double *C_ptr = calloc(i*i, sizeof(double));

        for(int j =0; j<i; j++)
        {
            A[j] = calloc(i,sizeof(double));
            B[j] = calloc(i,sizeof(double));
            C[j] = calloc(i,sizeof(double));
        }

        for(int j =0; j<i; j++)
        {
            for(int k=0; k<i; k++)
            {
                A[j][k] = (double)rand()/RAND_MAX;
                B[j][k] = (double)rand()/RAND_MAX;
                C[j][k] = 0.0;
            }
        }

        //Arrays for blas
        for(int j=0; j<i*i; j++)
        {
            A_ptr[j] = (double)rand()/RAND_MAX;
            B_ptr[j] = (double)rand()/RAND_MAX;
            C_ptr[j] = 0.0;
        }

        start_time = clock();
        naive_multiplication(A, B, C, i, i, i);
        end_time = clock();
        fprintf(fp,",%f", ((double) (end_time - start_time)) / CLOCKS_PER_SEC);

        start_time = clock();
        better_multiplication(A, B, C, i, i, i);
        end_time = clock();
        fprintf(fp,",%f", ((double) (end_time - start_time)) / CLOCKS_PER_SEC);

        start_time = clock();
        gsl_multiplication(A_ptr, B_ptr, C_ptr, i, i, i);
        end_time = clock();
        fprintf(fp,",%f\n", ((double) (end_time - start_time)) / CLOCKS_PER_SEC);



        for (int j = 0; j < i; j++) {
            free(A[i]);
            free(B[i]);
            free(C[i]);
        }

        free(A);
        free(B);
        free(C);
        free(A_ptr);
        free(B_ptr);
        free(C_ptr);

        printf(" %dx%d finished\n", i, i);
    }

    fclose(fp);

    return 0;
}

