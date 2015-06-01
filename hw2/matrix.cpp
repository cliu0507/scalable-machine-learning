#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define M 500
#define N 500

int main(int argc, char **argv){
    int i, j, k;
    double sum;
    double **A, **B, **C;
    int nthreads = 3;
    double wall_timer;
    int maxiter= 10;
    int iter_i;

    A = (double **)malloc(M*sizeof(double *));
    B = (double **)malloc(M*sizeof(double *));
    C = (double **)malloc(M*sizeof(double *));

    for(i=0;i<M;i++){
        A[i] = (double*)malloc(N*sizeof(double));
        B[i] = (double*)malloc(N*sizeof(double));
        C[i] = (double*)malloc(N*sizeof(double));
    }

    for(i=0;i<M;i++){
        for(j=0;j<N;j++){
            A[i][j] = i+j;
            B[i][j] = i*j;
            C[i][j] = 0;
        }
    }

    printf("-----------------------------------------\n");
    printf("Using number of threads: %d\n", nthreads);
    //Without open mp
    wall_timer = omp_get_wtime();
    for(iter_i=0;iter_i<maxiter;iter_i++){
        for(i=0;i<M;i++){
            for(j=0;j<N;j++){
                sum = 0;
                for(k=0;k<M;k++){
                    sum += A[i][k]*B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }
    printf("average running time without using open mp: %0.3f over %d iterations.\n", (omp_get_wtime() -
    wall_timer)/maxiter, maxiter);
   
    //with outer parallelization on the for loop
    wall_timer = omp_get_wtime();
    for(iter_i=0;iter_i<maxiter;iter_i++){
        #pragma omp parallel shared(A,B,C) private(i, j, k, sum) num_threads(nthreads)
        {
            #pragma omp for
            for(i=0;i<M;i++){
                for(j=0;j<N;j++){
                    sum = 0;
                    for(k=0;k<M;k++){
                        sum += A[i][k]*B[k][j];
                    }
                    C[i][j] = sum;
                }
            }

        }
    }
    printf("average running time with parallelization on the outer loop: %0.3f over %d iterations\n", (omp_get_wtime() -
    wall_timer)/maxiter, maxiter);

    //with inner parallelization on the for loop
    wall_timer = omp_get_wtime();
    for(iter_i=0;iter_i<maxiter;iter_i++){
        #pragma omp parallel shared(A,B,C) private(i, j, k, sum) num_threads(nthreads)
        {
            for(i=0;i<M;i++){
                #pragma omp for
                for(j=0;j<N;j++){
                    sum = 0;
                    for(k=0;k<M;k++){
                        sum += A[i][k]*B[k][j];
                    }
                    C[i][j] = sum;
                }
            }
        }
    }
    printf("average running time with parallelization on the middle loop: %0.3f over %d iterations\n", (omp_get_wtime()
    - wall_timer)/maxiter, maxiter);

    //with both inner and outer for loop
    wall_timer = omp_get_wtime();
    for(iter_i=0;iter_i<maxiter;iter_i++){
        #pragma omp parallel shared(A,B,C) private(i, j, k, sum) num_threads(nthreads)
        {
            #pragma omp for collapse(2)
            for(i=0;i<M;i++){
                for(j=0;j<N;j++){
                    sum = 0;
                    for(k=0;k<M;k++){
                        sum += A[i][k]*B[k][j];
                    }
                    C[i][j] = sum;
                }
            }
        }
    }
    printf("Running time with parallelization on the both loops: %0.3f over %d iterations\n", (omp_get_wtime() -
    wall_timer)/maxiter, maxiter);

    printf("------------------------------------------\n");
}
