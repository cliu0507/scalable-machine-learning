#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <limits.h>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include "spMatrix.h"

const int offset = 1;

void normalizeV(double * V, int size);
void update_and_init(double * V, double * ans, int size);
void sort_idx(double * X, int *idx, int size);
 
int main(int argc, char *argv[])
{

    if(argc != 4){
        std::cout<<"No enough arugments\n";
        exit(0);
    }

    srand((unsigned)time(0));

    double wall_timer = omp_get_wtime();

    int n_threads;
    std::string meta_file, data_file;

    std::cout<<"----------------------------------------------"<<"\n";
    std::cout<<"exec filename: "<<argv[0]<<""<<"\n";
    std::cout<<"Processing data...";
    n_threads = atoi(argv[1]);
    meta_file = argv[2];
    data_file = argv[3];

    //set number of threads
    omp_set_num_threads(n_threads); 

    int row, col, size;

    //reading the meta file
    std::ifstream fmeta(meta_file.c_str()); 
    fmeta >> row >> col >> size; 

    //reading col size data
    int *row_size = new int[col];
    for(int i=0;i<row;i++){
        fmeta >> row_size[i];
    }

    //create a ones vector
    double *X = new double[col];
    for(int i=0;i<row;i++){
        X[i] = 1.0;
    }

    std::ifstream fdata(data_file.c_str());
    int cur_row, cur_col;
    double cur_val;

    //create a sparse matrix
    spMatrix spA(row, col, row_size);
    for(int i=0;i<size;i++){
        fdata >> cur_row;
        fdata >> cur_col;
        cur_row = cur_row - offset;
        cur_col = cur_col - offset;
        fdata >> cur_val;
        spA.add_val(cur_row, cur_col, cur_val);
    }

    //init answer array
    double *ans = new double[col]();

    printf("Done. Time spent on preprocessing data: %0.2f\n", omp_get_wtime() - wall_timer);
    int maxIter=50;
    wall_timer = omp_get_wtime();
    for(int i=0;i<maxIter;i++){
        spA.multiply(X, ans);
        normalizeV(ans, col);
        update_and_init(X, ans, col);
    }
    printf("Time it takes for %d: %0.5f\n\n", maxIter, omp_get_wtime() - wall_timer);
    double lambda = 0;
    spA.multiply(X, ans);
    for(int i=0;i<col;i++){
        lambda += X[i]*ans[i];
    }
    printf("The current lambda is %f\n", lambda);
    printf("Top 100 nodes of A:\n");
    int maxTop = 100;
    int *idx = new int[col];
    sort_idx(X, idx, col);
    printf("Rank | Index\n");
    for(int i=0;i<maxTop;i++){
    	printf("rank: %d | %d\n", i+offset, idx[i]+offset);
    }
    /*--------------------------------post processing------------------------------*/
    delete[] idx;
    delete[] ans;
    delete[] X;
    delete[] row_size;
}

void normalizeV(double * V, int size){
    double tmp_sum = 0;
    for(int i=0;i<size;i++){
        tmp_sum += V[i]*V[i];
    }
    double V_norm = sqrt(tmp_sum); 
    for(int i=0;i<size;i++){
        V[i] = V[i]/V_norm;
    }
}

void update_and_init(double * V, double * ans, int size){
    for(int i=0;i<size;i++){
        V[i] = ans[i];
        ans[i] = 0;
    }
}

void sort_idx(double * X, int *idx, int size){
    for(int i=0;i<size;i++){
	idx[i] = i;
    }
    std::sort(idx, idx+size, [&](size_t a, size_t b){return X[a]>X[b];});
}
