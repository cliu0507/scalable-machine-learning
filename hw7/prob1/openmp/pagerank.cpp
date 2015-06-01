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

    //initialize R_i to be 1/col
    double *R = new double[col];
    for(int i=0;i<row;i++){
        R[i] = 1.0/col;
    }

    //set alpha
    double alpha=0.15;

    std::ifstream fdata(data_file.c_str());
    int cur_row, cur_col;
    double cur_val;

    //create a sparse matrix
    spMatrix spA(row, col, row_size, alpha);
    for(int i=0;i<size;i++){
        fdata >> cur_row;
        fdata >> cur_col;
        cur_row = cur_row - offset;
        cur_col = cur_col - offset;
        fdata >> cur_val;
        spA.add_val(cur_row, cur_col, cur_val);
    }

    //set weight for A in spA
    spA.setWeight();

    //init answer array
    double *ans = new double[col]();

    printf("Done. Time spent on preprocessing data: %0.2f\n", omp_get_wtime() - wall_timer);

    //set max iteration
    int maxIter=100;

    wall_timer = omp_get_wtime();
    for(int i=0;i<maxIter;i++){
        spA.multiply(R, ans);
        //normalizeV(ans, col);
        update_and_init(R, ans, col);
    }
    printf("# of threads %d, running time for %d iterations is %0.5f\n\n", n_threads, maxIter, omp_get_wtime() - wall_timer);
    int maxTop = 10;
    int *idx = new int[col];
    sort_idx(R, idx, col);
    printf("Rank | Index\n");
    for(int i=0;i<maxTop;i++){
    	printf("rank: %d | %d | %f\n", i+offset, idx[i], R[idx[i]]);
    }
    /*--------------------------------post processing------------------------------*/
    delete[] idx;
    delete[] ans;
    delete[] R;
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
