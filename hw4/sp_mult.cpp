#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <limits.h>
#include <cstdio>
#include <ctime>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "spMatrix.h"

const int offset = 1;

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
    int *row_size = new int[row];
    for(int i=0;i<row;i++){
        fmeta >> row_size[i];
    }

    //create a random vector
    double *X = new double[col];
    for(int i=0;i<col;i++){
        X[i] = (double)rand()/(RAND_MAX);
    }
    //create a ones vector
    /*double *X = new double[col];
    for(int i=0;i<col;i++){
        X[i] = 2.0;
    }*/

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
    int maxIter = 10;
    wall_timer = omp_get_wtime();
    for(int i=0;i<maxIter;i++){
    	spA.multiply(X, ans);
    }
    printf("Time it takes for A.multiple(X, ans): %0.5f\n\n", (omp_get_wtime() - wall_timer)/maxIter);
 /*--------------------------------post processing------------------------------*/
    delete[] ans;
    delete[] X;
    delete[] row_size;
}
