#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include "pr_spMatrix.h"


void sort_idx(double *X, int *idx, size_t size);

int main(int argc, char **argv)
{

    if(argc != 4){
        std::cout<<"No enough arugments\n";
        exit(0);
    }

    double alpha = 0.15;
    int maxIter = 50;
    int n_threads;
    std::string meta_file, data_file;

    n_threads = atoi(argv[1]);
    meta_file = argv[2];
    data_file = argv[3];

    int row, col, size;

    //reading the meta file
    std::ifstream fmeta(meta_file.c_str()); 
    fmeta >> row >> col >> size; 
    
    double *R = new double[col];
    for(int i=0;i<col;i++){
        R[i] = (double)1/col;
    }

    double *Y = new double[col]();
    double *pM = new double[col]();

    //create a sparse matrix
    pr_spMatrix spA(n_threads, data_file);
    spA.getpM(pM, alpha);
    spA.initNodes(R, Y);
    spA.initEdges(pM);
    spA.initV(alpha, col);

    //--------timing--------------------
    Galois::StatTimer T;
    T.start();
    for(size_t i=0;i<maxIter;i++){
        spA.pagerank_multiply();
    }
    T.stop();


    int maxTop = 10;
    spA.getR(R);

    int* idx = new int[col];
    sort_idx(R, idx, col);

    for(int i=0;i<maxTop;i++){
        printf("rank: %d | %d | %0.15f\n", i+1, idx[i], R[idx[i]]); 
    }

    printf("total running time: %f seconds\n", double(T.get())/1000);

    delete[] idx;
    delete[] pM;
    delete[] Y;
    delete[] R;
}

void sort_idx(double *X, int *idx, size_t size){
    for(size_t i=0; i<size; i++){
        idx[i] = i;
    }
    std::sort(idx, idx+size, [&](size_t a, size_t b){return X[a]>X[b];});
}
