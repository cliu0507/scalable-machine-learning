#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include "spMatrix.h"


int main(int argc, char **argv)
{

    if(argc != 4){
        std::cout<<"No enough arugments\n";
        exit(0);
    }

    srand((unsigned)time(0));

    int n_threads;
    std::string meta_file, data_file;

    n_threads = atoi(argv[1]);
    meta_file = argv[2];
    data_file = argv[3];

    int row, col, size;

    //reading the meta file
    std::ifstream fmeta(meta_file.c_str()); 
    fmeta >> row >> col >> size; 
    
    //create a random vector
    double *X = new double[col];
    double *W = new double[col];
/*    for(int i=0;i<col;i++){
        X[i] = (double)rand()/(RAND_MAX);
        W[i] = 1;
    }
*/
    //create twos vector
    for(int i=0;i<col;i++){
        X[i] = 2.0;
        W[i] = 1;
    }

    double *Y = new double[col]();

    //create a sparse matrix
    spMatrix spA(n_threads, data_file);
    spA.initNodes(X, Y);
    spA.initEdges(W);

    //--------timing--------------------
    Galois::StatTimer T;
    T.start();
    spA.multiply();
    T.stop();

//    printf("Y[0]=26 : %0.2f, Y[122]=218 : %0.2f, Y[12312]=184 :%0.2f\n", spA.get_y(0), spA.get_y(122), spA.get_y(12312));
    printf("total running time: %f seconds\n", double(T.get())/1000);

    delete[] Y;
    delete[] X;
}
