#ifndef SP_MATRIX
#define SP_MATRIX

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cmath>

class spMatrix{
    public:
        double **myA;
        int **myIdxes;
        int *myRow_size;
        int myRow;
        int myCol;
        int mySize;
        int myCur_ridx;
        int myCur_cidx;

        //------transpose matrix----------
        int** trs_idxes;
        int* trs_col_size;
        double** trs_vals;
    public:
        spMatrix(int row, int col, int *row_size);
        ~spMatrix();
        void multiply(double* X, double* ans);
        int get_row();
        int get_col();
        int get_size();
        double get_val(int row, int col);
        void add_val(int row, int col, double val);
        void sparseHessianProduct(double* ans, double* D, double* V, double C);
        double dotProduct(double* vecOne, double* vecTwo);
        void printMat();
        void setTranspose(int* col_size, int** idxes, double** vals);
        void getDiagonalD(double* D, double* WX);
        void getWX(double* WX, double* W, int* Y);
};

#endif
