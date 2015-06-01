#ifndef SP_MATRIX
#define SP_MATRIX

#include <algorithm>
#include <cstdlib>
#include <cstdio>

class spMatrix{
    private:
        double **myA;
        double myV;
        double myAlpha;
        int **myIdxes;
        int *myRow_size;
        int myRow;
        int myCol;
        int mySize;
        int myCur_ridx;
        int myCur_cidx;
    public:
        spMatrix(int row, int col, int *row_size, double alpha);
        ~spMatrix();
        void multiply(double* X, double* ans);
        int get_row();
        int get_col();
        int get_size();
        double get_val(int row, int col);
        void setWeight();
        //assume elements are added in order, row after row
        void add_val(int row, int col, double val);
};

#endif
