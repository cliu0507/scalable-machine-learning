#include "spMatrix.h"

spMatrix::spMatrix(int row, int col, int *row_size, double alpha){
    this->myRow = row;
    this->myCol = col;
    this->myA = new double*[row];
    this->myIdxes = new int*[row];
    this->myRow_size = new int[row];
    this->mySize = 0;
    this->myCur_ridx = 0;
    this->myCur_cidx = 0;
    this->myAlpha = alpha;
    this->myV = alpha/col;
    for(int i=0;i<myRow;i++){
        int tmp_size = row_size[i];
        myA[i] = new double[tmp_size];
        myIdxes[i] = new int[tmp_size];
        myRow_size[i] = tmp_size;
    } 
}

spMatrix::~spMatrix(){
    for(int i=0;i<myRow;i++){
        delete[] myA[i];
        delete[] myIdxes[i];
    }
    delete[] myA;
    delete[] myIdxes;
    delete[] myRow_size;
}

void spMatrix::multiply(double* X, double* ans){
    int chuck = 5;
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, chuck) nowait
        for(int i=0;i<myRow;i++){
            int * row_idx = myIdxes[i];
            double * row_val = myA[i];
            int srow = myRow_size[i];
            double *ansi = ans+i;
            for(int j=0;j<srow;j++){
                int idx = row_idx[j];
                *ansi += row_val[j]*X[idx];
            }
            *ansi += myV;
        }
    }
}

int spMatrix::get_row(){
    return myRow;
}

int spMatrix::get_col(){
    return myCol;
}

int spMatrix::get_size(){
    return mySize;
}

double spMatrix::get_val(int row, int col){
    int idx = std::distance(myIdxes[row], std::find(myIdxes[row], myIdxes[row]+myRow_size[row], col));
    return myA[row][idx];
}

void spMatrix::add_val(int row, int col, double val){
   if(myCur_ridx != row){
       myCur_ridx += 1; 
       myCur_cidx = 0;
   }
   myIdxes[myCur_ridx][myCur_cidx] = col;
   myA[myCur_ridx][myCur_cidx] = val;
   myCur_cidx += 1; 
}

//reset weight for page rank
void spMatrix::setWeight(){

    double scale = (1-myAlpha);
    double *D = new double[myCol]();
    for(int i=0;i<myRow;i++){
        int * row_idx = myIdxes[i];
        double * row_val = myA[i];
        int srow = myRow_size[i];
        for(int j=0;j<srow;j++){
             D[i] += row_val[j];
             row_val[j] = row_val[j] * scale;
        }
    }

    for(int i=0;i<myRow;i++){
        int * row_idx = myIdxes[i];
        double * row_val = myA[i];
        int srow = myRow_size[i];
        for(int j=0;j<srow;j++){
            int idx = row_idx[j];
            row_val[j] = row_val[j] * 1.0/D[idx];
        }
    }

    delete[] D;
}
