#include "spMatrix.h"

const int chuck = 100;

spMatrix::spMatrix(int row, int col, int *row_size){
    this->myRow = row;
    this->myCol = col;
    this->myA = new double*[row];
    this->myIdxes = new int*[row];
    this->myRow_size = new int[row];
    this->mySize = 0;
    this->myCur_ridx = 0;
    this->myCur_cidx = 0;
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
//    #pragma omp parallel
    {
//        #pragma omp for schedule(dynamic, chuck) nowait
        for(int i=0;i<myRow;i++){
            int * row_idx = myIdxes[i];
            double * row_val = myA[i];
            int srow = myRow_size[i];
            double *ansi = ans+i;
            for(int j=0;j<srow;j++){
                int idx = row_idx[j];
                *ansi += row_val[j]*X[idx];
            }
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

void spMatrix::setTranspose(int* col_size, int** idxes, double** vals){
    trs_col_size = col_size;
    trs_idxes = idxes;
    trs_vals = vals;
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

void spMatrix::sparseHessianProduct(double* ans, double* D, double* V, double C){

    #pragma omp parallel for
    for(int i=0;i<myCol;i++){
        ans[i] = 0;
    }

    double *partOne = new double[myRow]();
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, chuck) nowait
        for(int i=0;i<myRow;i++){
            int * row_idx = myIdxes[i];
            double * row_val = myA[i];
            int srow = myRow_size[i];
            double *pAns = partOne+i;
            for(int j=0;j<srow;j++){
                int idx = row_idx[j];
                *pAns += row_val[j]*V[idx];
            }
            *pAns *= D[i];
        }
    }
        #pragma omp parallel
     {
        #pragma omp for schedule(dynamic, chuck) nowait
        for(int i=0;i<myCol;i++){
            int * col_idx = trs_idxes[i];
            double * col_val = trs_vals[i];
            int scol = trs_col_size[i];
            double* pAns = ans + i;
            for(int j=0;j<scol;j++){
                int idx = col_idx[j];
                *pAns += col_val[j]*partOne[idx];
            }        
        }
    }

    #pragma omp parallel for
    for(int i=0;i<myCol;i++){
        ans[i] = C*ans[i] + V[i];
    }
    delete[] partOne;
}

void spMatrix::getDiagonalD(double* D, double* WX){
//    #pragma omp parallel
    {
//        #pragma omp for schedule(dynamic, chuck) nowait
        for(int i=0;i<myRow;i++){
            D[i] = WX[i]*(1 - WX[i]);
        }
    }
}

void spMatrix::getWX(double* WX, double* W, int* Y){
//    #pragma omp parallel
    {
//        #pragma omp for schedule(dynamic, chuck) nowait
        for(int i=0;i<myRow;i++){
            int * row_idx = myIdxes[i];
            double * row_val = myA[i];
            int srow = myRow_size[i];
            double dotVal = 0.0;
            for(int j=0;j<srow;j++){
                int idx = row_idx[j];
                dotVal += row_val[j]*W[idx];
            }
            WX[i] = 1.0/(1.0+exp(0.0-dotVal*Y[i]));
        }
    }
}


void spMatrix::printMat(){
     for(int i=0;i<myRow;i++){
         int * row_idx = myIdxes[i];
         double * row_val = myA[i];
         int srow = myRow_size[i];
         printf("%d->>: ", i);
         for(int j=0;j<srow;j++){
             printf(" %d:%f ", row_idx[j], row_val[j]);
         }
         printf("\n"); 
    }
    printf("transpose()\n");
    for(int i=0;i<myCol;i++){
        int * col_idx = trs_idxes[i];
        double * col_val = trs_vals[i];
        int scol = trs_col_size[i];
        printf("%d->>: ", i);
        for(int j=0;j<scol;j++){
            printf(" %d:%f ", col_idx[j], col_val[j]);
        }
        printf("\n");
    }
}

double spMatrix::dotProduct(double* vecOne, double* vecTwo){
    double value=0;
//    #pragma omp parallel for reduction(+:value)
    for(int i=0;i<myCol;i++){
        value += vecOne[i]*vecTwo[i];
    }
    return value;
}
