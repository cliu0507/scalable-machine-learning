#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include <dirent.h>
#include <vector>
#include <limits.h>
#include <cstdio>
#include <ctime>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include "spMatrix.h"

const int chuck = 100;

typedef Eigen::Triplet<double> T;

void get_meta_infos(std::string& file, int& nrow, int& ncol, int& size);
std::vector<std::string> split_array(const std::string &s, char delim);
void extract_values(std::string &s, int& index, double& value, char delim);
int get_entry_size(const std::string& s, char delim);
void get_row_size(std::string& file_path, int* row_size);
void fill_matrix_val(std::string& file_path, spMatrix& spA, int* Y);
double get_function_value(spMatrix& X, double* W, int* Y, double C);
void get_gradient_value(double* gradient, spMatrix& X, double* W, int* Y, double* WX, double C);
void CG_Solve(double* ans, spMatrix& X, double* D, double* b, double C, int size);
double get_accuracy(spMatrix& X, double* W, int* Y, int row);
void get_col_size(std::string& file_path, int* col_size, int** idxes, double** vals, int size, int row, int col);

int main(int argc, char *argv[])
{

    //preprocessing
    if(argc != 5){
        std::cout<<"No enough arugments\n";
        exit(0);
    }

    double wall_timer = omp_get_wtime();

    int n_threads;
    double C;
    std::string tr_file_path, t_file_path;

    std::cout<<"----------------------------------------------"<<"\n";
    std::cout<<"Processing data...";
    C = atof(argv[1]);
    n_threads = atoi(argv[2]);
    tr_file_path = argv[3];
    t_file_path = argv[4];

    //set number of threads
    omp_set_num_threads(n_threads); 

    //1: read data
    int tr_row, t_row, tr_col, t_col, tr_size, t_size, col;
    get_meta_infos(tr_file_path, tr_row, tr_col, tr_size); get_meta_infos(t_file_path, t_row, t_col, t_size);

    int* tr_row_size = new int[tr_row]; int* t_row_size = new int[t_row];
    int* tr_Y = new int[tr_size]; int* t_Y = new int[t_size];

    int* tr_col_size = new int[tr_col];
    int** tr_idxes = new int*[tr_col];
    double** tr_vals = new double*[tr_col];

    get_row_size(tr_file_path, tr_row_size); get_row_size(t_file_path, t_row_size);
    get_col_size(tr_file_path, tr_col_size, tr_idxes, tr_vals, tr_size, tr_row, tr_col);

    spMatrix tr_spA(tr_row, tr_col, tr_row_size); spMatrix t_spA(t_row, t_col, t_row_size);
    fill_matrix_val(tr_file_path, tr_spA, tr_Y); fill_matrix_val(t_file_path, t_spA, t_Y);
    tr_spA.setTranspose(tr_col_size, tr_idxes, tr_vals);

    printf("Done. Time spent on preprocessing data: %0.2f\n", omp_get_wtime() - wall_timer);
    //newton method iteration
    double* d = new double[tr_col];
    double* gradient = new double[tr_col];
    double* W = new double[tr_col]();
    double* aW = new double[tr_col]();
    double* D = new double[tr_row]();
    double* WX = new double[tr_row]();

    double alpha, scale_d_gradient;
    double alphaObjVal, objVal;
    double trAccu, tAccu;
    double startTimer, endTimer, totalTimer;

    totalTimer = 0.0;
    int maxIter = 10;
    for(int i=0;i<maxIter;i++){
        startTimer = omp_get_wtime();
        tr_spA.getWX(WX, W, tr_Y);
        get_gradient_value(gradient, tr_spA, W, tr_Y, WX, C);
        tr_spA.getDiagonalD(D, WX);
        CG_Solve(d, tr_spA, D, gradient, C, tr_col);
        scale_d_gradient = 0.01*tr_spA.dotProduct(d, gradient);
        //do a line search
        alpha = 1.0;
        objVal = get_function_value(tr_spA, W, tr_Y, C); 
        while(true){
            for(int i=0;i<tr_col;i++){
                aW[i] = W[i] + d[i]*alpha;
            }

            alphaObjVal = get_function_value(tr_spA, aW, tr_Y, C);
            if(alphaObjVal < objVal + alpha*scale_d_gradient){
                for(int i=0;i<tr_col;i++){
                    W[i] = aW[i];
                }
                break;
            }
            alpha = alpha/2.0;
        }
        endTimer = omp_get_wtime();
        totalTimer += (endTimer - startTimer);
        objVal = get_function_value(tr_spA, W, tr_Y, C);
        trAccu = get_accuracy(tr_spA, W, tr_Y, tr_row);
        tAccu = get_accuracy(t_spA, W, t_Y, t_row);
        printf("Iteration: %d, Objective Value: %0.6f, training time: %0.6f, train accuracy: %0.6f, test accuracy: %0.6f\n", i, objVal, totalTimer, trAccu, tAccu);
    }

//-------------------------------free variables-------------------------------
    delete[] tr_Y; delete[] t_Y;
    delete[] tr_row_size; delete[] t_row_size;
    delete[] d; delete[] gradient;
    delete[] W; delete[] aW; delete[] D;
    delete[] WX;
    delete[] tr_col_size; delete[] tr_idxes; delete[] tr_vals;
}

void get_col_size(std::string& file_path, int* col_size, int** idxes, double** vals, int size, int row, int col){
    std::string line;
    std::ifstream file(file_path.c_str());
    std::vector<T> list;
    list.reserve(size);
    int cur_i = 0;
    while(std::getline(file, line)){
        std::vector<std::string> tokens = split_array(line, ' ');
        for(int i=1;i<tokens.size();i++){
            int index;
            double value;
            extract_values(tokens[i], index, value, ':');
            list.push_back(T(cur_i, index, value));
        }
        cur_i++;
    }
    Eigen::SparseMatrix<double> X(row, col);
    X.setFromTriplets(list.begin(), list.end());

    //get sparse data
    for(int i=0;i<col;i++){
        int count =0;
        for(Eigen::SparseMatrix<double>::InnerIterator it(X, i); it; ++it){
            count++;
        }
        col_size[i] = count;
        idxes[i] = new int[count];
        vals[i] = new double[count];
        int j=0;
        for(Eigen::SparseMatrix<double>::InnerIterator it(X, i); it; ++it){
            idxes[i][j] = it.row();
            vals[i][j] = it.value();
            j++;
        }
    }
}

double get_accuracy(spMatrix& X, double* W, int* Y, int row){
    double* pred = new double[row]();
    
    X.multiply(W, pred);
    
    int err_cnt = 0;
    for(int i=0;i<row;i++){
        int pred_i;
        if(pred[i] > 0){
            pred_i = 1;
        }else{
            pred_i = -1;
        }
        if(pred_i != Y[i]){
            err_cnt += 1;
        }
    }

    delete[] pred;
    return 1 - double(err_cnt)/row;
}

void get_row_size(std::string& file_path, int* row_size){
     std::ifstream file(file_path.c_str());
     std::string line;
     char delim = ' ';
     int idx = 0;
     while(std::getline(file, line)){

        //accounting for y label
        int count = -1;
        int pos = line.find(delim);
        int initial_pos = 0;
        while( pos != std::string::npos ){
            initial_pos = pos + 1;
            pos = line.find(delim, initial_pos);
            count++;
        }
        if(line.substr(initial_pos, pos - initial_pos + 1) != ""){
            count++;
        }
        row_size[idx] = count;
        idx++;
     }
     file.close();
}

void fill_matrix_val(std::string& file_path, spMatrix& spA, int *Y){
    std::ifstream file(file_path.c_str());
    std::string line;
    int cur_row = 0;
    while(std::getline(file, line)){
        std::vector<std::string> tokens = split_array(line, ' ');
        int y = atoi( tokens[0].c_str() );
        Y[cur_row] = y;
        for(int i=1;i<tokens.size();i++){
            int cur_col;
            double cur_val;
            extract_values(tokens[i], cur_col, cur_val, ':');
            spA.add_val(cur_row, cur_col, cur_val);
        }
        cur_row++;
    }
    file.close();
}

double get_function_value(spMatrix& X, double* W, int* Y, double C){
    double regVal = 1.0/2*X.dotProduct(W, W);
    double lossVal = 0.0;
    {
        for(int i=0;i<X.myRow;i++){
            int* row_idces = X.myIdxes[i];
            double* row_vals = X.myA[i];
            int srow = X.myRow_size[i];
            double dot_val = 0.0;
            for(int j=0;j<srow;j++){
                int idx = row_idces[j];
                dot_val += row_vals[j]*W[idx];
            }
            lossVal += log(1+exp(-Y[i]*dot_val));
        }
    }
    lossVal = C * lossVal;
    return regVal + lossVal;
}

void get_gradient_value(double* gradient, spMatrix& X, double* W, int* Y, double* WX, double C){
    for(int i=0;i<X.myCol;i++){
        gradient[i] = W[i];
    }
    
   for(int i=0;i<X.myRow;i++){
       double tmpVal = WX[i] - 1;
       int* row_idces = X.myIdxes[i];
       double* row_vals = X.myA[i];
       int srow = X.myRow_size[i];
       double scale = C*tmpVal*Y[i];
       for(int j=0;j<srow;j++){
            int idx = row_idces[j];
            gradient[idx] += row_vals[j] * scale; 
       }   
    }
}

void CG_Solve(double* ans, spMatrix& X, double* D, double* b, double C, int size){
     //init vectors
     
     double* p = new double[size];
     double* r = new double[size];
     for(int i=0;i<size;i++){
         ans[i] = 0;
         p[i] = -b[i];
         r[i] = -b[i];
     }

    double alpha, beta, norm_r0, norm_r, new_norm_r;
    double* AP = new double[size];

    norm_r0 = X.dotProduct(r,r);
    norm_r = norm_r0;

    //start looping
    int count =0;
    while(true){
        //calculate AP_k
        X.sparseHessianProduct(AP, D, p, C);
        //calculate alpha_k
        alpha = norm_r/X.dotProduct(p, AP);
        //update ans and r_(k+1)
        for(int i=0;i<size;i++){
            ans[i] = ans[i] + alpha*p[i];
            r[i] = r[i] - alpha*AP[i];
        }

        new_norm_r = X.dotProduct(r,r);
        if(sqrt(new_norm_r)/sqrt(norm_r0) < 0.01){
            break;
        }

        //get beta
        beta = new_norm_r/norm_r;
        norm_r = new_norm_r; 

        //update p
        for(int i=0;i<size;i++){
            p[i] = r[i] + beta*p[i];
        }
    }

    delete[] AP;
    delete[] p;
    delete[] r;
}

void get_meta_infos(std::string& file_path, int& nrow, int& ncol, int& size){
    nrow = 0;
    ncol = 0;
    size = 0;
    std::string line;
    std::ifstream file(file_path.c_str());
    int start, end, cur_idx;
    while(std::getline(file, line)){
	    nrow++;
	    end = line.find_last_of(":");
	    start = line.find_last_of(" ",end);
	    cur_idx = atoi((line.substr(start, end+1)).c_str());
        size += get_entry_size(line, ':');
	    if(cur_idx > ncol){
	        ncol = cur_idx;
	    }
    }
    file.close();
}

int get_entry_size(const std::string& s, char delim){
    int count = 0;
    int pos = s.find(delim);
    while( pos != std::string::npos){
        count++;
        pos = s.find(delim, pos+1);
    }
    return count;
}

std::vector<std::string> split_array(const std::string &s, char delim){
    std::vector<std::string> elems;
    int pos = s.find(delim); 
    int initial_pos = 0;
    std::string temp;
    while( pos != std::string::npos){
	    temp = s.substr( initial_pos, pos - initial_pos + 1);
        initial_pos = pos+1;
	    pos = s.find(delim, initial_pos);
	    elems.push_back( temp );
    }
    temp = s.substr(initial_pos, pos - initial_pos + 1);
    if(temp != ""){
    	elems.push_back( temp );
    }
    return elems;
}

void extract_values(std::string &s, int& index, double& value, char delim){
    int pos = s.find(delim);
    value = atof(s.substr(pos+1).c_str());
    index = atoi(s.substr(0,pos).c_str());
    index--;
}
