#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <dirent.h>
#include <vector>
#include <limits.h>
#include <cstdio>
#include <ctime>
#include <math.h>
#include <algorithm>

void get_meta_infos(std::string& file, int& r_size, int& f_size, int& eles_size);
std::vector<std::string> split_array(const std::string &s, char delim);
void extract_values(std::string &s, int& index, double& value, char delim);
int get_entry_size(const std::string& s, char delim);
Eigen::SparseMatrix<double> form_sparse_matrix(std::string& file_path, int* Y ,int size, int row, int col); 
double output_diff(Eigen::SparseMatrix<double>& X_tps, double* weights, double* alphas, int* Y, int col);
double get_accuracy(Eigen::SparseMatrix<double>& X, double* weights, int* Y, int row);
double get_primal_obj(double* alphas, double* weights, int* Y, int row, int col, double C);
double get_dual_obj(double* weights, double* alphas, double C, int row, int col);
 
typedef Eigen::Triplet<double> T;
const int offset = 1;

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
    std::cout<<"exec filename: "<<argv[0]<<""<<"\n";
    std::cout<<"Processing data...";
    C = atof(argv[1]);
    n_threads = atoi(argv[2]);
    tr_file_path = argv[3];
    t_file_path = argv[4];

    //set number of threads
    omp_set_num_threads(n_threads); 

    //1: Set necessary arugments
    //2: read train data
    //3: read test data
    int tr_row, t_row, tr_col, t_col, tr_size, t_size, col;

    get_meta_infos(tr_file_path, tr_row, tr_col, tr_size);
    get_meta_infos(t_file_path, t_row, t_col, t_size);
    col = std::max(tr_col, t_col); 

    int Y[tr_size];
    int Yt[t_size];
    Eigen::SparseMatrix<double> X = form_sparse_matrix(tr_file_path, Y, tr_size, tr_row, col);
    Eigen::SparseMatrix<double> Xt = form_sparse_matrix(t_file_path, Yt, t_size, t_row, col);
    
    double alphas[tr_row];
    for(int i=0;i<tr_row;i++){
        alphas[i] = 0;
    }

    double weights[col];
    for(int i=0;i<col;i++){
        weights[i] = 0;
    }


    Eigen::SparseMatrix<double> X_tps = X.transpose();
    int per_idx[tr_row];
    for(int i = 0; i < tr_row; i++){
	    per_idx[i] = i;
    }

    int col_size[tr_row];
    int *idxes[tr_row];
    double *vals[tr_row];
    for(int i=0;i<tr_row;i++){
	int count = 0;
	for(Eigen::SparseMatrix<double>::InnerIterator it(X_tps, i); it; ++it){
	    count++; 
	}
	col_size[i] = count;
	idxes[i] = new int[count];
	vals[i] = new double[count];
	int j=0;
	for(Eigen::SparseMatrix<double>::InnerIterator it(X_tps, i); it; ++it){
		idxes[i][j] = it.row();
		vals[i][j] = it.value();
		j++;
	}
    }

    double Q[tr_row];
    for(int i=0;i<tr_row;i++){
        Q[i] = 0;
    	for(int j=0;j<col_size[i];j++){
            Q[i] += vals[i][j]*vals[i][j];
	    }
    }

    printf("Done. Time spent on preprocessing data: %0.2f\n",omp_get_wtime() - wall_timer);
    
    double total_time = 0;
    int maxiter = 20;
    int count = 0;
    printf("Running with %d of threads. \n", n_threads);
    for(int i = 0; i < maxiter; i++){
	    std::random_shuffle( per_idx, per_idx+tr_row );
        //inner loop
        printf("Iteration: %d takes ... ", i+1); 
        wall_timer = omp_get_wtime();
        //#pragma omp parallel for 
	    for(int j = 0;j<tr_row;j++){
            int idx = per_idx[j]; 
            double yi = Y[idx];
            
            //calculate xit * weights
            double xit_weights = 0.0;
	        for(int k=0;k<col_size[idx];k++){
	    	    xit_weights += *(weights+idxes[idx][k])*vals[idx][k];
            }

	        double min_delta = (1 - yi*xit_weights - \
            	alphas[idx]/(2*C))/(yi*yi*Q[idx]+1/(2*C));

            //perform updates
            double pre_alpha = alphas[idx];
            alphas[idx] = pre_alpha + min_delta;
            if(alphas[idx] < 0 ){
                alphas[idx] = 0;
            }
            //maintain w
            double diff_alpha = alphas[idx] - pre_alpha;
	    
            
		    for(int k=0;k<col_size[idx];k++){
		        int tmp_idx = idxes[idx][k];
		        double up_val = (diff_alpha) * yi * vals[idx][k];
		        double *address = weights + tmp_idx;
		        //#pragma omp atomic
		        *address += up_val;
		    }
        }
        
//------------------outputs per iteration--------------------------------//
       //output wall time
       double iter_time = omp_get_wtime() - wall_timer;
       total_time += iter_time;
       printf("%0.2f seconds\n", iter_time);

       //output dual objetive function
       printf("dual function value: %0.10f \n", get_dual_obj(weights, alphas, C, tr_row, col));

       //output primal objective function
       printf("primal function value: %0.10f \n", get_primal_obj(alphas, weights, Y, tr_row, col, C));

       //output prediction accuracy for both training and testing
       double tr_acc = get_accuracy(X, weights, Y, tr_row);
       double t_acc = get_accuracy(Xt, weights, Yt, t_row);
       printf("train accuracy: %0.2f | test accuracy: %0.2f ",tr_acc, t_acc);

       //output diff between alpha and w
       printf(" diff in the norm: %0.2f\n", output_diff(X_tps, weights, alphas, Y, col));
       
    }
	
    printf("Total running time: %0.2f seconds\n",total_time);

}

double get_dual_obj(double* weights, double* alphas, double C, int row, int col){
    double wt_w, at_a, alpha_sum;
    double tmp_sum;

    tmp_sum = 0;
    for(int i=0;i<col;i++){
        double tmp = weights[i];
        tmp_sum += tmp*tmp;    
    }
    wt_w = tmp_sum/2;
    
    tmp_sum = 0;
    alpha_sum = 0;
    for(int i=0;i<row;i++){
        double tmp = alphas[i];
        tmp_sum += tmp*tmp;
        alpha_sum += tmp;
    }
    at_a = tmp_sum/(4*C);
    return wt_w+at_a-alpha_sum;
}

double get_primal_obj(double* alphas, double* weights, int* Y, int row, int col, double C){
    double tol_loss = 0;
    double tmp_sum = 0;

    for(int j = 0;j<row;j++){
        double tmp = alphas[j]/(2*C);
        tol_loss += tmp*tmp;
    }

    for(int j = 0;j<col;j++){
        double tmp = weights[j];
        tmp_sum += tmp*tmp;
    }
    return tmp_sum/2 + C*tol_loss;
}

double get_accuracy(Eigen::SparseMatrix<double>& X, double* weights, int* Y, int row)
{
    double pred[row];
    //#pragma omp parallel for
    for(int k=0;k<row;k++){
        pred[k] = 0;
    }

    for(int k =0;k<X.outerSize(); ++k){
        for(Eigen::SparseMatrix<double>::InnerIterator it(X,k); it; ++it){
            pred[it.row()] += it.value()*weights[k];
        }
    }

    int err_cnt = 0;
    //#pragma omp parallel for reduction(+:err_cnt)
    for(int j = 0;j<row;j++){
        int pred_i;
        if(pred[j] > 0){
            pred_i = 1;
        }else{
            pred_i = -1;
        }
        if(pred_i != Y[j]){
            err_cnt += 1;
        }
    }
   return 1 - double(err_cnt)/row;

}

double output_diff(Eigen::SparseMatrix<double>& X_tps, double* weights, double* alphas, int* Y, int col)
{
    double sum_weights[col];
   
    //#pragma omp parallel for
    for(int k=0;k<col;k++){
        sum_weights[k]= 0;
    }
    
    for(int k=0;k<X_tps.outerSize();++k){
        for(Eigen::SparseMatrix<double>::InnerIterator it(X_tps,k); it; ++it){
            sum_weights[it.row()] += Y[k]*alphas[k]*it.value();
        }
    }

    double tol_norm = 0;
    //#pragma omp parallel for reduction(+:tol_norm)
    for(int j=0;j<col;j++){
        double tmp = sum_weights[j] - weights[j];
        tol_norm += tmp*tmp;
    }
    return tol_norm;   
}

Eigen::SparseMatrix<double> form_sparse_matrix(std::string& file_path, int* Y ,int size, int row, int col){
    std::string line;
    std::ifstream file(file_path.c_str());
    std::vector<T> list;
    list.reserve(size);
    int cur_i = 0;
    while(std::getline(file, line)){
	std::vector<std::string> tokens = split_array(line, ' ');
        //store y value
	int y = atoi( tokens[0].c_str() );
	Y[cur_i] = y;
        //store X values
        for(int i=1;i<tokens.size();i++){
	    int index;
	    double value;
        extract_values(tokens[i], index, value, ':');
	    list.push_back(T(cur_i, index-offset, value));
	}
	cur_i++;
    }
    Eigen::SparseMatrix<double> X(row, col);
    X.setFromTriplets(list.begin(), list.end());
    return X; 
}

void get_meta_infos(std::string& file_path, int& r_size, int& f_size, int& eles_size){
    r_size = 0;
    f_size = 0;
    eles_size = 0;
    std::string line;
    std::ifstream file(file_path.c_str());
    while(std::getline(file, line)){
	r_size++;
	int end = line.find_last_of(":");
	int start = line.find_last_of(" ",end);
	int f_index = atoi((line.substr(start, end+1)).c_str());
        eles_size += get_entry_size(line, ':');
	if(f_index > f_size){
	    f_size = f_index;
	}
    }
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
}
