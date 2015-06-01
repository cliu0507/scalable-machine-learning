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

typedef Eigen::Triplet<double> T;

int getdir(std::string, std::vector<std::string>& files);

const std::string META_PATTERN = "meta";
const std::string TRAIN_PATTERN = ".train";
const std::string TEST_PATTERN = ".test";
const int offset = 1;


int main(int argc, char *argv[])
{
    //for random number generator
    srand((unsigned)time(NULL));

    //preprocessing
    if(argc != 5){
        std::cout<<"No enough arugments\n";
        exit(0);
    }
    int k, n_threads;
    double lambda, wall_timer;
    std::string data_dir, meta_dir, train_dir, test_dir;
    std::string line;
    int row, col, train_size, test_size;

    std::cout<<"----------------------------------------------"<<std::endl;
    std::cout<<"exec filename: "<<argv[0]<<std::endl;
    wall_timer = omp_get_wtime();
    k = atoi(argv[1]);
    lambda = atof(argv[2]);
    n_threads = atoi(argv[3]);
    data_dir = argv[4];

    //set number of threads
    omp_set_num_threads(n_threads); 
   
    std::vector<std::string> files(3,"");
    files.reserve(3);
    getdir(data_dir, files);

    meta_dir = data_dir+files[0];
    train_dir = data_dir+files[1];
    test_dir = data_dir+files[2];
 

    //std::cout<<"Using [rank, lambda, n_threads, directory]->>: "<<"[ "<<k<<", "<<lambda<<", "<<n_threads<<", "<<data_dir<<"* ]\n";
    //std::cout<<meta_dir<<" "<<train_dir<<" "<<test_dir<<" "<<"\n";

    //read meta file
    std::ifstream meta_file(meta_dir.c_str());
        //get rows and clos
    std::getline(meta_file, line);
    std::stringstream meta_ss(line);
    meta_ss >> row >> col;
        //get size of train
    std::getline(meta_file, line);
    std::stringstream train_ss(line);
    train_ss >> train_size;
        //get size of test
    std::getline(meta_file, line);
    std::stringstream test_ss(line);
    test_ss >> test_size;
    //std::cout<<"row: "<<row<<" col: "<<col<<" train size: "<<train_size<<" test size: "<<test_size<<"\n";


    //read from training file, and construct matrix
    int *Ix = new int[train_size];
    int *Jx = new int[train_size];
    double *xx = new double[train_size];

    int *Iy = new int[train_size];
    int *Jy = new int[train_size];
    double *yy = new double[train_size];

    int *cc = new int[col](); //number of non zeros in each column
    int *rc = new int[row]();


    std::ifstream train_file(train_dir.c_str());
    std::vector<T> train_list;
    train_list.reserve(train_size);

    for(int i=0;i<train_size;i++){
	train_file >> Ix[i];
	train_file >> Jx[i];
	train_file >> xx[i];
        Ix[i] = Ix[i] - offset;
        Jx[i] = Jx[i] - offset;
        train_list.push_back(T(Ix[i], Jx[i], xx[i]));
        cc[Jx[i]] = cc[Jx[i]] + 1;
        rc[Ix[i]]= rc[Ix[i]] + 1;
    }
    Eigen::SparseMatrix<double> R(row, col);
    R.setFromTriplets(train_list.begin(), train_list.end());



    Eigen::SparseMatrix<double> R_tsp = R.transpose();
    int i_count = 0;
    for(int i=0;i<R_tsp.outerSize(); i++){
        for(Eigen::SparseMatrix<double>::InnerIterator it(R_tsp, i); it; ++it){
            Iy[i_count] = it.row();
            Jy[i_count] = it.col();
            yy[i_count] = it.value();
            i_count++;
        }
    }
    
    //std::cout<<"cc(1)=2->>: "<<cc[1]<<" cc(14)=1->>: "<<cc[14]<<" cc(32)=8->>: "<<cc[32]<<"\n";
    //std::cout<<"xx(0)=4->>: "<<xx[0]<<" xx(7)=5->>: "<<xx[7]<<" xx(38)=3->>: "<<xx[7]<<"\n";
    //std::cout<<"rc(4)=23->>: "<<rc[4]<<" rc(6)=1->>: "<<rc[6]<<" rc[12]=148->>: "<<rc[12]<<"\n";
    //std::cout<<"yy(2)=5->>: "<<yy[2]<<" yy(8)=4->>: "<<yy[8]<<" yy(19)=3->>: "<<yy[19]<<"\n";

    //read from testing file, and construct matrix
    
    int *Ixt = new int[test_size];
    int *Jxt = new int[test_size];
    double *xxt = new double[test_size];

    std::ifstream test_file(test_dir.c_str());
    std::vector<T> test_list;
    test_list.reserve(test_size);

    for(int i=0;i<test_size;i++){
    	test_file >> Ixt[i];
        test_file >> Jxt[i];
        test_file >> xxt[i];
        Ixt[i] = Ixt[i] - offset;
        Jxt[i] = Jxt[i] - offset;
        test_list.push_back(T(Ixt[i], Jxt[i], xx[i]));
    }
    Eigen::SparseMatrix<double> Rt(row, col);
    Rt.setFromTriplets(test_list.begin(), test_list.end());

    int maxiter = 10;

    Eigen::MatrixXd U(k, row);
    Eigen::MatrixXd M(k, col);

    //generate random numbers within 0 and 1 for U, and M
    #pragma omp parallel for
    for(int i=0;i<k;i++){
        for(int j=0;j<row;j++){
            U(i,j) = (double) rand() / (double) RAND_MAX;
        }
        for(int j=0;j<col;j++){
            M(i,j) = (double) rand() / (double) RAND_MAX;
        }
    }
    //preprocessing for parallelization
    int *cci = new int[col];
    int *rci = new int[row];
    int pre_count;

    pre_count = 0;
    for(int i=0;i<col;i++){
        cci[i] = pre_count;
        pre_count = cci[i] + cc[i];
    }

    pre_count = 0;
    for(int i=0;i<row;i++){
       rci[i] = pre_count;
       pre_count = rci[i] + rc[i];
    }


    std::cout<<"walltime spent on preprocessing data: "<<omp_get_wtime() - wall_timer<<"\n";

    //std::cout<<"R_tsp: "<<R_tsp.size()<<" R_tsp(4999, 1825) = 3, the output is: "<<R_tsp.coeffRef(4999,1825)<<"\n";

    //for small
    //std::cout<<"R size: "<<R.size()<<" R(1825, 4999) = 3, the output is: "<<R.coeffRef(1825,4999)<<"\n";
    //std::cout<<"Rt size: "<<Rt.size()<<" Rt(1395, 4999) = 3, the output is: "<<Rt.coeffRef(1395,4999)<<"\n";

    //for medium
    //std::cout<<"R size: "<<R.size()<<" R(4750, 3951) is 4, the output is:  "<<R.coeffRef(4750,3951)<<"\n";
    //std::cout<<"Rt size: "<<Rt.size()<<" R(2128, 3951) is 3, the output is: "<<Rt.coeffRef(2128,3951)<<"\n";

//------------------------------begin processing----------------------------------------------//
    double accu_sum=0;
    double rmse_test=0;
    double rmse_train=0;
    Eigen::MatrixXd U_tps = U.transpose();
    #pragma omp parallel for reduction(+:accu_sum) 
    for(int i=0;i<train_size;i++){
        accu_sum += pow(U_tps.row(Ix[i])*M.col(Jx[i]) - xx[i], 2);
    }
    rmse_train = sqrt(accu_sum/train_size);

    accu_sum = 0;
    #pragma omp parallel for reduction(+:accu_sum)
    for(int i=0;i<test_size;i++){
        accu_sum += pow(U_tps.row(Ixt[i])*M.col(Jxt[i]) - xxt[i], 2);
    }
    rmse_test = sqrt(accu_sum/test_size);

    Eigen::MatrixXd iden = Eigen::MatrixXd::Identity(k,k);

    std::cout<<"start with rmse on train: "<< rmse_train <<" rmse on test: "<< rmse_test << " n_threads: "<<n_threads<<std::endl;
    double total_timer, end_timer;
    for(int t=0;t<maxiter;t++){
	printf("iter: %d\n",t+1);

	printf("Minimize M while fixing U ...");
        wall_timer = omp_get_wtime();
        //minimize M while fixing U
	#pragma omp parallel for schedule(dynamic, 1)
        for(int i=0;i<col;i++){
            if( cc[i]>0 ){
                //construct subU, and subR
                Eigen::MatrixXd subU(k, cc[i]);
                Eigen::VectorXd subR(cc[i]);
		int j=cci[i];
                for(int l=0; l<cc[i];l++){
                    subU.col(l) = U.col(Ix[j+l]);
		    subR[l] = xx[j+l];
                }
                M.col(i) = (lambda*iden+subU*subU.transpose()).llt().solve((subU*subR));
            }else{
                M.col(i) = Eigen::VectorXd::Zero(k);
            }
        
        }
        end_timer = omp_get_wtime();
	total_timer += end_timer-wall_timer;	
        printf("%0.2f seconds\n", end_timer - wall_timer);

	printf("Minimize U whilt fixing M ...");
	wall_timer = omp_get_wtime();
        //minimize U while fixing M
        #pragma omp parallel for schedule(dynamic, 1)
        for(int i=0;i<row;i++){
            if( rc[i] > 0){
                //construct subM, and subR
                Eigen::MatrixXd subM(k, rc[i]);
                Eigen::VectorXd subR(rc[i]);
		int j=rci[i];
                for(int l=0;l<rc[i];l++){
                    subM.col(l) = M.col(Iy[j+l]);
		    subR[l] = yy[j+l];
                }
                U.col(i) = (lambda*iden+subM*subM.transpose()).llt().solve((subM*subR));
            }else{
                U.col(i) = Eigen::VectorXd::Zero(k);
            }
        }
	end_timer = omp_get_wtime();
	total_timer += end_timer-wall_timer;
	printf("%0.2f seconds\n", end_timer - wall_timer);

	Eigen::MatrixXd U_tps = U.transpose();
    
        accu_sum = 0;
        #pragma omp parallel for reduction(+:accu_sum)
        for(int i=0;i<train_size;i++){
            accu_sum += pow(U_tps.row(Ix[i])*M.col(Jx[i]) - xx[i], 2);
        }
        rmse_train = sqrt(accu_sum/train_size);
        
        accu_sum = 0;
        #pragma omp parallel for reduction(+:accu_sum)
        for(int i=0;i<test_size;i++){
            accu_sum += pow(U_tps.row(Ixt[i])*M.col(Jxt[i]) - xxt[i], 2);
        }
        rmse_test = sqrt(accu_sum/test_size);

        printf("rmse on train: %0.6f, rmse on test: %0.6f\n",rmse_train, rmse_test);
    }
    printf("total running time: %0.2f\n",total_timer);

    //free variables
    delete[] Ix;
    delete[] Jx;
    delete[] xx;
    delete[] Iy;
    delete[] Jy;
    delete[] yy;
    delete[] Ixt;
    delete[] Jxt;
    delete[] xxt;
    delete[] rci;
    delete[] cci;
}

/*
*Getdir code is borrowed from the following website
*http://www.linuxquestions.org/questions/programming-9/c-list-files-in-directory-379323/
*/

int getdir(std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        std::cout << "Error(" << errno << ") opening " << dir << "\n";
        return errno;
    }
    while ((dirp = readdir(dp)) != NULL) 
    {
	std::string s1 = std::string(dirp->d_name);
	if(s1.find(META_PATTERN) != std::string::npos){
		files[0] = s1;
	}
	else if(s1.find(TRAIN_PATTERN) != std::string::npos){
		files[1] = s1;
	}else if(s1.find(TEST_PATTERN) != std::string::npos){
        	files[2] = s1;
	}
    }
    closedir(dp);
    return 0;
}
