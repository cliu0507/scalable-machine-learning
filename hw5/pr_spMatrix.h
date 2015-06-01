#ifndef MY_PR_SP_MATRIX
#define MY_PR_SP_MATRIX

#include <string>
#include <iostream>
#include "Galois/Galois.h"
#include "Galois/Graph/Graph.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"

struct MyNode{
    int id;
    double x1;
    double x2;
    double v;
};

//typedef Galois::Graph::LC_CSR_Graph<MyNode, double> Graph;
typedef Galois::Graph::LC_CSR_Graph<MyNode, double>::with_no_lockable<true>::type Graph;
//typedef Galois::Graph::LC_CSR_Graph<MyNode, double>::with_out_of_line_lockable<true>::type Graph;

typedef Graph::GraphNode GNode;

struct pr_spMatrix{
    Graph g;
    pr_spMatrix(int n_threads, std::string data_file);
    void initNodes(double* R, double* Y);
    void initEdges(double* W);
    void pagerank_multiply();
    void getpM(double* pM, double alpha);
    void initV(double alpha, int col);
    void getR(double* R);
};

#endif
