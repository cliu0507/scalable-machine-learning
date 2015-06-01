#ifndef MY_SP_MATRIX
#define MY_SP_MATRIX

#include <string>
#include <iostream>
#include "Galois/Galois.h"
#include "Galois/Graph/Graph.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"

struct MyNode{
    double x;
    double y;
};

//typedef Galois::Graph::LC_CSR_Graph<MyNode, double> Graph;
typedef Galois::Graph::LC_CSR_Graph<MyNode, double>::with_no_lockable<true>::type Graph;
//typedef Galois::Graph::LC_CSR_Graph<MyNode, double>::with_out_of_line_lockable<true>::type Graph;

typedef Graph::GraphNode GNode;

struct spMatrix{
    Graph g;
    spMatrix(int n_threads, std::string data_file);
    void initNodes(double* X, double* Y);
    void initEdges(double* W);
    void multiply();
    double get_y(int pos);
};

#endif
