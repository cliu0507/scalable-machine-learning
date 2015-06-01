#include "pr_spMatrix.h"

struct update{
    Graph& my_g;
    update(Graph& g): my_g(g){}
    void operator()(GNode n, Galois::UserContext<GNode>& ctx){
        MyNode& my_data = my_g.getData(n);
        for(auto ii = my_g.edge_begin(n), ei = my_g.edge_end(n); ii != ei; ++ii){
            double& my_w = my_g.getEdgeData(ii);
            GNode dst = my_g.getEdgeDst(ii);
            MyNode& dst_data = my_g.getData(dst);
            my_data.x2 += dst_data.x1 * my_w;
        }
        my_data.x2 += my_data.v;
    }
};


void pr_spMatrix::getR(double* R){
   int idx = 0;
   for(auto ii=g.begin(), ei=g.end(); ii != ei;++ii){
       GNode src = *ii;
       MyNode& cur_node = g.getData(src);
       R[idx] = cur_node.x1;
       idx++;
   }
}

pr_spMatrix::pr_spMatrix(int n_threads, std::string data_file){
    std::cout<<"Using "<<Galois::setActiveThreads(n_threads)<< " threads\n";
    Galois::Graph::readGraph(g, data_file);
}

void pr_spMatrix::initNodes(double* R, double* Y){
    int idx = 0;
    for(auto ii = g.begin(), ei=g.end(); ii!=ei;++ii){
        GNode src = *ii;
        MyNode& cur_node = g.getData(src);
        cur_node.x1 = R[idx];
        cur_node.x2 = Y[idx];
        cur_node.id = idx;
        idx++;
    }
}

void pr_spMatrix::initV(double alpha, int col){
    int idx=0;
    for(auto ii = g.begin(), ei=g.end(); ii != ei; ++ii){
        GNode src = *ii;
        MyNode& cur_node = g.getData(src);
        cur_node.v = alpha/col;
    }
}

void pr_spMatrix::initEdges(double* W){
    for(auto ii = g.begin(), ei=g.end(); ii!=ei;++ii){
        GNode src = *ii;
        for(auto jj=g.edge_begin(src), ej=g.edge_end(src); jj != ej;++jj){
            double& w = g.getEdgeData(jj);
            GNode dst_Node = g.getEdgeDst(jj);
            MyNode& dst_data = g.getData(dst_Node);
            int id = dst_data.id;
            w = W[id];
        }
    }
}

void pr_spMatrix::getpM(double* pM, double alpha){
     int idx = 0;
     for(auto ii = g.begin(), ei=g.end(); ii!=ei;++ii){
        GNode src = *ii;
        double n_eles = g.edge_end(src) - g.edge_begin(src);
        pM[idx] = (1.0-alpha)*1.0/n_eles;
        idx++;
    }
}

struct move_and_reset{
    Graph& my_g;
    move_and_reset(Graph& g): my_g(g){}
    void operator()(GNode n, Galois::UserContext<GNode>& ctx){
        MyNode& my_data = my_g.getData(n);
        my_data.x1 = my_data.x2;
        my_data.x2 = 0.0; 
    }    
};

void pr_spMatrix::pagerank_multiply(){
    Galois::for_each(g.begin(), g.end(), update(g));
    Galois::for_each(g.begin(), g.end(), move_and_reset(g));
}
