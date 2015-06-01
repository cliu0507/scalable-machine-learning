#include "spMatrix.h"

struct update_yi{
    Graph& my_g;
    update_yi(Graph& g): my_g(g) {}
    void operator()(GNode n, Galois::UserContext<GNode>& ctx){
        MyNode& my_data = my_g.getData(n);
        for(auto ii = my_g.edge_begin(n), ei = my_g.edge_end(n); ii != ei; ++ii){
            double& my_w = my_g.getEdgeData(ii);
            GNode dst = my_g.getEdgeDst(ii);
            MyNode& other_data = my_g.getData(dst);
            my_data.y += other_data.x * my_w;
        }
    }
};

spMatrix::spMatrix(int n_threads, std::string data_file){
    std::cout<<"Using "<<Galois::setActiveThreads(n_threads)<< " threads\n";
    Galois::Graph::readGraph(g, data_file);
}

void spMatrix::initNodes(double* X, double* Y){
    int idx = 0;
    for(auto ii = g.begin(), ei=g.end(); ii!=ei;++ii){
        GNode src = *ii;
        MyNode& cur_node = g.getData(src);
        cur_node.x = X[idx];
        cur_node.y = Y[idx];
        idx++;
    }
}

void spMatrix::initEdges(double* W){
    int idx = 0;
    for(auto ii = g.begin(), ei=g.end(); ii!=ei;++ii){
        GNode src = *ii;
        for(auto jj=g.edge_begin(src), ej=g.edge_end(src); jj != ej;++jj){
            double& w = g.getEdgeData(jj);
            w = W[idx];
        }
        idx++;
    }
}

void spMatrix::multiply(){
    Galois::for_each(g.begin(), g.end(), update_yi(g)); 
}

double spMatrix::get_y(int pos){
    auto ii_pos = g.begin() + pos;
    MyNode& data = g.getData(*ii_pos);
    return data.y;
}
