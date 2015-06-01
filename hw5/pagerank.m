function pagerank()
    clear;
    clc;
    load('hw4-data');
    [nc,nr] = size(A);
    D_inv = 1./sum(A,2);

    for i=1:nr
        R(i) = 1/nr;
        V(i) = 1/nr;
    end

    R_trans = R';
    V_trans = V';

    maxIter = 50;
    alpha = 0.15;
 
    tic; 
    for i=1:maxIter
        R_trans = (1-alpha)*A*(D_inv.*R_trans)+alpha*sum(R_trans)*V_trans;
    end
    toc;

    [sort_R, Ix] = sort(R_trans, 'descend');
    for i=1:10
        fprintf('%d\n', Ix(i)-1);
    end
end
