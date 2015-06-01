function matrix
    M = 500;
    N = 500;

    A = zeros(M,N);
    B = zeros(M,N);
    C = zeros(M,N);
    
    for i=1:M
        for j=1:N
            A(i,j) = i+j;
            B(i,j) = i*j;
        end
    end
   
    tic;
    C = A*B;
    toc;
end
