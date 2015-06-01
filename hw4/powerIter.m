function powerIter()
    maxIter = 50;
    load('hw4-data');
    [nr, nc] = size(A);
    for i=1:nc
	X(i) = 1;
    end
    X = X';
    X = X/norm(X);
    tic;
	for t=1:maxIter
	    X = A*X;
	    X = X/norm(X);
        end
    toc;
    lambda = X'*A*X;
    lambda
    [sX, I] = sort(X,'descend');
    for i=1:100
    	fprintf('index: %d, value: %f\n',I(i),sX(i));
    end
end
