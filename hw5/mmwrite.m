function mymmwrite()
	load('hw4-data');
	[Ix, Jx, xx] = find(A);
	[nr, nc] = size(A);
	nnz = size(xx, 1);
	data =[Ix, Jx, xx];
	tic;
	datafile = fopen('mydatamm','w');
	fprintf('%%MatrixMarket matrix coordinate real general\n');
	fprintf('%d %d %d\n', nr, nc, nnz);
	fprintf(datafile, '%d %d %d\n', data');
	fclose(datafile);
	toc;
end
