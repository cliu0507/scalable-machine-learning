function convert()
    load('hw4-data');
    [nr, nc] = size(A);
    [Ix, Jx, xx] = find(A'); 
    clear A;
    %[Ix, sortedIdx] = sort(Ix); 
    %Jx = Jx(sortedIdx);
    row_max = max(Ix);
    col_max = max(Jx);
    size_ele = size(Ix,1);
    rr = histc(Ix, 1:nr);
    cc = histc(Jx, 1:nc);
    %--------write to a file--------------
    data = [Jx, Ix, xx];
    clear Jx;
    clear Ix;
    clear xx;
    tic;    
    meta_file = fopen('meta_data.txt','w');
    fprintf(meta_file,'%d %d %d\n', row_max, col_max, size_ele);
    fprintf(meta_file,'%d\n', rr');
    fclose(meta_file);
    toc;

    tic;
    data_file = fopen('data.txt','w');
    fprintf(data_file,'%d %d %d\n', data');
    fclose(data_file);
    toc;

end
