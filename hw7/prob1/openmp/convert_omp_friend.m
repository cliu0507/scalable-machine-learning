function convert_omp_friend()
    load friendster.dat;
    Ix = friendster(:,1);
    Jx = friendster(:,2);
    clear friendster;
    size_ele = size(Ix,1);
    xx = ones(size_ele, 1);


    [Ix, sortedIdx] = sort(Ix); 
    Jx = Jx(sortedIdx);
    clear sortedIdx;
    row_max = max(Ix);
    col_max = max(Jx);
    rr = histc(Ix, 1:row_max);
    %--------write to a file--------------
    data = [Ix, Jx, xx];

    tic;    
    meta_file = fopen('friendster_omp.meta','w');
    fprintf(meta_file,'%d %d %d\n', row_max, col_max, size_ele);
    fprintf(meta_file,'%d\n', rr');
    fclose(meta_file);
    toc;

    clear Ix;
    clear Jx;
    clear xx;
    clear rr;

    tic;
    data_file = fopen('friendster_omp.txt','w');
    fprintf(data_file,'%d %d %d\n', data');
    fclose(data_file);
    toc;

end
