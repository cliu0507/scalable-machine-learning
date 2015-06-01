function convert_omp_live()
    load livejournal.dat;
    Ix = livejournal(:,1);
    Jx = livejournal(:,2);
    size_ele = size(Ix,1);
    xx = ones(size_ele, 1);


    [Ix, sortedIdx] = sort(Ix); 
    Jx = Jx(sortedIdx);
    row_max = max(Ix);
    col_max = max(Jx);
    rr = histc(Ix, 1:row_max);
    %--------write to a file--------------
    data = [Ix, Jx, xx];

    tic;    
    meta_file = fopen('livejournal_omp.meta','w');
    fprintf(meta_file,'%d %d %d\n', row_max, col_max, size_ele);
    fprintf(meta_file,'%d\n', rr');
    fclose(meta_file);
    toc;

    tic;
    data_file = fopen('livejournal_omp.txt','w');
    fprintf(data_file,'%d %d %d\n', data');
    fclose(data_file);
    toc;

end
