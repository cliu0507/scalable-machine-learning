function convert_graphlab_live()
    load livejournal.dat;
    Ix = livejournal(:,1);
    Jx = livejournal(:,2);

    clear livejournal;

    [Ix, sortedIdx] = sort(Ix); 
    Jx = Jx(sortedIdx);

    clear sortedIdx;
    %--------write to a file--------------
    data = [Ix, Jx];

    clear Ix;
    clear Jx;

    tic;
    data_file = fopen('livejournal_graphlab.txt','w');
    fprintf(data_file,'%d %d\n', data');
    fclose(data_file);
    toc;

end
