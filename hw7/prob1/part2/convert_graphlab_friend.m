function convert_graphlab_friend()
    load friendster.dat;
    Ix = friendster(:,1);
    Jx = friendster(:,2);
    clear friendster;

    [Ix, sortedIdx] = sort(Ix); 
    Jx = Jx(sortedIdx);
    clear sortedIdx;
    %--------write to a file--------------
    data = [Ix, Jx];

    clear Ix;
    clear Jx;

    tic;
    data_file = fopen('friendster_graphlab.txt','w');
    fprintf(data_file,'%d %d\n', data');
    fclose(data_file);
    toc;

end
