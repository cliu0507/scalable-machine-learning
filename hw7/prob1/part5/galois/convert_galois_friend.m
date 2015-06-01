function convert_galois_friend()

    load friendster.dat
    [nr, nc] = size(friendster);
    my_ones = ones(nr, 1);
    x1 = friendster(:,1);
    x2 = friendster(:,2);
    clear friendster;
    ele_size = size(x1, 1);
    data = [x1';x2';my_ones'];
    clear x1;
    clear x2;
    clear my_ones;
    tic;
    fileID = fopen('friendster_galois.mm', 'w');
    fprintf(fileID, '%d %d %d\n', data);
    fclose(fileID);
    toc;
end
