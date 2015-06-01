function convert_galois_live()

    load livejournal.dat
    [nr, nc] = size(livejournal);
    my_ones = ones(nr, 1);
    x1 = livejournal(:,1);
    x2 = livejournal(:,2);
    fileID = fopen('livejournal.mm', 'w');
    fprintf(fileID, '%d %d %d\n', [x1';x2';my_ones']);
end
