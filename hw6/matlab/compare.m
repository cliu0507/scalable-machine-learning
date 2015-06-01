function compare()
    clc;
    k = 3;

     [Y, X] = libsvmread('classic90');

     tic;
     [b_Y] = batch_k(X, k);
     toc;

     fprintf('confusion matrix using batch k means on classic90:\n');
     CM = confusion_matrix(Y, b_Y, k);
     CM

     tic;
     [I_Y] = inc_k(X, k);
     toc;
     fprintf('confusion matrix using incremental k means on classic90:\n');
     CM = confusion_matrix(Y, I_Y, k);
     CM
%-------------------data set------------------------------

    [Y, X] = libsvmread('classic3893');

    tic;
    [b_Y] = batch_k(X, k);
    toc;
    fprintf('confusion matrix using batch k means on classic3893:\%n');
    CM = confusion_matrix(Y, b_Y, k);
    CM

    tic;
    [I_Y] = inc_k(X, k);
    toc;
    fprintf('confusion matrix using incremental k means on classic3893:\n');
    CM = confusion_matrix(Y, I_Y, k);
    CM
end
