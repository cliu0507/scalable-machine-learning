function [label] = batch_k(X, k)
    X = X';
    [m, n] = size(X);
    tol = 0.01;
    %normalize X
    for i=1:n
        X(:,i) = X(:,i)/norm(X(:,i));
    end

    %Init random clusters
    label = randi(k, n, 1);

    %Compute initial cluster centroids 
    S = zeros(m, k);
    for i=1:n
        lb = label(i);
        S(:, lb) = S(:, lb) + X(:, i);
    end
    %normalize centroid
    C = normc(S);


    %Compute Q for each centroids
    Q = 0.0;
    for i=1:k
        Q = Q + norm(S(:,i)); 
    end

    fprintf('beginning obj: %f\n', Q);
    Q_old = 0.0;
    while( Q - Q_old > tol)
        Q_old = Q;

        %assign xi to each of its closest cluster
        for i=1:n
            max_val = -1;
            max_idx = 0;
            for j=1:k
                tmp_max = dot(X(:,i), C(:,j));
                if(tmp_max > max_val)
                    max_val = tmp_max;
                    max_idx = j;
                end
            end
            label(i) = max_idx;
        end
        %Compute cluster centroids
        S = zeros(m, k);
        for i=1:n
            lb = label(i);
            S(:, lb) = S(:, lb) + X(:, i);
        end
        C = normc(S);
        
        %Compute Q
        Q = 0.0;
        for i=1:k
            Q = Q + norm(S(:,i));
        end
    end
    fprintf('obj: %f\n', Q);
end
