function [label] = inc_k(X, k)
    X = X';
    [m,n] = size(X);
    tol = 0.0001;

    %Initialize X
    for i=1:n
        X(:,i) = X(:,i)/norm(X(:,i));
    end

    %Init random clusters
    label = randi(k, n, 1);

    %Compute initial cluster centroids
    S = zeros(m, k);
    C = zeros(m, k);
    for i=1:n
        lb = label(i);
        S(:, lb) = S(:,lb)+X(:,i);
    end
    %normalize centroid
    for i=1:k
        C(:,i) = S(:,i)/norm(S(:,i));
    end

    %compute initial similarity matrix
    for i=1:n
        for j=1:k
            SM(i,j) = dot(X(:,i), C(:,j));
        end
    end

    %compute initial QK
    for i=1:k
        QK(i) = norm(S(:,i));
    end

    %compute Q for each centroids
    Q = sum(QK);

    fprintf('beginning obj: %f\n', Q);

    delta = 1;
    while(delta > tol)
        max_i = 0;
        max_c = 0;
        delta = 0;
        for i=1:n
            lb = label(i);
            qi = QK(lb);
            nqi = sqrt(qi*qi - 2*qi*SM(i, lb) + 1);
            for j=1:k
                qj = QK(j);
                nqj = sqrt(qj*qj + 2*qj*SM(i,j) + 1);
                cur_delta = nqi + nqj - (qi + qj);
                if(cur_delta > delta)
                    delta = cur_delta;
                    max_i = i;
                    max_c = j;
                end
            end
        end
    
        if(label(max_i) == max_c)
            break;
        end

        %updates
        lb = label(max_i);
        qi = QK(lb);
        qj = QK(max_c);
        QK(lb) = sqrt(qi*qi - 2*qi*SM(max_i, lb) + 1);
        QK(max_c) = sqrt(qj*qj + 2*qj*SM(max_i, max_c) + 1);

        for i=1:n
            xi_lb = dot(X(:,i), X(:,lb));
            SM(i, lb) = (qi*SM(i, lb) - xi_lb) / QK(lb);
            xi_max = dot(X(:,i), X(:,max_c));
            SM(i, max_c) = (qj*SM(i, max_c) + xi_max) / QK(max_c);
        end
        label(max_i) = max_c;
        Q = Q + delta;
        fprintf('delta: %f, obj: %f\n', delta, Q);
    end
    fprintf('obj: %f\n', Q);
end
