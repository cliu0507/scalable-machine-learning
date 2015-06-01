function [CM] = confusion_matrix(tY, pY, k)
    n = numel(tY);
    permmap = perms(1:k);
    n_map = size(permmap, 1);
    max_count = 0;

    %find maximum mapping
    for i=1:n_map
        lab_order = permmap(i,:);
        count = 0;
        for j=1:n
            cur_idx = pY(j);
            true_lb = tY(j);
            cur_lb = lab_order(cur_idx);
            if(true_lb == cur_lb)
                count = count + 1;
            end
        end
        if(count >= max_count)
            max_count = count;
            map_idx = i;
        end
    end

    %output confusion matrix
    max_order = permmap(map_idx,:);
    CM = zeros(k,k);
    for i=1:n
        cur_idx = pY(i);
        true_lb = tY(i);
        cur_lb = max_order(cur_idx);
        CM(true_lb, cur_lb) = CM(true_lb, cur_lb) + 1;
    end
end
