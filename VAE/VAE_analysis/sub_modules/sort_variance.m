function [x_decoded_cell, x_decoded_array_sorted, z_array_sorted] = sort_variance(x_decoded_array, z_array, cluster_idx)  
    k = max(cluster_idx);
    x_decoded_cell = cell(1,k);
    x_decoded_array_sorted = zeros(size(x_decoded_array));
    z_array_sorted = zeros(size(z_array));
    
    var_array = squeeze(var(x_decoded_array(1,:,:,:),0,3));
    var_sum = sum(var_array,1);    
    var_cluster = zeros(1,k);

    % between clusters
    for i= 1:k
        idx = find(cluster_idx == i);
        var_cluster(i) = mean(var_sum(idx));
    end
    [var_between_cluster_sorted,between_cluster_idx] = sort(var_cluster,'descend');
    
    % within clusters
    count = 1;
    for i = 1:k
        idx = find(cluster_idx == between_cluster_idx(i));
        var_within_cluster = var_sum(idx);
        [var_within_cluster_sorted,within_cluster_idx] = sort(var_within_cluster,'descend');
        x_decoded_cell{i} = x_decoded_array(:,:,:,idx(within_cluster_idx));
        for j = 1:length(within_cluster_idx)
            x_decoded_array_sorted(:,:,:,count) = x_decoded_array(:,:,:,idx(within_cluster_idx(j)));
            z_array_sorted(:,count) = z_array(:,idx(within_cluster_idx(j)));
            count = count + 1;
        end
    end
end

