function [] = show_clusters(x_decoded_array, x_decoded_cell, clustered_flag, key, fig)
    hidden_size = size(x_decoded_array,4);
    if strcmp(key,'FC')
        fc_vector = get_fc_vector(x_decoded_array);
        A = mask_fc(fc_vector);
    elseif strcmp(key,'SP')
        A = get_spatial_profile(x_decoded_array);
    end
    
    %% calculate variance
    var_array = squeeze(var(x_decoded_array(1,:,:,:),0,3));
    var_sum = sum(var_array,1);

    %% similarity matrix
    R1 = corrcoef(A);
    
    figure(fig),
    subplot(4,1,[1,2]),
    if strcmp(key,'FC')
        imagesc(R1,[0,0.6])
    elseif strcmp(key,'SP')
        imagesc(R1,[-0.6,0.6])
    end
        
    set(gca,'Fontsize',14);
    xlabel('latent variable #')
    ylabel('latent variable #')
    title('Spatial similarity at max-var time')
    axis square

    %% cluster id
    xx = (1:hidden_size) - 0.5;
    subplot(4,1,3),cla,hold on;
    if clustered_flag == 0
        plot(xx,xx+0.5)
        xlim([0,hidden_size])
        ylim([0,hidden_size])
    else
        k = length(x_decoded_cell);
        cluster_idx = [];
        boundary_idx = ones(1, k - 1);
        count = 0;
        for i = 1:k
            num_variable = size(x_decoded_cell{i},4);
            cluster_idx = [cluster_idx, i * ones(1, num_variable)];
            count = count + num_variable;
            if i ~= k
                boundary_idx(i) = count;
            end
        end
        
        plot(xx,cluster_idx)
        plot(xx,cluster_idx,'o')
        plot([boundary_idx;boundary_idx],[zeros(1,k-1);(k+2)*ones(1,k-1)],'k')
        xlim([0,hidden_size])
        ylim([0,k+2])
    end
    
    set(gca,'Fontsize',14,'box','on');
    xlabel('latent variable #')
    ylabel('cluster #')

    %% variance
    subplot(4,1,4),cla,hold on;
    plot(xx,var_sum)
    plot(xx,var_sum,'o')

    if clustered_flag == 1
        plot([boundary_idx;boundary_idx],[ones(1,k-1);80*ones(1,k-1)],'k')
    end
    
    xlim([0,hidden_size])
    set(gca,'Fontsize',14,'box','on');
    xlabel('latent variable #')
    ylabel('variance')
end

