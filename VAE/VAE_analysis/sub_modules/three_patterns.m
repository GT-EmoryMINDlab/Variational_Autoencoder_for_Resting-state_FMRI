function [] = three_patterns(x_decoded_cell, fig)
    k = length(x_decoded_cell);

    label_idx = [0,39,73,106,136,158,184,210,246];
    label = {'SC','VIS','SM','DA','VA','LIM','FP','DM'};
    label_loc = (label_idx(1:end-1)+label_idx(2:end))/2;

    cursor_x1 = [-3*ones(1,9);3*ones(1,9)];
    cursor_x2 = [0*ones(1,9);33*ones(1,9)];
    cursor_y = [label_idx;label_idx];

    max_num = 0;
    for i = 1:k
        num = size(x_decoded_cell{i}, 4);
        max_num = max(max_num, num);
    end
        
    %% Spatial Pattern
    figure(fig(1)),clf
    for i=1:k
        temp = x_decoded_cell{i};
        for j=1:size(temp,4)
            [M,max_time] = max(squeeze(sum(abs(temp(1,:,:,j)),2)));
            ptr = max_num*(i-1)+j;
            
            temp_show = squeeze(temp(:,:,max_time,j))';
            a = prctile(temp_show(:),98);

            subplot(k,max_num,ptr),cla, 
            set(gca,'ytick',[])
            xlim([-3,3])
            ylim([1,246])
            
            imagesc([-3,3],[0,246],temp_show, [-0.8, 0.8])
            hold on
            plot(cursor_x1, cursor_y, 'w','LineWidth',1.5)
            for cnt=1:8
                text(-4,label_loc(cnt),label{cnt},'fontsize',10)
            end
        end
    end

    %% Temporal Pattern
    figure(fig(2)),clf
    for i=1:k
        temp = x_decoded_cell{i};
        for j=1:size(temp,4)
            ptr = max_num*(i-1)+j;

            temp_show = squeeze(mean(temp(:,96:100,:,j),2));
            a = prctile(temp_show(:),98);
            
            subplot(k,max_num,ptr),
            set(gca,'ytick',[-2,0,2])
            imagesc([0,23.76],[-3,3],temp_show,[-a,a])
        end
    end

    %% Spatiotemporal Pattern
    figure(fig(3)),clf
    for i=1:k
        temp = x_decoded_cell{i};
        for j=1:size(temp,4)
            ptr = max_num*(i-1)+j;
            [M,max_time] = max(squeeze(sum(abs(temp(1,:,:,j)),2)));
            temp_show = squeeze(temp(1,:,:,j));
            a = prctile(temp_show(:),98);
            
            subplot(k,max_num,ptr),cla, 
            set(gca,'ytick',[])
            xlim([0,23.76])
            ylim([1,246])
            
            imagesc([0,23.76],[0,246],temp_show,[-0.15, 0.15])
            hold on
            plot(cursor_x2, cursor_y, 'w','LineWidth',1.5)
            for cnt=1:8
                text(-3.5,label_loc(cnt),label{cnt},'fontsize',10)
            end
            plot([max_time,max_time]*0.72,[1,246],'k')
        end
    end

    %% Individual Functional connectivity
    R_average = zeros(246,246,k);
    figure(fig(4)),clf
    for i=1:k
        temp = x_decoded_cell{i};
        for j=1:size(temp,4)
            ptr = max_num*(i-1)+j;
            temp_show = squeeze(temp(1,:,:,j));

            R = corrcoef(temp_show');
            var_array2 = squeeze(var(temp(1,:,:,:),0,3));
            var_sum_cell{i} = sum(var_array2,1);
            R_average(:,:,i) = R_average(:,:,i) + var_sum_cell{i}(j) * R;
            
            subplot(k,max_num,ptr),imagesc(R)
        end
    end
    
    %% Averaged Functional connectivity
    label_idx = [0,39,73,106,136,158,184,210,246];
    label = {'SC','VIS','SM','DA','VA','LIM','FP','DM'};
    label_loc = (label_idx(1:end-1)+label_idx(2:end))/2;

    cursor_xx = [zeros(1,9);246*ones(1,9)];
    cursor_yy = [label_idx;label_idx];

    figure(fig(5)),clf
    for i=1:k
        subplot(2,3,i),cla,hold on
        imagesc(R_average(:,:,i)/sum(var_sum_cell{i}),[-1,1])
        plot(cursor_xx, cursor_yy, 'w','LineWidth',1.5)
        plot(cursor_yy, cursor_xx, 'w','LineWidth',1.5)
        for cnt=1:8
            text(-25,label_loc(cnt),label{cnt},'fontsize',10)
            text(label_loc(cnt)-10,-15,label{cnt},'fontsize',10)
        end
        set(gca,'ytick',[],'xtick',[],'box','on')
        xlim([0,246])
        ylim([0,246])
        title(['Cluster' num2str(i)]);

    %     mycolormap = flipud(cbrewer('div', 'Spectral', 256));
    %     colormap(mycolormap)
    end
end

