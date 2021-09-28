function [] = reconstruction(in_img,out_img, z_array_sorted, fig)   
    label_idx = [0,39,73,106,136,158,184,210,246];
    label = {'SC','VIS','SM','DA','VA','LIM','FP','DM'};
    label_loc = (label_idx(1:end-1)+label_idx(2:end))/2;

    N=20;
    T=N*33*0.72;
    cursor_x3 = [0*ones(1,9);T*ones(1,9)];
    cursor_y = [label_idx;label_idx];    
    
    %% Reconstruction
    in_img_cat=[];
    out_img_cat=[];
    z_array_cat=[];
    for i = 1:N
        in_img_cat = [in_img_cat,squeeze(in_img(20+2*i-1,:,:))];
        out_img_cat = [out_img_cat,squeeze(out_img(20+2*i-1,:,:))];
    end
    for i = 0:(2*N)
        z_array_cat = [z_array_cat,z_array_sorted(20+i,:)'];
    end
    
    figure(fig),
    
    subplot(4,1,1),cla,hold on
    a = prctile(in_img_cat(:),98);
    imagesc([0,T], [1,246], in_img_cat,[-3,3])
    plot(cursor_x3, cursor_y, 'w','LineWidth',1.5)
    for j=1:8
        text(-30,label_loc(j),label{j},'fontsize',12)
    end
    set(gca,'ytick',[],'Fontsize',14)
    xlabel('Time(sec)')
    xlim([0,T])
    ylim([1,246])

    subplot(4,1,2),cla,hold on
    a = prctile(out_img_cat(:),98);
    imagesc([0,T], [1,246], out_img_cat,[-2.5,2.5])
    plot(cursor_x3, cursor_y, 'w','LineWidth',1.5)
    for j=1:8
        text(-30,label_loc(j),label{j},'fontsize',12)
    end
    set(gca,'ytick',[],'Fontsize',14)
    xlabel('Time(sec)')
    xlim([0,T])
    ylim([1,246])

    subplot(4,1,3),
    imagesc([0,T], [1,6], z_array_cat(1:6,:))
    xlim([0,T])
    set(gca,'Fontsize',14)
    xlabel('Time(sec)')
    ylabel('Latent Variable #')

    subplot(4,1,4),
    imagesc([0,T], [1,5], z_array_cat(7:11,:))
    xlim([0,T])
    set(gca,'Fontsize',14)
    xlabel('Time(sec)')
    ylabel('Latent Variable #')
end

