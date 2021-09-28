function [] = show_QPP_helpler(time_course, ax)
    %%    
    label_idx = [0,39,73,106,136,158,184,210,246];
    label = {'SC','VIS','SM','DA','VA','LIM','FP','DM'};
    label_loc = (label_idx(1:end-1)+label_idx(2:end))/2;

    cursor_x2 = [0*ones(1,9);33*ones(1,9)];
    cursor_y = [label_idx;label_idx];    

    %%
    ax;
    imagesc([0,23.76],[0,246],time_course,[-1, 1])
    plot(cursor_x2, cursor_y, 'w','LineWidth',2)
    for cnt=1:8
        text(-2.5,label_loc(cnt),label{cnt},'fontsize',10)
    end
    set(gca,'ytick',[],'FontSize',11)

    xlabel('Time(second)')
    xlim([0,23.76])
    ylim([1,246])
end

