figure(1),clf;
for i = 1:5
%     if i==1
%         hidden = 16;
%         flag = 2;
%     elseif i==2
%         hidden = 32;
%         flag = 2;
%     elseif i==3
%         hidden = 64;
%         flag = 2;
%     elseif i==4
%         hidden = 32;
%         flag = 1;
%     else
%         hidden = 32;
%         flag = 3;
%     end
    if i==1
        beta = 0;
    elseif i==2
        beta = 1;
    elseif i==3
        beta = 2;
    elseif i==4
        beta = 4;
    else
        beta = 8;
    end
    trial = 2;
    epoch = 90;
    hidden = 32;
    flag = 2;
    
    filename = [num2str(epoch) 'epoch/loss/loss_model' num2str(flag) '_hidden' num2str(hidden)...
        '_beta' num2str(beta) '_trial' num2str(trial) '.mat'];
    temp = load(filename);
    train_loss = temp.train_loss_array;
    val_loss = temp.val_loss_array;

    figure(1),hold on;
    if i==2
        plot(1:epoch,val_loss,'LineWidth',3)
    else
        plot(1:epoch,val_loss,'LineWidth',1.5)
    end
    xlabel('Training Epochs')
    ylabel('Validation Loss')
end
% legend('16-unit,10-layer(narrow)','32-unit,10-layer(selected)','64-unit,10-layer(wide)',...
%     '32-unit,4-layer(shallow)','32-unit,16-layer(deep)')
legend('beta = 0','beta = 1','beta = 2','beta = 4','beta = 8')