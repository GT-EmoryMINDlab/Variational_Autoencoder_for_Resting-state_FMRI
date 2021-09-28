function [] = compare_QPP(x_decoded_cell_sorted, fig)
    %% latent pattern
    latent_pattern = squeeze(x_decoded_cell_sorted{1}(1,:,:,1));

    %% QPP
    temp = load('QPP1_VolumetricHCP_GSR.mat');
    T1 = temp.T1;
    
    temp = load('Fan2Yeo.mat');
    Fan2Yeo = temp.Fan2Yeo;
    
    [A, Yeo_idx] = sort(Fan2Yeo);
    QPP_full = T1(Yeo_idx,:);

    %% max correlation lag
    corr_array = zeros(1,11);
    for i = 1:11
        lag = i - 6;
        A = QPP_full(:,17+lag:49+lag);
        B = -latent_pattern;
        corr = corrcoef(A,B);
        corr_array(i) = corr(2);
    end
    %figure(20),plot(-5:5,corr_array)
    [M, idx] = max(corr_array);
    max_lag = idx - 6;
    QPP = QPP_full(:,17+max_lag:49+max_lag);
    
    %% normalize
    latent_pattern = normalize(latent_pattern);
    QPP = normalize(QPP);
    
    %% visualize
    figure(fig(1)),
    ax1 = subplot(1,3,1);cla,hold on
    show_QPP_helpler(-latent_pattern, ax1)
    title('Patterns Represented by Latent Variable 1')
    
    ax2 = subplot(1,3,2);cla,hold on
    show_QPP_helpler(QPP, ax2)
    title('Primary QPP')

    ax3 = subplot(1,3,3);cla,hold on
    show_QPP_helpler(QPP+latent_pattern, ax3)
    title('Difference')
end

