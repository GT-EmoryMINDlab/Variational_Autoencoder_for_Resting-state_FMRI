function [] = orthogonality(x_decoded_array_sorted, z_array_sorted, hidden_size, fig)
    %% calculations
    temp = squeeze(x_decoded_array_sorted(1,:,:,:));
    temp_flatten = reshape(temp,[246*33,hidden_size]);
    Bases_Corr = corrcoef(temp_flatten);
    TC_Corr = corrcoef(z_array_sorted);
    
    %% show orthogonality of bases
    figure(fig(1)),
    imagesc(Bases_Corr,[-1, 1])
    axis image
    colorbar
    title('Correlation among the Spatiotemporal Patterns')
    xlabel('latent variable #')
    ylabel('latent variable #')

    %% show orthogonality of latent variable time course
    figure(fig(2)),
    imagesc(TC_Corr,[-0.6, 0.6])
    axis image
    colorbar
    title('Correlation among the Latent Variable Time Courses')
    xlabel('latent variable #')
    ylabel('latent variable #')
end

