function [R_vector] = get_fc_vector(x_decoded_array)
    [N, M, T, hidden_size] = size(x_decoded_array);
    R_vector = zeros(M * M, hidden_size);
    for i = 1:hidden_size
        spatiotemporal_pattern = squeeze(x_decoded_array(1,:,:,i));
        R = corrcoef(spatiotemporal_pattern');
        R_vector(:,i) = R(:);
    end
end