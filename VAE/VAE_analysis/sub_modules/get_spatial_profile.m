function [A] = get_spatial_profile(x_decoded_array)
    hidden_size = size(x_decoded_array,4);
    A = zeros(246,hidden_size);
    for i = 1:hidden_size
        [M,max_time] = max(squeeze(sum(abs(x_decoded_array(1,:,:,i)),2)));
        A(:,i) = x_decoded_array(1,:,max_time,i);
    end
end

