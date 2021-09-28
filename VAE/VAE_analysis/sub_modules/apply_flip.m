function [x_decoded_array_flipped, z_array_flipped] = apply_flip(x_decoded_array, z_array, threshold)
    hidden_size = size(x_decoded_array,4);
    A = get_spatial_profile(x_decoded_array);
	x_decoded_array_flipped = zeros(size(x_decoded_array));
    z_array_flipped = zeros(size(z_array));
    
    % correlation can be pre-computed
    R = corrcoef(A);
    
    % calculate flipping
    flip_flag = ones(1,hidden_size);
    for j = 1:3
        for i = 1:hidden_size
            correlaion_vector = (R(i,:) .* flip_flag) * flip_flag(i); % flip others, and flip self
            correlaion_vector = correlaion_vector(1:i); % first i already sorted
            mask = abs(correlaion_vector) > threshold;
            score = sum(correlaion_vector(find(mask))) - 1;
            if score < 0 % more negative
                flip_flag(i) = -1 * flip_flag(i);
            else
        end
    end
    
    % apply flipping
    for i = 1:hidden_size
        A(:,i) = A(:,i) * flip_flag(i);
        x_decoded_array_flipped(:,:,:,i) = x_decoded_array(:,:,:,i) * flip_flag(i);
        z_array_flipped(:,i) = z_array(:,i) * flip_flag(i);
    end
end

