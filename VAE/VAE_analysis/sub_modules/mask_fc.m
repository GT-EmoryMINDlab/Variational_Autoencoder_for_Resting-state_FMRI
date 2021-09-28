function [masked_fc_vector] = mask_fc(fc_vector)
    label_idx = [0,39,73,106,136,158,184,210,246];
    hidden_size = size(fc_vector,2);
    masked_fc_vector = [];
    fc = reshape(fc_vector, [246,246,hidden_size]);
    mask = ones(246,246);
    figure(21),
    subplot(121),imagesc(fc(:,:,1))
    for i = 1:8
        fc(label_idx(i)+1:label_idx(i+1), label_idx(i)+1:label_idx(i+1),:) = 0;
        mask(label_idx(i)+1:label_idx(i+1), label_idx(i)+1:label_idx(i+1)) = 0;
    end
    subplot(122),imagesc(fc(:,:,1))
    for i = 1:hidden_size 
        A = fc(:,:,i);
        idx = find(mask(:));
        masked_fc_vector = [masked_fc_vector, A(idx)];
    end
end

