clear
addpath('sub_modules')

model = 2;
hidden_size=32;
beta = 4;
trial = -1;

folder = '90epoch/encoded_data/';
file = ['Encoded_Z_GSR_model' num2str(model) '_hidden' num2str(hidden_size)...
    '_beta' num2str(beta) '_trial' num2str(trial) '.mat'];

temp = load([folder file]);
x_decoded_array = temp.x_decoded_array;
z_array = temp.z_array;
in_img = temp.in_img;
out_img = temp.out_img;

%% sort variance
var_array = squeeze(var(x_decoded_array(1,:,:,:),0,3));
var_sum = sum(var_array,1);
[var_sum_sorted,latent_idx] = sort(var_sum,'descend');
x_decoded_array = x_decoded_array(:,:,:,latent_idx);
z_array = z_array(:,latent_idx);

k=6;

fig = 1; clustered_flag = 0; key = 'SP';
show_clusters(x_decoded_array, [], clustered_flag, key, fig)

%flip
threshold = 0.4;
[x_decoded_array_flipped, z_array_flipped] = apply_flip(x_decoded_array, z_array, threshold);
figure(22),
subplot(311),imagesc(z_array,[-1,1])
subplot(312),imagesc(z_array_flipped,[-1,1])
subplot(313),imagesc(z_array_flipped-z_array,[-1,1])

fig = 3; clustered_flag = 0; key = 'SP';
show_clusters(x_decoded_array_flipped, [], clustered_flag, key, fig)

% sort based on variance
spatial_profile = get_spatial_profile(x_decoded_array_flipped);
% fc_vector = get_fc_vector(x_decoded_array);
% masked_fc_vector = mask_fc(fc_vector);

cluster_idx1 = kmeans(spatial_profile',k, 'Display','final','Distance','correlation','Replicates',200)';
%cluster_idx2 = kmeans(fc_vector',k, 'Display','final','Distance','cityblock','Replicates',200)';
[x_decoded_cell, x_decoded_array_sorted, z_array_sorted] = sort_variance(x_decoded_array_flipped, z_array_flipped, cluster_idx1);

fig = 4; clustered_flag = 1; key = 'SP';
show_clusters(x_decoded_array_sorted, x_decoded_cell, clustered_flag, key, fig)

% three patterns
fig = 5:9;
three_patterns(x_decoded_cell, fig)

% reconstruction
fig = 10;
reconstruction(in_img,out_img, z_array_sorted, fig)

% compare with QPP
fig = 11;
compare_QPP(x_decoded_cell, fig)

% orthogonality
fig = [12,13];
orthogonality(x_decoded_array_flipped, z_array_flipped, hidden_size, fig)

