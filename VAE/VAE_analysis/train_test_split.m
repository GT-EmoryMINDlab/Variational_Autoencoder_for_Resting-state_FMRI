clear,clc
%%
temp = load('Resting_State_1_GM_regression.mat');
X = temp.BOLD_GMreg_mat;
% %% Divide into 33-TR segments with 50% overlapping
% N = floor(1195/16)-1;
% N = floor(1195/8)-3;
% X_segment = zeros(412,N,33,246);
%% Divide into 65-TR segments with 50% overlapping
N = floor(1195/32)-1;
X_segment = zeros(412,N,65,246);
for i = 1:N
    X_segment(:,i,:,:) = X(:,(1+32*(i-1)):(32*(i+1)+1),:);
%     X_segment(:,i,:,:) = X(:,(1+16*(i-1)):(16*(i+1)+1),:);
end
temp = load('Fan2Yeo.mat');
Fan2Yeo = temp.Fan2Yeo;

%%
rng(1)
idx = randperm(412);
train_idx = idx(1:248);
val_idx = idx(249:248+82);
test_idx = idx(248+82+1:end);

[A, Yeo_idx] = sort(Fan2Yeo);

%%
X_train = X_segment(train_idx,:,:,:);
X_val = X_segment(val_idx,:,:,:);
X_test = X_segment(test_idx,:,:,:);

X_train = reshape(permute(X_train,[2 1 3 4]),[248*N,65,246]);
X_train = X_train(:,:,Yeo_idx);
X_train = permute(X_train,[1 3 2]);

X_val = reshape(permute(X_val,[2 1 3 4]),[82*N,65,246]);
X_val = X_val(:,:,Yeo_idx);
X_val = permute(X_val,[1 3 2]);

X_test = reshape(permute(X_test,[2 1 3 4]),[82*N,65,246]);
X_test = X_test(:,:,Yeo_idx);
X_test = permute(X_test,[1 3 2]);

ptr = size(X_train,1)/2;
X_train1 = X_train(1:ptr,:,:);
X_train2 = X_train(ptr+1:end,:,:);
%%
save('Resting_State_GSR_segments_long.mat','X_train', 'X_val', 'X_test', 'idx')