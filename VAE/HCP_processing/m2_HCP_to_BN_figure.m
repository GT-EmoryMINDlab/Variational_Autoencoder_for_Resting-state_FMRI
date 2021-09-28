clear 
clc
%% Define Path
dir_read = 'Y:\keilholz-lab\Xiaodi\HCP_S500_Volumetric';
dir_write = 'Y:\keilholz-lab\Xiaodi\HCP_S500_Processed';

Table = readtable([dir_read '\HCP_412.csv']);
subject_vector = Table.Subject;

%% Load Data
% load HCP
flag = 1;
switch flag
    case 0 
        task_str1 = 'Resting_State_1';      task_str2 = 'rfMRI_REST1';      task_str3 = 'rest';
    case 1
        task_str1 = 'Emotion_Task';         task_str2 = 'tfMRI_EMOTION';    task_str3 = 'emotion';
    case 2
        task_str1 = 'Gambling_Task';        task_str2 = 'tfMRI_GAMBLING';   task_str3 = 'gambling';
    case 3
        task_str1 = 'Language_Task';        task_str2 = 'tfMRI_LANGUAGE';   task_str3 = 'language';
    case 4
        task_str1 = 'Motor_Task';           task_str2 = 'tfMRI_MOTOR';      task_str3 = 'motor';
    case 5
        task_str1 = 'Relational_Task';      task_str2 = 'tfMRI_RELATIONAL'; task_str3 = 'relational';
    case 6
        task_str1 = 'Social_Task';          task_str2 = 'tfMRI_SOCIAL';     task_str3 = 'social';
    case 7
        task_str1 = 'Working_Memory_Task';  task_str2 = 'tfMRI_WM';         task_str3 = 'wm';
end
        
for i = 3:3
    subject_id = subject_vector(i);
    filename1 = [dir_read '\' task_str1 '\' num2str(subject_id) '\MNINonLinear\Results\' task_str2 '_LR\' task_str2 '_LR.nii.gz'];
    filename2 = [dir_read '\' task_str1 '\' num2str(subject_id) '\MNINonLinear\Results\' task_str2 '_LR\Movement_Regressors.txt'];
    I_HCP = niftiread(filename1);
    %info_HCP = niftiinfo(filename1);
    Motion = readmatrix(filename2)';
end

I_HCP = I_HCP(:,:,:,6:end); % delete the first 5 frames
Motion = Motion(:,6:end); % delete the first 5 frames

% load BN atlas
I_BN = niftiread('..\Atlas\Fan246\BN_Atlas_246_2mm.nii.gz');
info_BN = niftiinfo('..\Atlas\Fan246\BN_Atlas_246_2mm.nii.gz');

% load WM and CSF mask
foldername = [dir_write '\' num2str(subject_id)];
I_GM = niftiread([foldername '\' num2str(subject_id) '_GM_mask.nii']);
info_GM = niftiinfo([foldername '\' num2str(subject_id) '_GM_mask.nii']);
I_WM = niftiread([foldername '\' num2str(subject_id) '_WM_mask.nii']);
info_WM = niftiinfo([foldername '\' num2str(subject_id) '_WM_mask.nii']);
I_CSF = niftiread([foldername '\' num2str(subject_id) '_CSF_mask.nii']);
info_CSF = niftiinfo([foldername '\' num2str(subject_id) '_CSF_mask.nii']);

%% Flatten data
[dim_x,dim_y,dim_z,dim_t]=size(I_HCP);
I_HCP_flattened = double(reshape(I_HCP,[dim_x*dim_y*dim_z,dim_t]));
I_BN_flattened = double(reshape(I_BN,[dim_x*dim_y*dim_z,1]));
I_GM_flattened = double(reshape(I_GM,[dim_x*dim_y*dim_z,1]));
I_WM_flattened = double(reshape(I_WM,[dim_x*dim_y*dim_z,1]));
I_CSF_flattened = double(reshape(I_CSF,[dim_x*dim_y*dim_z,1]));

%% Calculate GS, WM and CSF
HCP_mask = I_HCP_flattened(:,1)~=0;
GM = (I_HCP_flattened' * I_GM_flattened / sum(I_GM_flattened))';
WM = (I_HCP_flattened' * I_WM_flattened / sum(I_WM_flattened))';
CSF = (I_HCP_flattened' * I_CSF_flattened / sum(I_CSF_flattened))';

GM = (GM - mean(GM,2));
WM = (WM - mean(WM,2));
CSF = (CSF - mean(CSF,2));

%% mask brain to reduce size
BN_mask_idx = find(I_BN_flattened~=0);
I_HCP_masked = I_HCP_flattened(BN_mask_idx,:);
I_HCP_masked = I_HCP_masked - repmat(mean(I_HCP_masked,2),[1,dim_t]);

%% WM CSF and motion regression
% Y = Xb + e 
% Y = BOLD,       [T,N] = [171, 140680]
% X = Regressors, [T,L] = [171, 17    ]
% b = Weights,    [L,N] = [17,  140680]
% b = inv(X'X)X'Y
t = (1:dim_t)/dim_t;
Trend = [t.^2; t ;ones(1,dim_t)];
Regressor_mat = [Motion; GM; WM; CSF; Trend];

Y = I_HCP_masked';
X = Regressor_mat';
b = (X'*X)\X'*Y;

I_HCP_regressed = (Y - X*b)';

R0 = corrcoef([I_HCP_masked(20000,:); Regressor_mat]')    % corr before regression
R1 = corrcoef([I_HCP_regressed(20000,:); Regressor_mat]') % corr after regression

%% Design Filter
fs = 1/0.72;
MyFilter = designfilt('bandpassiir','FilterOrder',6, ...
         'HalfPowerFrequency1',0.01,'HalfPowerFrequency2',0.1, ...
         'SampleRate',fs);

%% band pass filter BOLD data
tau=40;
A = [zeros(length(BN_mask_idx),tau),I_HCP_regressed,zeros(length(BN_mask_idx),tau)];
B = filtfilt(MyFilter,A')';
I_HCP_filtered = B(:,1+tau:end-tau);

% A = [zeros(length(BN_mask_idx),tau),I_HCP_masked,zeros(length(BN_mask_idx),tau)];
% B = filtfilt(MyFilter,A')';
% I_HCP_filtered2 = B(:,1+tau:end-tau);

%figure(1),plot(1:dim_t,I_HCP_masked(20000,:),1:dim_t,I_HCP_regressed(20000,:),1:dim_t,I_HCP_filtered(20000,:)),legend('raw','regressed','filtered');
figure(9),imagesc(zscore(I_HCP_filtered,[],2))

%% apply BN mask
I_BN_matrix = zeros(length(BN_mask_idx),246);
I_BN_masked = I_BN_flattened(BN_mask_idx,:);

for i=1:246
    I_BN_matrix(:,i) = I_BN_masked==i;
end
% figure(2),plot(1:length(BN_mask_idx),I_BN_masked/max(I_BN_masked),1:length(BN_mask_idx),I_HCP_masked(:,1)/max(I_HCP_masked(:,1)))
% A=sum(I_BN_matrix,1);
% figure(3),imagesc(I_BN_matrix)
% figure(4),plot(A)
num_per_parcel = sum(I_BN_matrix,1);
HCP_in_BN = zscore(I_BN_matrix' * I_HCP_filtered,[],2);
% HCP_in_BN2 = zscore(I_BN_matrix' * I_HCP_filtered2,[],2);

%% 
load('my_BN_order.mat');
figure(13),cla;
A1 = HCP_in_BN(BN_idx,:);
imagesc(A1)
set(gca,'YGrid', 'on', 'YTick', boundary_idx, 'GridAlpha', 1, 'LineWidth', 1)

% figure(14),cla;
% A2 = HCP_in_BN2(BN_idx,:);
% imagesc(A2)
% set(gca,'YGrid', 'on', 'YTick', boundary_idx, 'GridAlpha', 1, 'LineWidth', 1)

R1 = corrcoef(A1');
figure(15)
imagesc(R1,[-0.6,0.6])
set(gca,'YGrid', 'on', 'YTick', boundary_idx, 'GridAlpha', 1, 'LineWidth', 1)

%%
dir_read = 'E:\HCP_DATA_BN_fullatlas';
filename1 = [dir_read '\' num2str(subject_id) '\' task_str2(7:end) '_LR\' task_str3 '.txt'];
HCP_in_BN_Amrit = readmatrix(filename1);

figure(17),cla;
A2 = HCP_in_BN_Amrit(BN_idx,:);
imagesc(A2)
set(gca,'YGrid', 'on', 'YTick', boundary_idx, 'GridAlpha', 1, 'LineWidth', 1)

R2 = corrcoef(A2');
figure(18)
imagesc(R2,[-0.6,0.6])
set(gca,'YGrid', 'on', 'YTick', boundary_idx, 'GridAlpha', 1, 'LineWidth', 1)