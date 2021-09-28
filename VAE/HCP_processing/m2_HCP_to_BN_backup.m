clear 
clc
%% Define Path
dir_read = 'Y:\keilholz-lab\Xiaodi\HCP_S500_Volumetric';
dir_write = 'Y:\keilholz-lab\Xiaodi\HCP_S500_Processed';

Table = readtable([dir_read '\HCP_412.csv']);
subject_vector = Table.Subject;

%% Load Data
% load HCP
for i = 8:8
    subject_id = subject_vector(i);
    filename1 = [dir_read '\Emotion_Task\' num2str(subject_id) '\MNINonLinear\Results\tfMRI_EMOTION_LR\tfMRI_EMOTION_LR.nii.gz'];
    filename2 = [dir_read '\Emotion_Task\' num2str(subject_id) '\MNINonLinear\Results\tfMRI_EMOTION_LR\Movement_Regressors.txt'];
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
I_WM = niftiread([foldername '\' num2str(subject_id) '_WM_mask.nii']);
info_WM = niftiinfo([foldername '\' num2str(subject_id) '_WM_mask.nii']);
I_CSF = niftiread([foldername '\' num2str(subject_id) '_CSF_mask.nii']);
info_CSF = niftiinfo([foldername '\' num2str(subject_id) '_CSF_mask.nii']);

%% Flatten data
[dim_x,dim_y,dim_z,dim_t]=size(I_HCP);
I_HCP_flattened = double(reshape(I_HCP,[dim_x*dim_y*dim_z,dim_t]));
I_BN_flattened = double(reshape(I_BN,[dim_x*dim_y*dim_z,1]));
I_WM_flattened = double(reshape(I_WM,[dim_x*dim_y*dim_z,1]));
I_CSF_flattened = double(reshape(I_CSF,[dim_x*dim_y*dim_z,1]));

%% Design Filter
fs = 1/0.72;
MyFilter = designfilt('bandpassiir','FilterOrder',6, ...
         'HalfPowerFrequency1',0.01,'HalfPowerFrequency2',0.1, ...
         'SampleRate',fs);
% [b,a] = butter(3,[0.01 0.1]/(fs/2),'bandpass');
%fvtool(MyFilter)

%% Calculate GS, WM and CSF
HCP_mask = I_HCP_flattened(:,1)~=0;
GS = sum(I_HCP_flattened,1) / sum(HCP_mask);
WM = (I_HCP_flattened' * I_WM_flattened / sum(I_WM_flattened))';
CSF = (I_HCP_flattened' * I_CSF_flattened / sum(I_CSF_flattened))';

GS = (GS - mean(GS,2));
WM = (WM - mean(WM,2));
CSF = (CSF - mean(CSF,2));

% figure(1),plot(1:dim_t,GS,1:dim_t,WM,1:dim_t,CSF),legend('GS','WM','CSF');

%% Bandpass Filter WM and CSF
tau=40;
A = [zeros(1,tau),GS,zeros(1,tau)];
B = filtfilt(MyFilter,A);
GS_filtered = B(1+tau:end-tau);

A = [zeros(1,tau),WM,zeros(1,tau)];
B = filtfilt(MyFilter,A);
WM_filtered = B(1+tau:end-tau);

A = [zeros(1,tau),CSF,zeros(1,tau)];
B = filtfilt(MyFilter,A);
CSF_filtered = B(1+tau:end-tau);

% figure(2),plot(1:dim_t,GS_filtered,1:dim_t,WM_filtered,1:dim_t,CSF_filtered),legend('GS','WM','CSF');
% figure(3),plot(1:dim_t+2*tau,A,1:dim_t+2*tau,B)
% figure(4),plot(1:dim_t,GS,1:dim_t,GS_filtered)
% figure(5),plot(GS-mean(GS))
% hold on;
% plot(GS_filtered)

%% mask brain to reduce size
BN_mask_idx = find(I_BN_flattened~=0);
I_HCP_masked = I_HCP_flattened(BN_mask_idx,:);
I_HCP_masked = I_HCP_masked - repmat(mean(I_HCP_masked,2),[1,dim_t]);

%% band pass filter BOLD data
A = [zeros(length(BN_mask_idx),tau),I_HCP_masked,zeros(length(BN_mask_idx),tau)];
B = filtfilt(MyFilter,A')';
I_HCP_filtered = B(:,1+tau:end-tau);

% for i=1:100
%     figure(6),plot(1:dim_t,I_HCP_masked(200*i,:),1:dim_t,I_HCP_filtered(200*i,:))
%     pause(0.2)
% end
%% WM CSF and motion regression
% Y = Xb + e 
% Y = BOLD,       [T,N] = [171, 140680]
% X = Regressors, [T,L] = [171, 17    ]
% b = Weights,    [L,N] = [17,  140680]
% b = inv(X'X)X'Y
t = (1:dim_t)/dim_t;
Trend = [t.^2; t ;ones(1,dim_t)];
Regressor_mat = [Motion; WM; CSF; Trend];

Y = I_HCP_filtered

%% apply BN mask
% I_BN_matrix = zeros(902629,246);
% for i=1:246
%     I_BN_matrix(:,i) = I_BN_flattened==i;
% end
% plot(1:902629,I_BN_flattened/max(I_BN_flattened),1:902629,I_HCP_flattened(:,1)/max(I_HCP_flattened(:,1)))