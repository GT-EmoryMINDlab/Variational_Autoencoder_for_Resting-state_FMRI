clear 
clc
%% Define Path
dir_read = 'VAE\HCP_S500_Volumetric';
dir_write = 'VAE\HCP_S500_Processed';

Table = readtable([dir_read '\HCP_412.csv']);
subject_vector = Table.Subject;
status_vec = zeros(412,3);

%% Load Data
% load HCP
flag = 7;
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
        
for i = 1:412
    subject_id = subject_vector(i);
    filename1 = [dir_read '\' task_str1 '\' num2str(subject_id) '\MNINonLinear\Results\' task_str2 '_LR\' task_str2 '_LR.nii.gz'];
    filename2 = [dir_read '\' task_str1 '\' num2str(subject_id) '\MNINonLinear\Results\' task_str2 '_LR\Movement_Regressors.txt'];
    I_HCP = niftiread(filename1);
    Motion = readmatrix(filename2)';
    
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
    Regressor_mat1 = [Motion; WM; CSF; Trend];
    Regressor_mat2 = [Motion; GM; WM; CSF; Trend];
    
    Y = I_HCP_masked';
    X1 = Regressor_mat1';
    b1 = (X1'*X1)\X1'*Y;
    I_HCP_regressed1 = (Y - X1*b1)';
    
    X2 = Regressor_mat2';
    b2 = (X2'*X2)\X2'*Y;
    I_HCP_regressed2 = (Y - X2*b2)';
    
%     R0 = corrcoef([I_HCP_masked(20000,:); Regressor_mat1]');    % corr before regression
%     R1 = corrcoef([I_HCP_regressed(20000,:); Regressor_mat1]'); % corr after regression
    
    %% Design Filter
    fs = 1/0.72;
    MyFilter = designfilt('bandpassiir','FilterOrder',6, ...
        'HalfPowerFrequency1',0.01,'HalfPowerFrequency2',0.1, ...
        'SampleRate',fs);
    
    %% band pass filter BOLD data
    tau=40;
    A1 = [zeros(length(BN_mask_idx),tau),I_HCP_regressed1,zeros(length(BN_mask_idx),tau)];
    B1 = filtfilt(MyFilter,A1')';
    I_HCP_filtered1 = B1(:,1+tau:end-tau);
    
    A2 = [zeros(length(BN_mask_idx),tau),I_HCP_regressed2,zeros(length(BN_mask_idx),tau)];
    B2 = filtfilt(MyFilter,A2')';
    I_HCP_filtered2 = B2(:,1+tau:end-tau);
    
    %% apply BN mask
    I_BN_matrix = zeros(length(BN_mask_idx),246);
    I_BN_masked = I_BN_flattened(BN_mask_idx,:);
    
    for j=1:246
        I_BN_matrix(:,j) = I_BN_masked==j;
    end
    HCP_in_BN_no_GM_regression = zscore(I_BN_matrix' * I_HCP_filtered1,[],2);
    HCP_in_BN_GM_regression = zscore(I_BN_matrix' * I_HCP_filtered2,[],2);
    
    %% Output Path
    foldername_out1 = [dir_write '\' num2str(subject_id) '\P1_no_GM_regression'];
    foldername_out2 = [dir_write '\' num2str(subject_id) '\P2_GM_regression'];
    foldername_qa = [dir_write '\' num2str(subject_id) '\QA'];
    status_vec(i,1) = mkdir(foldername_out1);
    status_vec(i,2) = mkdir(foldername_out2);
    status_vec(i,3) = mkdir(foldername_qa);
    filename_out1 = [foldername_out1 '\' num2str(subject_id) '_' task_str2 '_no_GMreg.mat'];
    filename_out2 = [foldername_out2 '\' num2str(subject_id) '_' task_str2 '_GMreg.mat'];
    filename_qa1 = [foldername_qa '\' num2str(subject_id) '_' task_str2 '_TC_1_no_GMreg.png'];
    filename_qa2 = [foldername_qa '\' num2str(subject_id) '_' task_str2 '_FC_1_no_GMreg.png'];
    filename_qa3 = [foldername_qa '\' num2str(subject_id) '_' task_str2 '_TC_2_GMreg.png'];
    filename_qa4 = [foldername_qa '\' num2str(subject_id) '_' task_str2 '_FC_2_GMreg.png'];
    save(filename_out1,'HCP_in_BN_no_GM_regression');
    save(filename_out2,'HCP_in_BN_GM_regression');
    
    %% Quality Assurance
    load('my_BN_order.mat');
    label = {'L Subcortical', 'L Occipital', 'L Limbic', 'L Insular', 'L Parietal', 'L Temporal', 'L Frontal',...
        'R Frontal', 'R Temporal', 'R Parietal', 'R Insular', 'R Limbic', 'R Occipital', 'R Subcortical'};
    label_loc = ([1;boundary_idx]+[boundary_idx;246])/2;
    
    f1 = figure(1);
    A1 = HCP_in_BN_no_GM_regression(BN_idx,:);
    imagesc(A1);
    set(gca,'YGrid', 'on', 'YTick', boundary_idx, 'YTickLabel',{}, 'GridAlpha', 1, 'LineWidth', 1);
    text(-45*ones(14,1),label_loc,label);
    saveas(f1,filename_qa1);
    
    f2 = figure(2);
    R1 = corrcoef(A1');
    imagesc(R1,[-0.6,0.6])
    axis square
    set(gca,'XGrid', 'on', 'XTick', boundary_idx, 'XTickLabel',{}, 'YGrid', 'on', 'YTick', boundary_idx, 'YTickLabel',{}, 'GridAlpha', 1, 'LineWidth', 1)
    text(-60*ones(14,1),label_loc,label);
    saveas(f2,filename_qa2);
    
    f3 = figure(3);
    A2 = HCP_in_BN_GM_regression(BN_idx,:);
    imagesc(A2);
    set(gca,'YGrid', 'on', 'YTick', boundary_idx, 'YTickLabel',{}, 'GridAlpha', 1, 'LineWidth', 1);
    text(-45*ones(14,1),label_loc,label);
    saveas(f3,filename_qa3);
    
    f4 = figure(4);
    R2 = corrcoef(A2');
    imagesc(R2,[-0.6,0.6])
    axis square
    set(gca,'XGrid', 'on', 'XTick', boundary_idx, 'XTickLabel',{}, 'YGrid', 'on', 'YTick', boundary_idx, 'YTickLabel',{}, 'GridAlpha', 1, 'LineWidth', 1)
    text(-60*ones(14,1),label_loc,label);
    saveas(f4,filename_qa4);
    
    i
end
status_vec
