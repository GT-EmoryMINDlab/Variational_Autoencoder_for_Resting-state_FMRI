clear 
clc
%% Define Path
dir_read = 'VAE\HCP_S500_Processed';
dir_write = 'VAE\HCP_S500_All_in_One';

Table = readtable('VAE\HCP_S500_Volumetric\HCP_412.csv');
subject_vector = Table.Subject;

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

foldername_out = [dir_write '\' task_str1];
status = mkdir(foldername_out)

for i = 1:412
    subject_id = subject_vector(i);
    foldername_in = [dir_read '\' num2str(subject_id)];
    filename_in1 = [foldername_in '\P1_no_GM_regression\' num2str(subject_id) '_' task_str2 '_no_GMreg.mat'];
    filename_in2 = [foldername_in '\P2_GM_regression\' num2str(subject_id) '_' task_str2 '_GMreg.mat'];
%     filename_qa1 = [foldername '\QA\' num2str(subject_id) '_' task_str2 '_TC_1_no_GMreg.png'];
%     filename_qa2 = [foldername '\QA\' num2str(subject_id) '_' task_str2 '_FC_1_no_GMreg.png'];
%     filename_qa3 = [foldername '\QA\' num2str(subject_id) '_' task_str2 '_TC_2_GMreg.png'];
%     filename_qa4 = [foldername '\QA\' num2str(subject_id) '_' task_str2 '_FC_2_GMreg.png'];
    
    %% load BOLD data
    Temp = load(filename_in1);
    BOLD_no_GMreg = Temp.HCP_in_BN_no_GM_regression;
    Temp = load(filename_in2);
    BOLD_GMreg = Temp.HCP_in_BN_GM_regression;
    
    if i == 1
        % Matrix in (N, T, L): N subjects, T time points, L parcels
        BOLD_no_GMreg_mat = zeros(412,size(BOLD_no_GMreg,2),246);
        BOLD_GMreg_mat = zeros(412,size(BOLD_no_GMreg,2),246);
    end
    
    BOLD_no_GMreg_mat(i,:,:) = BOLD_no_GMreg';
    BOLD_GMreg_mat(i,:,:) = BOLD_GMreg';
end

filename_out1 = [foldername_out '\' task_str1 '_no_GM_regression.mat'];
filename_out2 = [foldername_out '\' task_str1 '_GM_regression.mat'];
save(filename_out1, 'BOLD_no_GMreg_mat');
save(filename_out2, 'BOLD_GMreg_mat');
