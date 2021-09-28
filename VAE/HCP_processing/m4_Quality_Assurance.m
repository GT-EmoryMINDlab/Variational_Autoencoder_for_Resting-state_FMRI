clear
clc
%% Define Path
dir_read = 'VAE\HCP_S500_All_in_One';

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

%% Define file name
foldername_in = [dir_read '\' task_str1];
filename_in1 = [foldername_in '\' task_str1 '_no_GM_regression.mat'];
filename_in2 = [foldername_in '\' task_str1 '_GM_regression.mat'];

foldername_qa_tc1 = [dir_read '\' task_str1 '\QA\TC_No_GM_regression\'];
foldername_qa_tc2 = [dir_read '\' task_str1 '\QA\TC_GM_regression\'];
foldername_qa_fc1 = [dir_read '\' task_str1 '\QA\FC_No_GM_regression\'];
foldername_qa_fc2 = [dir_read '\' task_str1 '\QA\FC_GM_regression\'];
status1 = mkdir(foldername_qa_tc1)
status2 = mkdir(foldername_qa_tc2)
status3 = mkdir(foldername_qa_fc1)
status4 = mkdir(foldername_qa_fc2)

%% Load all BOLD data
Temp = load(filename_in1);
BOLD_no_GMreg_mat = Temp.BOLD_no_GMreg_mat;
Temp = load(filename_in2);
BOLD_GMreg_mat = Temp.BOLD_GMreg_mat;

%% Load BN atlas order
load('my_BN_order.mat');
label = {'L Subcortical', 'L Occipital', 'L Limbic', 'L Insular', 'L Parietal', 'L Temporal', 'L Frontal',...
    'R Frontal', 'R Temporal', 'R Parietal', 'R Insular', 'R Limbic', 'R Occipital', 'R Subcortical'};
label_loc = ([1;boundary_idx]+[boundary_idx;246])/2;

%% 
for i=1:412
    subject_id = subject_vector(i);
    filename_qa_tc1 = [foldername_qa_tc1 '\' num2str(subject_id) '_' task_str2 '_TC_no_GMreg.png'];
    filename_qa_tc2 = [foldername_qa_tc2 '\' num2str(subject_id) '_' task_str2 '_TC_GMreg.png'];
    filename_qa_fc1 = [foldername_qa_fc1 '\' num2str(subject_id) '_' task_str2 '_FC_no_GMreg.png'];
    filename_qa_fc2 = [foldername_qa_fc2 '\' num2str(subject_id) '_' task_str2 '_FC_GMreg.png'];
    
    HCP_in_BN_no_GM_regression = squeeze(BOLD_no_GMreg_mat(i,:,:))';
    HCP_in_BN_GM_regression = squeeze(BOLD_GMreg_mat(i,:,:))';
    
    text_loc = -0.15*size(HCP_in_BN_no_GM_regression,2);
    
    f1 = figure(1);
    A1 = HCP_in_BN_no_GM_regression(BN_idx,:);
    imagesc(A1);
    set(gca,'YGrid', 'on', 'YTick', boundary_idx, 'YTickLabel',{}, 'GridAlpha', 1, 'LineWidth', 1);
    text(text_loc*ones(14,1),label_loc,label);
    saveas(f1,filename_qa_tc1);
    
    f2 = figure(2);
    R1 = corrcoef(A1');
    imagesc(R1,[-0.6,0.6])
    axis square
    set(gca,'XGrid', 'on', 'XTick', boundary_idx, 'XTickLabel',{}, 'YGrid', 'on', 'YTick', boundary_idx, 'YTickLabel',{}, 'GridAlpha', 1, 'LineWidth', 1)
    text(-60*ones(14,1),label_loc,label);
    saveas(f2,filename_qa_fc1);
    
    f3 = figure(3);
    A2 = HCP_in_BN_GM_regression(BN_idx,:);
    imagesc(A2);
    set(gca,'YGrid', 'on', 'YTick', boundary_idx, 'YTickLabel',{}, 'GridAlpha', 1, 'LineWidth', 1);
    text(text_loc*ones(14,1),label_loc,label);
    saveas(f3,filename_qa_tc2);
    
    f4 = figure(4);
    R2 = corrcoef(A2');
    imagesc(R2,[-0.6,0.6])
    axis square
    set(gca,'XGrid', 'on', 'XTick', boundary_idx, 'XTickLabel',{}, 'YGrid', 'on', 'YTick', boundary_idx, 'YTickLabel',{}, 'GridAlpha', 1, 'LineWidth', 1)
    text(-60*ones(14,1),label_loc,label);
    saveas(f4,filename_qa_fc2);
end
