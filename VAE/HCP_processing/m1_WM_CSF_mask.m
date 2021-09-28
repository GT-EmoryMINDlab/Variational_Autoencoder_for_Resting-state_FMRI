clear 
clc
%% read csv
dir_read = 'Y:\keilholz-lab\Xiaodi\HCP_S500_Volumetric';
dir_write = 'Y:\keilholz-lab\Xiaodi\HCP_S500_Processed';

T = readtable([dir_read '\HCP_412.csv']);
subject_vector = T.Subject;

%% load wmparc
status_vec=zeros(1,412);
for i = 1:412
    subject_id = subject_vector(i);
    filename = [dir_read '\Structural_Scan\' num2str(subject_id) '\MNINonLinear\ROIs\wmparc.2.nii.gz'];
    I = niftiread(filename);
    info = niftiinfo(filename);

    %% CSF_mask
    CSF_idx = [4, 5, 14, 15, 24, 31, 43, 44, 63];
    CSF_mask = single(zeros(size(I)));
    for j=1:length(CSF_idx)
        CSF_mask = CSF_mask + (I == CSF_idx(j));
    end

    %% WM_mask
    WM_mask = single(I >= 3000);
%     sum(WM_mask(:));

    %% GM_mask
    GM_mask = single((I>=1000)&(I<3000));
%     sum(GM_mask(:));
    
    %% Eroding WM mask to avoid contaimnation
    SE = strel('cube',2);
    WM_mask = imerode(WM_mask,SE);

    %% write to processed folder
    foldername = [dir_write '\' num2str(subject_id)];
    status_vec(i) = mkdir(foldername);
    niftiwrite(CSF_mask,[foldername '\' num2str(subject_id) '_CSF_mask.nii'],info)
    niftiwrite(WM_mask,[foldername '\' num2str(subject_id) '_WM_mask.nii'],info)
    niftiwrite(GM_mask,[foldername '\' num2str(subject_id) '_GM_mask.nii'],info)
end

status_vec