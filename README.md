# Variational-Autoencoder-for-Resting-State-fMRI

This repository contains the source codes that were used for the following paper:
Zhang X, Maltbie E, Keilholz S. Spatiotemporal Trajectories in Resting-state FMRI Revealed by Convolutional Variational Autoencoder. bioRxiv. 2021 Jan 1.
https://www.biorxiv.org/content/10.1101/2021.01.25.427841v1

There are 3 main steps:
1. Preprocessing HCP resting state data
2. Train Variational autoencoder
3. Visualization of results

The detailed steps are the following:
1. Preprocessing HCP resting state data
  a. Download data from the HCP website https://www.humanconnectome.org/study/hcp-young-adult/data-releases
  b. Put the raw data in the folder "VAE/HCP_processing/HCP_S500_Volumetric/". There is a dummy file that illustrates the correct path:
  "VAE/HCP_processing/HCP_S500_Volumetric/Resting_State_1/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR(dummy).nii.gz"
  c. Run the following matlab scripts sequentially:
      m1_WM_CSF_mask.m
      m2_HCP_to_BN.m
      m3_Move_Files.m
      m4_Quality_Assurance.m
  As a result, a file "Resting_State_1_GM_regression.mat" will be generated in the folder "VAE/HCP_processing/HCP_S500_All_in_One/Resting_State_1", which contains the processed, parcellated fMRI data of all subjects in a single file.
  
2. Train Variational autoencoder
  a. Move the generated "VAE/HCP_processing/HCP_S500_All_in_One/Resting_State_1/Resting_State_1_GM_regression.mat" file to the folder "VAE/VAE_analysis/"
  b. Run "VAE/VAE_analysis/train_test_split.m", which generates 'Resting_State_GSR_segments.mat'
  
