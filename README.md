# Variational-Autoencoder-for-Resting-State-fMRI

Xiaodi Zhang, 09/28/2021

This repository contains the source codes that were used for the following paper:
Zhang X, Maltbie E, Keilholz S. Spatiotemporal Trajectories in Resting-state FMRI Revealed by Convolutional Variational Autoencoder. bioRxiv. 2021 Jan 1.
https://www.biorxiv.org/content/10.1101/2021.01.25.427841v1

There are 3 main steps:
1. Preprocessing HCP resting state data
2. Train Variational autoencoder
3. Visualization of results

Before we start, because of the file size constraints, the files that are large in size were replaced by dummy files. If you run the code in the correct order you should be able to generate the correct file locally.

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
  
  b. Run "VAE/VAE_analysis/train_test_split.m", which generates a dataset called "Resting_State_GSR_segments.mat" that is ready for training the networks. The dataset contains fMRI segments as individual data points.
  
  c. Run "main.py". In the function, it creates an object "VAE_model" defined in "myVAE.py". There are 4 functions in the class "VAE_model":
  constructor, data_partition(), training() and testing(). You can change hyperparameters like model_flag = 3, hidden_size = 32, trial = 0, beta = 4 to others and the result would be saved in a separate file with corresponding name
  
3. Visualization of results
  Run "VAE/VAE_analysis/display_result.m"
  
If you have any questions, please contact the corresponding author Dr. Shella Keilholz shella.keilholz@bme.gatech.edu or me xiaodizhang12@gmail.com. Please cite our paper if you do end up using our code. Thank you!
  
