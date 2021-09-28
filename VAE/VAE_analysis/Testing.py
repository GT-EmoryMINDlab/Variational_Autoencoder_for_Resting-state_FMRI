# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:52:04 2021

@author: Xiaodi
"""
from __future__ import print_function
import argparse
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import models
from scipy.io import loadmat, savemat
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import time

#%%
if torch.cuda.is_available():
    device1 = torch.device("cuda")
else:
    device1 = torch.device("cpu")
print(device1)

device2 = torch.device("cpu")

batch_size = 512
hidden_size = 32
beta = 4
flag = 2 #model flag
trial = 0

filename = 'model'+str(flag)+'_hidden'+str(hidden_size)+'_beta'+str(beta)+'_trial'+str(trial)
model = torch.load('150epoch/models/' + filename + '.pt')

#%%
GSR_flag = 1
if GSR_flag==1:
    temp = loadmat('Resting_State_GSR_segments.mat')
else:
    temp = loadmat('Resting_State_no_GSR_segments.mat')

X_test = torch.from_numpy(temp['X_test'])
print(X_test.shape)
test_dataset = torch.utils.data.TensorDataset(X_test)
test_loader = torch.utils.data.DataLoader(test_dataset,
                 batch_size=batch_size, shuffle=False, pin_memory = True, num_workers = 0)

#%%
model.eval()
model.to(device1)
N = X_test.shape[0]
z_array=torch.zeros(N,hidden_size)
out_img=torch.zeros(N,246,33)
in_img=torch.zeros(N,246,33)

with torch.no_grad():  
    for i, data in enumerate(test_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data[0].float().to(device1)
        l = inputs.shape[0]
        
        outputs, mu, logvar, z = model(inputs)
        
        z_array[i*batch_size:i*batch_size+l,:] = z
        out_img[i*batch_size:i*batch_size+l,:] = outputs
        in_img[i*batch_size:i*batch_size+l,:] = inputs

z_array = z_array.to(device2).numpy()
out_img = out_img.to(device2).numpy()
in_img = in_img.to(device2).numpy() 

#%%
model.eval()
n = 100
m = hidden_size
grid_x = torch.linspace(-3, 3, n)
x_decoded_array = np.zeros((n,246,33,m))
var_array = np.zeros((246,33,m))

plt.figure(6,dpi=600)
with torch.no_grad(): 
    for i in range(m):
        z_sample = torch.zeros(n,m).to(device1)
        z_sample[:,i] = grid_x
        x_decoded = model.decode(z_sample)
        x_decoded_array[:,:,:,i] = x_decoded.cpu().numpy()
        
        plt.subplot(8,8,i+1)
        
        plt.imshow(x_decoded[:,1,:].cpu(), aspect='auto')
        plt.margins(0,0)
        
#%%
if GSR_flag==1:
    savemat( '150epoch/encoded_data/Encoded_Z_GSR_' + filename +'.mat', 
            {'z_array': z_array, 'in_img': in_img, 'out_img': out_img, 
             'grid_x': grid_x.numpy(), 'x_decoded_array': x_decoded_array}) 
else:
    savemat( '150epoch/encoded_data/Encoded_Z_no_GSR_' + filename +'.mat', 
            {'z_array': z_array, 'in_img': in_img, 'out_img': out_img, 
             'grid_x': grid_x.numpy(), 'x_decoded_array': x_decoded_array}) 