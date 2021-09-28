# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:58:33 2021

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

num_epochs = 150
beta = 4
hidden_size = 32
batch_size = 512
learning_rate = 0.001
trial = 1

flag = 3
if flag == 1:
    model = models.VAE_Temporal_Conv_Shallow(hidden_size)
elif flag == 2:
    model = models.VAE_Temporal_Conv(hidden_size)
elif flag == 3:
    model = models.VAE_Temporal_Conv_Deep(hidden_size)
    
#%%
temp = loadmat('Resting_State_GSR_segments.mat')
X_train = torch.from_numpy(temp['X_train'])
X_val = torch.from_numpy(temp['X_val'])
X_test = torch.from_numpy(temp['X_test'])

train_dataset = torch.utils.data.TensorDataset(X_train)
train_loader = torch.utils.data.DataLoader(train_dataset,
                 batch_size=batch_size, shuffle=True, num_workers = 0)

val_dataset = torch.utils.data.TensorDataset(X_val)
val_loader = torch.utils.data.DataLoader(val_dataset,
                 batch_size=batch_size, shuffle=True, num_workers = 0)

test_dataset = torch.utils.data.TensorDataset(X_test)
test_loader = torch.utils.data.DataLoader(test_dataset,
                 batch_size=batch_size, shuffle=False, num_workers = 0)

#%%
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (1*MSE + beta*KLD, MSE, KLD)

#%%
model.train()
model.to(device1)

train_loss_array = np.zeros(num_epochs)
val_loss_array = np.zeros(num_epochs)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


t = time.time()
for epoch in range(num_epochs):  # loop over the dataset multiple times
    train_loss = 0.0
    train_MSE = 0.0
    train_KLD = 0.0
    val_loss = 0.0
    val_MSE = 0.0
    val_KLD = 0.0
    
    # if epoch == 90:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 1e-4
    
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data[0].float().to(device1)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs, mu, logvar, _ = model(inputs)
        (loss, MSE, KLD)= loss_function(outputs, inputs, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        train_MSE += MSE.item()
        train_KLD += KLD.item()
        optimizer.step()
        
        #progress_percent = np.round(i/(np.floor(344160/512))*1000)
        #print("{:.1f}".format(progress_percent),end='');
    
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].float().to(device1)

            outputs, mu, logvar, _ = model(inputs)
            (loss, MSE, KLD)= loss_function(outputs, inputs, mu, logvar)
            val_loss += loss.item()
            val_MSE += MSE.item()
            val_KLD += KLD.item()

    print("Epoch {}/{}, Train_Loss: {:.9f}".format(epoch+1, num_epochs, train_loss / len(train_loader.dataset)))
    print("Epoch {}/{}, Train_MSE: {:.9f}".format(epoch+1, num_epochs, train_MSE / len(train_loader.dataset)))
    print("Epoch {}/{}, Train_KLD: {:.9f}\n".format(epoch+1, num_epochs, train_KLD / len(train_loader.dataset)))
    
    print("Epoch {}/{}, Val_Loss: {:.9f}".format(epoch+1, num_epochs, val_loss / len(val_loader.dataset)))
    print("Epoch {}/{}, Val_MSE: {:.9f}".format(epoch+1, num_epochs, val_MSE / len(val_loader.dataset)))
    print("Epoch {}/{}, Val_KLD: {:.9f}\n".format(epoch+1, num_epochs, val_KLD / len(val_loader.dataset)))
    train_loss_array[epoch] = train_loss / len(train_loader.dataset)    
    val_loss_array[epoch] = val_loss / len(val_loader.dataset)

print('Finished Training')
elapsed = time.time() - t
print("Elapsed Time", elapsed)

#%%
filename = 'model'+str(flag)+'_hidden'+str(hidden_size)+'_beta'+str(beta)+'_trial'+str(trial)

filename_loss1 = '150epoch/loss/loss_' + filename + '.dat'
np.savetxt(filename_loss1, (train_loss_array,val_loss_array))

filename_loss2 = '150epoch/loss/loss_' + filename + '.mat'
savemat(filename_loss2, {'train_loss_array': train_loss_array, 'val_loss_array': val_loss_array})

filename_model = '150epoch/models/' + filename + '.pt'
torch.save(model, filename_model)