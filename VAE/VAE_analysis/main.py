# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:20:56 2021

@author: Xiaodi
"""
import myVAE

num_epochs = 90
batch_size = 512
learning_rate = 0.001
model_flag = 3
hidden_size = 32
trial = 3
beta = 4
# #%% training and testing
# for i in range(6):
#     if i == 0:
#         beta = 0
#     elif i == 1:
#         beta = 1
#     elif i == 2:
#         beta = 2
#     elif i == 3:
#         beta = 4
#     elif i == 4:
#         beta = 8
#     elif i == 5:
#         beta = 16
        
#     mymodel = myVAE.VAE_model(model_flag, beta, hidden_size, trial, num_epochs, batch_size, learning_rate)
#     mymodel.training()
#     mymodel.testing()
  
#%% get num of parameter
mymodel = myVAE.VAE_model(model_flag, beta, hidden_size, trial, num_epochs, batch_size, learning_rate)
total = 0
for name, parameter in mymodel.model.named_parameters():
    param = parameter.numel()
    total += param
    print(name, param)
    
print(total)