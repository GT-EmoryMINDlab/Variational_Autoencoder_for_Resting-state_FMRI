# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:00:23 2020

@author: Xiaodi
"""

from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


#%%---------------------------------------
class VAE_Temporal_Conv(nn.Module):
    def __init__(self, hidden_size):
        super(VAE_Temporal_Conv, self).__init__()
        self.conv_i1 = nn.Conv1d(in_channels=246, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.conv_i2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.conv_i3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=2)
        self.fc_i4 = nn.Linear(64*6, 256)
        self.fc_i5a = nn.Linear(256, hidden_size)
        self.fc_i5b = nn.Linear(256, hidden_size)
        
        self.fc_o5 = nn.Linear(hidden_size, 256)
        self.fc_o4 = nn.Linear(256, 64*6)
        self.conv_o3 = nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=2)
        self.conv_o2 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.conv_o1 = nn.ConvTranspose1d(in_channels=128, out_channels=246, kernel_size=5, stride=2, padding=2)
        
    def encode(self, x):
        h1 = self.conv_i1(x)
        h2 = self.conv_i2(F.relu(h1))
        h3 = self.conv_i3(F.relu(h2))
        h3_flatten = torch.flatten(h3,start_dim=1)
        h4 = self.fc_i4(F.relu(h3_flatten))
        h5a = self.fc_i5a(F.relu(h4))
        h5b = self.fc_i5b(F.relu(h4))
        return h5a, h5b

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        if self.training:
            out = mu + eps*std
        else:
            out = mu    
        return out

    def decode(self, z):
        h5 = self.fc_o5(z)
        h4 = self.fc_o4(F.relu(h5))
        dim = h4.shape
        h4_reshape = torch.reshape(h4, [dim[0],64,6])
        h3 = self.conv_o3(F.relu(h4_reshape))
        h2 = self.conv_o2(F.relu(h3))
        h1 = self.conv_o1(F.relu(h2))
        return h1

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

#%%---------------------------------------
class VAE_Temporal_Conv_Deep(nn.Module):
    def __init__(self, hidden_size):
        super(VAE_Temporal_Conv_Deep, self).__init__()
        self.conv_i1 = nn.Conv1d(in_channels=246, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.conv_i2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv_i3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.conv_i4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv_i5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=2)
        self.fc_i6 = nn.Linear(64*6, 256)
        self.fc_i7 = nn.Linear(256, 128)
        self.fc_i8a = nn.Linear(128, hidden_size)
        self.fc_i8b = nn.Linear(128, hidden_size)
        
        self.fc_o8 = nn.Linear(hidden_size, 128)
        self.fc_o7 = nn.Linear(128, 256)
        self.fc_o6 = nn.Linear(256, 64*6)
        self.conv_o5 = nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=2)
        self.conv_o4 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv_o3 = nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.conv_o2 = nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv_o1 = nn.ConvTranspose1d(in_channels=256, out_channels=246, kernel_size=5, stride=2, padding=2)
        
    def encode(self, x):
        h1 = self.conv_i1(x)
        h2 = self.conv_i2(F.relu(h1))
        h3 = self.conv_i3(F.relu(h2))
        h4 = self.conv_i4(F.relu(h3))
        h5 = self.conv_i5(F.relu(h4))
        h5_flatten = torch.flatten(h5,start_dim=1)
        h6 = self.fc_i6(F.relu(h5_flatten))
        h7 = self.fc_i7(F.relu(h6))
        h8a = self.fc_i8a(F.relu(h7))
        h8b = self.fc_i8b(F.relu(h7))
        return h8a, h8b

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        if self.training:
            out = mu + eps*std
        else:
            out = mu    
        return out

    def decode(self, z):
        h8 = self.fc_o8(z)
        h7 = self.fc_o7(F.relu(h8))
        h6 = self.fc_o6(F.relu(h7))
        dim = h6.shape
        h6_reshape = torch.reshape(h6, [dim[0],64,6])
        h5 = self.conv_o5(F.relu(h6_reshape))
        h4 = self.conv_o4(F.relu(h5))
        h3 = self.conv_o3(F.relu(h4))
        h2 = self.conv_o2(F.relu(h3))
        h1 = self.conv_o1(F.relu(h2))
        return h1

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z 

#%%---------------------------------------
class VAE_Temporal_Conv_Shallow(nn.Module):
    def __init__(self, hidden_size):
        super(VAE_Temporal_Conv_Shallow, self).__init__()
        self.fc_i1 = nn.Linear(246*33, 256)
        self.fc_i2a = nn.Linear(256, hidden_size)
        self.fc_i2b = nn.Linear(256, hidden_size)
        
        self.fc_o2 = nn.Linear(hidden_size, 256)
        self.fc_o1 = nn.Linear(256, 246*33)
        
    def encode(self, x):
        x_flatten = torch.flatten(x,start_dim=1)
        h1 = self.fc_i1(x_flatten)      
        h2a = self.fc_i2a(F.relu(h1))
        h2b = self.fc_i2b(F.relu(h1))
        return h2a, h2b

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        if self.training:
            out = mu + eps*std
        else:
            out = mu    
        return out

    def decode(self, z):             
        h2 = self.fc_o2(z)                
        h1 = self.fc_o1(F.relu(h2))
        dim = h1.shape
        h1_reshape = torch.reshape(h1, [dim[0],246,33])
        return h1_reshape

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z