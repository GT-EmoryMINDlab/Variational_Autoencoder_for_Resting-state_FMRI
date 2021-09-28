# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:58:33 2021

@author: Xiaodi
"""
import torch
import torch.utils.data
import numpy as np
import models
from scipy.io import loadmat, savemat
from torch import optim
from torch.nn import functional as F
import time

class VAE_model:
    def __init__(self, model_flag, beta, hidden_size, trial, num_epochs, batch_size, learning_rate):
        if torch.cuda.is_available():
            self.device1 = torch.device("cuda")
        else:
            self.device1 = torch.device("cpu")
        self.device2 = torch.device("cpu")
    
        self.beta = beta
        self.hidden_size = hidden_size
        self.trial = trial
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    
        self.foldername = str(num_epochs) + 'epoch/'
        self.filename = 'model'+str(model_flag)+'_hidden'+str(hidden_size)+'_beta'+str(beta)+'_trial'+str(trial)
    
        if model_flag == 1:
            self.model = models.VAE_Temporal_Conv_Shallow(hidden_size)
        elif model_flag == 2:
            self.model = models.VAE_Temporal_Conv(hidden_size)
        elif model_flag == 3:
            self.model = models.VAE_Temporal_Conv_Deep(hidden_size)
   
    def data_partition(self):
        temp = loadmat('Resting_State_GSR_segments.mat')
            
        X_train = torch.from_numpy(temp['X_train'])
        X_val = torch.from_numpy(temp['X_val'])
        X_test = torch.from_numpy(temp['X_test'])
        
        train_dataset = torch.utils.data.TensorDataset(X_train)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                         batch_size=self.batch_size, shuffle=True, num_workers = 0)
        
        val_dataset = torch.utils.data.TensorDataset(X_val)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                         batch_size=self.batch_size, shuffle=True, num_workers = 0)
        
        test_dataset = torch.utils.data.TensorDataset(X_test)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                         batch_size=self.batch_size, shuffle=False, num_workers = 0)
        
        N_train, N_val, N_test = X_train.shape[0], X_val.shape[0], X_test.shape[0]
        
        return N_train, N_val, N_test, train_loader, val_loader, test_loader

    #%%
    def training(self):        
        def loss_function(recon_x, x, mu, logvar):
            MSE = F.mse_loss(recon_x, x, reduction = 'sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return (1*MSE + self.beta*KLD, MSE, KLD)
        
        def train_epoch():
            nonlocal train_loss, train_MSE, train_KLD
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data[0].float().to(self.device1)
        
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs, mu, logvar, _ = self.model(inputs)
                (loss, MSE, KLD)= loss_function(outputs, inputs, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                train_MSE += MSE.item()
                train_KLD += KLD.item()
                optimizer.step()
                
        def validation_epoch():
            nonlocal val_loss, val_MSE, val_KLD
            with torch.no_grad():
                for i, data in enumerate(val_loader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs = data[0].float().to(self.device1)
        
                    outputs, mu, logvar, _ = self.model(inputs)
                    (loss, MSE, KLD)= loss_function(outputs, inputs, mu, logvar)
                    val_loss += loss.item()
                    val_MSE += MSE.item()
                    val_KLD += KLD.item()
        
        def save_model():        
            filename_loss1 = self.foldername + 'loss/loss_' + self.filename + '.dat'
            np.savetxt(filename_loss1, (train_loss_array,val_loss_array))
            
            filename_loss2 = self.foldername + 'loss/loss_' + self.filename + '.mat'
            savemat(filename_loss2, {'train_loss_array': train_loss_array, 'val_loss_array': val_loss_array})
            
            filename_model = self.foldername + 'models/' + self.filename + '.pt'
            torch.save(self.model, filename_model)
           
        #----------------------
        # main
        self.model.train()
        self.model.to(self.device1)
        
        _, _, _, train_loader, val_loader, _ = self.data_partition()
        
        train_loss_array = np.zeros(self.num_epochs)
        val_loss_array = np.zeros(self.num_epochs)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        t = time.time()
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            train_loss, train_MSE, train_KLD = 0.0, 0.0, 0.0
            val_loss, val_MSE, val_KLD = 0.0, 0.0, 0.0
    
            train_epoch()
            validation_epoch()
        
            print("Epoch {}/{}, Train_Loss: {:.9f}".format(epoch+1, self.num_epochs, train_loss / len(train_loader.dataset)))
            print("Epoch {}/{}, Train_MSE: {:.9f}".format(epoch+1, self.num_epochs, train_MSE / len(train_loader.dataset)))
            print("Epoch {}/{}, Train_KLD: {:.9f}\n".format(epoch+1, self.num_epochs, train_KLD / len(train_loader.dataset)))
            
            print("Epoch {}/{}, Val_Loss: {:.9f}".format(epoch+1, self.num_epochs, val_loss / len(val_loader.dataset)))
            print("Epoch {}/{}, Val_MSE: {:.9f}".format(epoch+1, self.num_epochs, val_MSE / len(val_loader.dataset)))
            print("Epoch {}/{}, Val_KLD: {:.9f}\n".format(epoch+1, self.num_epochs, val_KLD / len(val_loader.dataset)))
            train_loss_array[epoch] = train_loss / len(train_loader.dataset)    
            val_loss_array[epoch] = val_loss / len(val_loader.dataset)
        
        print('Finished Training')
        elapsed = time.time() - t
        print("Elapsed Time", elapsed)
        
        save_model()
        print('training ' + self.filename + ' complete')
        
    #%%
    def testing(self, n = 100):
        def reconstruction():
            z_array=torch.zeros(N_test,self.hidden_size)
            out_img=torch.zeros(N_test,246,33)
            in_img=torch.zeros(N_test,246,33)
            
            with torch.no_grad():  
                for i, data in enumerate(test_loader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs = data[0].float().to(self.device1)
                    l = inputs.shape[0]
                    
                    outputs, mu, logvar, z = self.model(inputs)
                    
                    z_array[i*self.batch_size:i*self.batch_size+l,:] = z
                    out_img[i*self.batch_size:i*self.batch_size+l,:] = outputs
                    in_img[i*self.batch_size:i*self.batch_size+l,:] = inputs
            
            z_array = z_array.to(self.device2).numpy()
            out_img = out_img.to(self.device2).numpy()
            in_img = in_img.to(self.device2).numpy() 
            return z_array, out_img, in_img
            
        def perturbation():
            grid_x = torch.linspace(-3, 3, n)
            x_decoded_array = np.zeros((n,246,33,self.hidden_size))
            
            with torch.no_grad(): 
                for i in range(self.hidden_size):
                    z_sample = torch.zeros(n,self.hidden_size).to(self.device1)
                    z_sample[:,i] = grid_x
                    x_decoded = self.model.decode(z_sample)
                    x_decoded_array[:,:,:,i] = x_decoded.cpu().numpy()
            
            return x_decoded_array, grid_x
                
        def save_data():
            save_dict = {'z_array': z_array, 'in_img': in_img, 'out_img': out_img,
                         'grid_x': grid_x.numpy(), 'x_decoded_array': x_decoded_array}
            savemat(self.foldername + 'encoded_data/Encoded_Z_GSR_' + self.filename +'.mat', save_dict)
            print('testing ' + self.filename + ' complete')
        
        #----------------------
        # main
        self.model.eval()
        self.model.to(self.device1)
        
        _, _, N_test, _, _, test_loader = self.data_partition()
        
        z_array, out_img, in_img = reconstruction()
        
        x_decoded_array, grid_x = perturbation()
        
        save_data()