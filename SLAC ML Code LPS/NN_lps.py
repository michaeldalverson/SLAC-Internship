#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:24:39 2019
@author: cemma
"""
# 1. Importing data from Lucretia sim
from __future__ import print_function
import numpy as np
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import time

from plotting_functions_twobunch import *
from plotting_lps_fcns import *
from sklearn import preprocessing
scan_date = '3_29';
path = "/Users/MIchael Alverson/Desktop/SLAC ML Code/LPS Code/"
scandata = sio.loadmat(path+'scandata2bunch_filtered.mat',squeeze_me=True); 
lps = sio.loadmat(path+'twobunch_2019_'+scan_date+'_LPS_ROI.mat',squeeze_me=True); 
sys.path.append(path)
# Scandata is in the following columns [L1p,L2p,L1v,L2v,Qi,bc11pkI,bc14pkI,IPpkI,bc11.centroidx,bc14.centroidx]
scandata = scandata['scandata2bunch_filtered'];
lpsdata = lps['tcav_lps_ROI'];
def trunc_norm(mu,sigma,ntrunc,nsamples):
    import scipy.stats as stats
    lower, upper = -ntrunc*sigma, ntrunc*sigma
    X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    out = X.rvs(nsamples)
    return out
#%%# Add noise in physical units to the predictors before pre-processing data 
nsims=scandata.shape[0];
#noise_values = [0.001,0.001,0.0001,0.0001,0.001,0.001,0.01,0.001,0.001]
noise_values = [0.001,0.001,0.0001,0.0001]
X = np.empty([nsims,4])
for i in range(X.shape[1]):
    noise = trunc_norm(0,noise_values[i],2,nsims);
    X[:,i]=scandata[:,i]+noise; 
# Now choose a number of random training, validation and test shots
ntrain = int(np.round(nsims*0.8));
ntest = int(np.round(nsims*0.2));
# Randomly index your shots for traning and test sets
idx = np.random.permutation(nsims);
idxtrain = idx[0:ntrain];
idxtest = idx[ntrain:ntrain+ntest];
# Outputs in an array of size ntrain * 2
lpstrain = np.zeros((ntrain-1,lpsdata.shape[1],lpsdata.shape[2]));
lpstrain=lpsdata[idxtrain,:]; #LPS data for training
# Test data outputs in an array of size ntest
lpstest = np.zeros((ntest,lpsdata.shape[1],lpsdata.shape[2]));
lpstest=lpsdata[idxtest,:]; #LPS data for test
# Normalize the lps
lpstrain_scaled = lpstrain/np.max(lpsdata)
lpstest_scaled = lpstest/np.max(lpsdata)

#Scale the input data between 0 and 1 
X_train_scaled = np.zeros((ntrain,X.shape[1]));
X_test_scaled = np.zeros((ntest,X.shape[1]));
scale_x = preprocessing.MinMaxScaler(feature_range=(0,1))
for i in range(X.shape[1]):
    X_pv = scale_x.fit_transform(X[:,i].reshape(-1,1));
    X_train_scaled[:,i] = X_pv[idxtrain,0]
    X_test_scaled[:,i] = X_pv[idxtest,0]

# Reshape the lps image array so it's in the format for Tensorflow - the two lines below are if you wanna feed the lps to a CNN for an inverse model
#lpstrainreshaped = lpstrain.reshape(lpstrain.shape[0], lpstrain.shape[1], lpstrain.shape[2], 1)
#lpstestreshaped = lpstest.reshape(lpstest.shape[0], lpstest.shape[1], lpstest.shape[2], 1)

# Reshape the 3d array into a 2d array so you can use it in the scikitlearn neural net
lpstrainreshaped = np.zeros((lpstrain.shape[0],lpsdata.shape[1]*lpsdata.shape[2]))
lpstestreshaped = np.zeros((lpstest.shape[0],lpsdata.shape[1]*lpsdata.shape[2]))
for i in range(lpstrain.shape[0]):
    lpstrainreshaped[i,:] = lpstrain_scaled[i,:].reshape(lpsdata.shape[1]*lpsdata.shape[2])    
for i in range(lpstest.shape[0]):
    lpstestreshaped[i,:] = lpstest_scaled[i,:].reshape(lpsdata.shape[1]*lpsdata.shape[2])

# Now train the NN to predict 2d 
dt = 1; # Time resolution From Alberto's TCAV axis conversion [um/pixel]
dE = 1;# Energy resolution [MeV/pix]
x = (np.arange(lpstrain.shape[1])-np.round(lpstrain.shape[1]/2))*dt;
y = (np.arange(lpstrain.shape[2])-np.round(lpstrain.shape[2]/2))*dE;
# Plot lps for a few random shots
for i in range(1):
    plot_lps_vs_prediction_lucretia(lpstrain,lpstrain,x,y) 
#%%# Set up the Tensorflow environment to use only one thread.
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
set_session(tf.Session(config=session_conf))

# Build the MLP to train
model = Sequential()
model.add(Dense(500, activation='relu',input_shape = (X_train_scaled.shape[1],)))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
#model.add(LeakyReLU(100,alpha=0.05))

#model.add(Dense(500, activation='relu',input_shape = (X_train_scaled.shape[1],)))
#model.add(Dense(500, activation='relu'))
#model.add(Dense(200, activation='relu'))
#model.add(Dense(100, activation='relu'))

model.add(Dense(lpstrainreshaped.shape[1], activation='linear'))# Output layer must have the shape of your current profile


print(model.summary())
# 'Configure the learning process' by compiling
# For a mean squared error regression problem
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])
#%% # Train the model (AKA fit)
start_time = time.time()
batch_size  = 2**6;
epochs      = 100;
# Note in this case your x_train and x_test are the amplitude and phases of the 
# linac sections - you wanna be able to train the NN to predict those based on the LPS image
history = model.fit(X_train_scaled, lpstrainreshaped, batch_size=batch_size, epochs=epochs, verbose=1,
         validation_data=(X_test_scaled, lpstestreshaped))
print('Time to perform fit: %s seconds' % (time.time() - start_time))

model.save('modellps.h5')
np.save('lpstest_scaled',lpstest_scaled)