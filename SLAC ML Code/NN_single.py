# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:23:33 2019

@author: MIchael Alverson
"""

# 1. Importing for TensorFlow
from __future__ import print_function
import numpy as np
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
import scipy.io as sio
import scipy
import matplotlib.pyplot as plt
import sys, time

# 2. Importing simulation data for reference

from plotting_functions_twobunch import *
from sklearn import preprocessing
scan_date = '3_29';
path = "/Users/MIchael Alverson/Desktop/SLAC ML Code/"
scandata = sio.loadmat(path+'scandata2bunch_filtered.mat',squeeze_me=True); 
currprofile = sio.loadmat(path+'currprofile_filtered.mat',squeeze_me=True); 

scandata = scandata['scandata2bunch_filtered'];
Iz = currprofile['currprofile_filtered']; 
#
def trunc_norm(mu,sigma,ntrunc,nsamples):
    import scipy.stats as stats
    lower, upper = -ntrunc*sigma, ntrunc*sigma
    X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    out = X.rvs(nsamples)
    return out
#
noise_values = [0.001,0.001,0.0001,0.0001,0.0001,0.0001,0.001]
randsample = np.random.randint(0,1555)
Iz_sample = Iz[randsample,:]
scandata = scandata[:,0:4]

nsims=scandata.shape[0];
X = np.empty([nsims,4])
#
for i in range(X.shape[1]):
    noise = trunc_norm(0,noise_values[i],2,nsims);
    X[:,i]=scandata[:,i]+noise; 
#
# 3. Now choose a number of random training and test shots
ntrain = int(np.round(nsims*0.8));
ntest = int(np.round(nsims*0.2));

# 4. Randomly index your shots for traning and test sets
idx = np.random.permutation(nsims);
idxtrain = idx[:ntrain];
idxtest = idx[ntrain:];

# 5. Normalize the current
Iz_scaled = Iz/np.max(Iz) 
Iz_train_scaled = Iz_scaled[idxtrain,:]
Iz_test_scaled = Iz_scaled[idxtest,:]

# Scale the input data between -1 and 1 
X_train_scaled = np.zeros((ntrain,X.shape[1]));
X_test_scaled = np.zeros((ntest,X.shape[1]));
#
scale_x = preprocessing.MinMaxScaler(feature_range=(0,1))
for i in range(X.shape[1]):
    x1 = X[:,i];
    x2 = x1.reshape(-1,1);
    X_pv = scale_x.fit_transform(x2);
    X_train_scaled[:,i] = X_pv[idxtrain,0]
    X_test_scaled[:,i] = X_pv[idxtest,0]
#
####################################################################################################
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
model.add(Dense(Iz_train_scaled.shape[1], activation='linear'))# Output layer must have the shape of your current profile
print(model.summary())
# 'Configure the learning process' by compiling For a mean squared error regression problem
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#%%  Train the model 
start_time = time.time();
batch_size  = 2**6;
epochs      = 5;
history = model.fit(X_train_scaled, Iz_train_scaled, batch_size=batch_size, epochs=epochs, verbose=1,
                    validation_data=(X_test_scaled,Iz_test_scaled))
print('Time to perform fit [mins] =  {0:.3f}' .format((time.time() - start_time)/60))
####################################################################################################

# 8. Predict on training and validation set and plot performance
predict_Iz_train = model.predict(X_train_scaled)
predict_Iz_test = model.predict(X_test_scaled)

np.save('model',model)
np.save('model',model)