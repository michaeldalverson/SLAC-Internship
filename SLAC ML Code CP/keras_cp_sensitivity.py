#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 12:24:39 2019
@author: malverson
"""
# 1. Importing data from Lucretia sim
from __future__ import print_function
import numpy as np
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
import scipy.io as sio
import matplotlib.pyplot as plt
import sys, time

sys.path.append(path)
from plotting_functions_twobunch import *
from sklearn import preprocessing
scan_date = '3_29';
path = "/Users/MIchael Alverson/Desktop/SLAC ML Code/"
scandata = sio.loadmat(path+'scandata2bunch_filtered.mat',squeeze_me=True); 
currprofile = sio.loadmat(path+'currprofile_filtered.mat',squeeze_me=True); 
#currprofile = sio.loadmat(path+'currprofile_weighted.mat',squeeze_me=True); 
# Scandata is in the following columns [L1p,L2p,L1v,L2v,Qi,bc11pkI,bc14pkI,IPpkI,bc11.centroidx,bc14.centroidx]
scandata = scandata['scandata2bunch_filtered'];
Iz = currprofile['currprofile_filtered']; 
#Iz = currprofile['currprofile_weighted']; 
def trunc_norm(mu,sigma,ntrunc,nsamples):
    import scipy.stats as stats
    lower, upper = -ntrunc*sigma, ntrunc*sigma
    X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    out = X.rvs(nsamples)
    return out
#%% Add noise in physical units to the predictors before pre-processing data 
# Variables ordering is [L1s phase, L2 phase, L1 amp, L2 amp, BC11 pkI, BC14 pkI, IP pkI, BC11 energy, BC14 energy]
# Noise values are the sigma of the noise you want to add in physical units

#All of the nested for loops were just so I could run several different scenarios and only press run once
    
    
nsims=scandata.shape[0];
total = np.zeros((9,12))
#noise_values = [1000,1000,1000,1000,1000,1000,1000,1000,1000]
#noise_values = [0.005, 0.005, 0.00005, 0.00005, 0.0012459472, 0.133385908, 0.3305109657, 0.00001216562905, 0.00008658611114]
#increase_values = [.11*2,.11*2,.0011*2,.0011*2,0.0274108384*2,2.934489976*2,0.0002676438391*2,0.001904894*2]
for m in range(1):
    #noise_values = [0.005, 0.005, 0.00005, 0.00005, 0.0012459472, 0.133385908, 0.3305109657, 0.00001216562905, 0.00008658611114]
    #increase_values = [.11*2,.11*2,.0011*2,.0011*2,0.0274108384*2,2.934489976*2,0.0002676438391*2,0.001904894*2]
    for k in range(1):
        #if m == 0:
         #   noise_values = [0.001,0.001,0.0001,0.0001,0.001,0.001,0.01,0.001,0.001]
            #all open
        #if m == 1:
         #   noise_values = [100000000,10000000000,10000000000,1000000000000,10000000000,100000000000000,10000000000000,1000000000000,100000000000000]
            #all closed
        #if m==2:
         #   noise_values = [1000000000,100000000,0.0001,0.0001,0.001,0.001,0.01,0.001,0.001]
            #lp block
        #if m==3:
         #   noise_values = [0.001,0.001,100000000000,10000000000,0.001,0.001,0.01,0.001,0.001]
            #lv block
        #if m==4:
         #   noise_values = [0.001,0.001,0.0001,0.0001,10000000000,100000000000,10000000000000,0.001,0.001]
            #pk block
        #if m==5:
         #   noise_values = [0.001,0.001,0.0001,0.0001,0.001,0.001,0.01,10000000,100000000]
            #centroid block
        #if m==6:
         #   noise_values = [10000000000,100000000000000,10000000000000,10000000000,0.001,0.001,0.01,0.001,0.001]
            #lp/lv block
        #if m==7:
         #   noise_values = [100000000,10000000000000,0.0001,0.0001,1000000000,100000000000,1000000000,0.001,0.001]
            #lp/pk block
        #if m==8:
         #   noise_values = [100000000,10000000000,0.0001,0.0001,0.001,0.001,0.01,1000000000,10000000000]
            #lp/centroid block
        #noise_values = [100000,100000,100000,100000,100000,100000,100000,100000,100000]
        noise_values = [0.0001,0.0001,0.0001,0.0001,0.001,0.0001,0.001,0.001,0.001]
        #noise_values[k] = 1000000000
    #noise_values = [100000000,1000000000,0.0001,0.0001,0.001,0.001,0.01,0.001,0.001]
    #noise_values = [100000000,100000000,0.0001,0.0001,0.001,0.001,0.01,1000000000,1000000000]
    

        X = np.empty([nsims,9])
        for i in range(X.shape[1]):
            noise = trunc_norm(0,noise_values[i],2,nsims);
            X[:,i]=scandata[:,i]+noise;     

# Now choose a number of random training and test shots
        ntrain = int(np.round(nsims*0.8));
        ntest = int(np.round(nsims*0.2));
# Randomly index your shots for traning and test sets
        idx = np.random.permutation(nsims);
        idxtrain = idx[0:ntrain];
        idxtest = idx[ntrain:ntrain+ntest];
# Normalize the current
        Iz_scaled = Iz/np.max(Iz) 
        Iz_train_scaled = Iz_scaled[idxtrain,:]
        Iz_test_scaled = Iz_scaled[idxtest,:]

# Scale the input data between -1 and 1 
        X_train_scaled = np.zeros((ntrain,X.shape[1]));
        X_test_scaled = np.zeros((ntest,X.shape[1]));
        scale_x = preprocessing.MinMaxScaler(feature_range=(0,1))
        for i in range(X.shape[1]):
            x1 = X[:,i];
            x2 = x1.reshape(-1,1);
            X_pv = scale_x.fit_transform(x2);
            X_train_scaled[:,i] = X_pv[idxtrain,0]
            X_test_scaled[:,i] = X_pv[idxtest,0]
        print(X_test_scaled)
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
        epochs      = 250;
        history = model.fit(X_train_scaled, Iz_train_scaled, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(X_test_scaled,Iz_test_scaled))
        print('Time to perform fit [mins] =  {0:.3f}' .format((time.time() - start_time)/60))
#%% Predict on training and validation set and plot performance
        predict_Iz_train = model.predict(X_train_scaled)
        predict_Iz_test = model.predict(X_test_scaled)
        plot_model_history(history)
        for i in range(10):
            plot_2bunch_prediction_vs_lucretia(Iz_test_scaled,predict_Iz_test,np.max(Iz))
            preset_plot_2bunch_prediction_vs_lucretia(Iz_test_scaled,predict_Iz_test,np.max(Iz),1169)

# Make a histogram of the score
        score = np.zeros(Iz_test_scaled.shape[0])
        for n in range(Iz_test_scaled.shape[0]):
            trueval = Iz_test_scaled[n,:];
            predval = predict_Iz_test[n,:];
            rmse = ((trueval-predval)**2).sum()
            norm = ((trueval-trueval.mean())**2).sum()
            score[n]= 1-rmse/norm;
        plt.hist(score)
        plt.show()
        print(np.mean(score))

        #Calculating the R^2 value
        accuracy = 0;
        numer = 0
        denom = 0
        numersum = 0
        denomsum = 0
        for i in range(1555):
            for j in range(270):
    
                numer = (predict_Iz_test[i,j] - Iz_test_scaled[i,j])**2
                denom = (Iz_test_scaled[i,j] - Iz_test_scaled[:,j].mean())**2
                numersum = numer + numersum
                denomsum = denomsum + denom
        #print(accuracy)
        #noise_values[m] = noise_values[m] + increase_values[m]
        Rsqr = 1 - (numersum/denomsum)
        #total[m,k] = accuracy
        #print(total)
        #noise_values[m] = noise_values[m] + (increase_values[m] * k)
    #print(noise_values)