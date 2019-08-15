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

# 2. Importing model and reference
model = keras.models.load_model('model.h5')
Iz_test_scaled = np.load('Iz_test_scaled.npy')

# 3. Importing simulation data for reference
from plotting_functions_twobunch import *
from sklearn import preprocessing
scan_date = '3_29';
path = "/Users/MIchael Alverson/Desktop/SLAC ML Code/"
scandata = sio.loadmat(path+'scandata2bunch_filtered.mat',squeeze_me=True); 
currprofile = sio.loadmat(path+'currprofile_filtered.mat',squeeze_me=True); 
scandata = scandata['scandata2bunch_filtered'];
Iz = currprofile['currprofile_filtered']; 

randsample = np.random.randint(0,1555)
Iz_sample = Iz[randsample,:]
scandata = scandata[:,0:4]
#currprof(Iz_test_scaled,np.max(Iz),randsample)
####################################################################################################
# 4. Creating cost function for minimization
def cost_function(pv,scandata,randsample,Iz_test_scaled,L1pplt1,L1pplt2,L1pplt3,L2pplt1,L2pplt2,L2pplt3,L1vplt1,L1vplt2,L1vplt3,L2vplt1,L2vplt2,L2vplt3,L1porg,L2porg,L1vorg,L2vorg,costplt):
        settings_pv = np.zeros((4,1))
    #################################################
    #Rescaling data between 0 and 1
        for i in range(len(pv)):
          settings_pv[i] = (pv[i] - min(scandata[:,i]))/(max(scandata[:,i]) - min(scandata[:,i]))
    #################################################
        settings_qv = np.transpose(settings_pv)
        prediction = model.predict(settings_qv)
        #optimizationv2_plot_2bunch_prediction_vs_lucretia(Iz_test_scaled,prediction,np.max(Iz),randsample)
        #plt.figure()
        L1p = pv[0]
        L2p = pv[1]
        L1v = pv[2]
        L2v = pv[3]
        #Creating lists to track the progress of the settings in the optimizer
        #L1pplt.append(L1p)
        #L2pplt.append(L2p)
        #L1vplt.append(L1v)
        #L2vplt.append(L2v)
    ############################################
    #Plotting settings with each iteration
        #if itercolor < 4:
        #    plt.figure(1)
        #    plt.subplot(211)
        #    if itercolor == 1:
        #        L1pplt1.append(L1p)
        #        L2pplt1.append(L2p)
        #        plt.plot(L1pplt1,L2pplt1,color='orange')
        #        plt.xlabel('L1 phase')
        #        plt.ylabel('L2 phase')
        #    if itercolor == 2:
        #        L1pplt2.append(L1p)
        #        L2pplt2.append(L2p)
        #        plt.plot(L1pplt1,L2pplt1,color='orange')
        #        plt.plot(L1pplt2,L2pplt2,color = 'purple')
        #        plt.xlabel('L1 phase')
        #        plt.ylabel('L2 phase')
        #    if itercolor == 3:
        #        L1pplt3.append(L1p)
        #        L2pplt3.append(L2p)
        #        plt.plot(L1pplt1,L2pplt1,color='orange')
        #        plt.plot(L1pplt2,L2pplt2,color = 'purple')
        #        plt.plot(L1pplt3,L2pplt3,color = 'magenta')
        #        plt.xlabel('L1 phase')
        #        plt.ylabel('L2 phase')
        #    plt.plot(L1p,L2p,marker = 'o',markersize = 6, color = 'blue')
        #    plt.plot(L1porg,L2porg, marker = 'o', markersize = 6,color='red')
        #    plt.plot(scandata[randsample,0],scandata[randsample,1],marker = 'o', markersize = 6,color = 'yellow')

        #    plt.subplot(212)
        #    if itercolor == 1:
        #        L1vplt1.append(L1v)
        #        L2vplt1.append(L2v)
        #        plt.plot(L1vplt1,L2vplt1,color='orange')
        #        plt.xlabel('Top - L1 phase, Bottom - L1 voltage')
        #        plt.ylabel('L2 voltage')
        #    if itercolor == 2:
        #        L1vplt2.append(L1v)
        #        L2vplt2.append(L2v)
        #        plt.plot(L1vplt1,L2vplt1,color='orange')
        #        plt.plot(L1vplt2,L2vplt2,color = 'purple')
        #        plt.xlabel('Top - L1 phase, Bottom - L1 voltage')
        #        plt.ylabel('L2 voltage')
        #    if itercolor == 3:
        #        L1vplt3.append(L1v)
        #        L2vplt3.append(L2v)
        #        plt.plot(L1vplt1,L2vplt1,color='orange')
        #        plt.plot(L1vplt2,L2vplt2,color = 'purple')
        #        plt.plot(L1vplt3,L2vplt3,color = 'magenta')
        #        plt.xlabel('Top - L1 phase, Bottom - L1 voltage')
        #        plt.ylabel('L2 voltage')
        #    plt.plot(L1v,L2v,marker = 'o',markersize = 6,color = 'blue')
        #    plt.plot(L1vorg,L2vorg,marker = 'o',markersize = 6,color='red')
        #    plt.plot(scandata[randsample,2],scandata[randsample,3],marker = 'o',markersize = 6,color = 'yellow')
        #    plt.show()
        
    ##############################################3
        #plt.plot(prediction[0,:])  
        #plt.plot(Iz_test_scaled[randsample,:])
        #plt.show()
        error = 0
        cost = 0
        #Calculating the cost function for the optimizer function
        for i in range(270):
            error = (prediction[0,i] - Iz_test_scaled[randsample,i])**2
            cost = cost + error
        costplt.append(cost)
        #plt.figure(2)
        #plt.plot(range(len(costplt)),costplt)
        return cost
    #%%
#####################################################################################################
# 5. Running optimizer and minimizing the cost function
cost = 0.5
grad = 0.5
pcost = 0
L1p = 50
L2p = 50
L1v = 50
L2v = 50
L1porg = L1p
L2porg = L2p
L1vorg = L1v
L2vorg = L2v
itera = 0
check = np.zeros((100,1))
#These variables track the settings within the optimizer
L1pplt1 = []
L1pplt2 = []
L1pplt3 = []
L2pplt1 = []
L2pplt2 = []
L2pplt3 = []
L1vplt1 = []
L1vplt2 = []
L1vplt3 = []
L2vplt1 = []
L2vplt2 = []
L2vplt3 = []
costplt = []
itercolor = 0
while cost >= 0.01 and itera <= 100:
##########################################################################################################
    #This is what changes the color of the settings line
    itercolor = itercolor + 1
    pv = [L1p,L2p,L1v,L2v]
    #Placeholder for the pv variable
    costlb = lambda Lpv: cost_function(Lpv,scandata,randsample,Iz_test_scaled,L1pplt1,L1pplt2,L1pplt3,L2pplt1,L2pplt2,L2pplt3,L1vplt1,L1vplt2,L1vplt3,L2vplt1,L2vplt2,L2vplt3,L1porg,L2porg,L1vorg,L2vorg,costplt)
    [L1popt, L2popt, L1vopt,L2vopt] = scipy.optimize.fmin(costlb, x0 = pv)
    ############################################
    #new settings
    pvopt = [L1popt,L2popt,L1vopt,L2vopt]
    pvopt_pv = np.transpose(pvopt)
    settings_opt_pv = np.zeros((4,1))
    #Rescaling
    for i in range(len(pvopt)):
        settings_opt_pv[i] = (pvopt[i] - min(scandata[:,i]))/(max(scandata[:,i]) - min(scandata[:,i]))
    settings_opt_qv = np.transpose(settings_opt_pv)
    prediction_opt = model.predict(settings_opt_qv)
    #Calculating new cost for the while loop
    error = 0
    cost = 0
    for i in range(270):
        error = (prediction_opt[0,i] - Iz_test_scaled[randsample,i])**2
        cost = cost + error

    L1p = L1popt
    L2p = L2popt
    L1v = L1vopt
    L2v = L2vopt
    ############################################
    #This is an incomplete version of plotting the setting with each iteration
    #Plotting settings with each iteration
    #plt.figure(1)
    #plt.subplot(211)
    #plt.plot(v1plt,v2plt)
    #plt.plot(v1org,v2org, marker = 'o', markersize = 6,color='red')
    #plt.plot(scandata[randsample,0],scandata[randsample,1],marker = 'o', markersize = 6,color = 'yellow')

    #plt.subplot(212)
    #plt.plot(L1pplt,L2pplt)
    #plt.plot(L1porg,L2porg,marker = 'o',markersize = 6,color='red')
    #plt.plot(scandata[randsample,2],scandata[randsample,3],marker = 'o',markersize = 6,color = 'yellow')
    ############################################
    #The following code is what "bumps" the optimizer when it gets stuck
    pcost = cost
    itera += 1
    if itera%5 == 0 and itera != 0:
        L1v = np.random.uniform(-0.005,0.005)
        L2v = np.random.uniform(-0.005,0.005)
        if itera == 30 or itera == 60 or itera == 90:
            L1p = np.random.uniform(-19.7,-18.7)
            L2p = np.random.uniform(-38.85,-37.85)
        best = 10
        if cost < best:
            best = cost
            prediction_best = prediction_opt
            L2p_best = L2p
            L1p_best = L1p
            L1v_best = L1v
            L2v_best = L2v
    #This resets all of the values for the settings to be tracked in the optimizer
    if itercolor == 4:
            itercolor = 0
            L1pplt1 = []
            L1pplt2 = []
            L1pplt3 = []
            L2pplt1 = []
            L2pplt2 = []
            L2pplt3 = []
            L1vplt1 = []
            L1vplt2 = []
            L1vplt3 = []
            L2vplt1 = []
            L2vplt2 = []
            L2vplt3 = []
    #A variation of the original plot that takes the shot number as input so I get results from
    #the same shot every time
    optimization_plot_2bunch_prediction_vs_lucretia(Iz_test_scaled,prediction_opt,np.max(Iz),randsample)