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
import scipy
import matplotlib.pyplot as plt
import sys
import time
from skimage.measure import compare_ssim as ssim
model = keras.models.load_model('modellps.h5')
lpstest_scaled = np.load('lpstest_scaled.npy')
lpstest = np.load('lpstest.npy')
lpstrain = np.load('lpstrain.npy')
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
#0.8423 @ 50 epochs
scandata=scandata[:,:7]
#0.85 @ 50 epochs
scandata=scandata[:,:4]
#0.83197 @ 50 epochs
lpsdata = lps['tcav_lps_ROI'];
randsample = np.random.randint(0,1555)
#%% Predict on training and validation set and plot stuff
dt = 1; # Time resolution From Alberto's TCAV axis conversion [um/pixel]
dE = 1;# Energy resolution [MeV/pix]
x = (np.arange(lpstrain.shape[1])-np.round(lpstrain.shape[1]/2))*dt;
y = (np.arange(lpstrain.shape[2])-np.round(lpstrain.shape[2]/2))*dE
def ssim_cost_function(vp,scandata,lpstestscaled,randsample,x,y,lpsdata,lpstest):
    settings_pv = np.zeros((4,1))
    for i in range(len(vp)):
        settings_pv[i] = (vp[i]-min(scandata[:,i]))/(max(scandata[:,i]))-min(scandata[:,i])
    settings_qv = np.transpose(settings_pv)
    prediction = model.predict(settings_qv)
    predict_lpstestreshaped = np.zeros((lpstestscaled.shape[0],lpsdata.shape[1],lpsdata.shape[2]))
    for i in range(predict_lpstestreshaped.shape[0]):
        predict_lpstestreshaped[i,:] = prediction[0,:].reshape(100,lpsdata.shape[2])
    structuralsimilarity = ssim(lpstestscaled[randsample],predict_lpstestreshaped[randsample])
    cost = 1 - structuralsimilarity
    costex = 0
    
    #for i in range(100):
    #    for j in range(240):
    #        error = (predict_lpstestreshaped[randsample,i,j] - lpstestscaled[randsample,i,j])
    #        costex = costex + error
    cost = cost + costex
    return cost
#def cost_function_lps(vp,scandata,randsample,lpstest):
#    settings_pv = np.zeros((4,1))
#    lpstestleft = lpstest[:,:,:120]
#    lpstestright = lpstest[:,:,120:]
#    for i in range(len(vp)):
#        settings_pv[i] = (vp[i]-min(scandata[:,i]))/(max(scandata[:,i]))-min(scandata[:,i])

#    settings_qv = np.transpose(settings_pv)
#   prediction = model.predict(settings_qv)
    
#Reshape predictions on the test set
#    predict_lpstestreshaped = np.zeros((lpstest.shape[0],lpsdata.shape[1],lpsdata.shape[2]))
#   for i in range(predict_lpstestreshaped.shape[0]):
#        predict_lpstestreshaped[i,:] = prediction[0,:].reshape(100,lpsdata.shape[2])
#    predict_lpstestleft = predict_lpstestreshaped[:,:,:120]
#    predict_lpstestright = predict_lpstestreshaped[:,:,120:]
#    plt.plot(predict_lpstestleft[randsample])
#    error = 0
#    cost = 0
#    for i in range(100):
#        for j in range(240):
#            error = (predict_lpstestreshaped[0,i,j] - lpstest[randsample,i,j])**2
#            cost = cost + error
#    return cost
cost = 2
L1p = scandata[randsample,0]
L2p = scandata[randsample,1]
L1v = scandata[randsample,2]
L2v = scandata[randsample,3]
itera = 0

while cost >= 0.005:
    print(itera)
    Lpv = [L1p,L2p,L1v,L2v]
    
    print(Lpv)
    costlb = lambda pv12: ssim_cost_function(pv12,scandata,lpstest_scaled,randsample,x,y,lpsdata,lpstest)
    [L1popt,L2popt,L1vopt,L2vopt] = scipy.optimize.fmin(costlb,Lpv)
    Lpvopt = [L1popt,L2popt,L1vopt,L2vopt]
    settings_opt_pv = np.zeros((4,1))
    for i in range(len(Lpvopt)):
        settings_opt_pv[i] = (Lpvopt[i]-min(scandata[:,i]))/(max(scandata[:,i]))-min(scandata[:,i])
    settings_qv = np.transpose(settings_opt_pv)
    prediction = model.predict(settings_qv)
    predict_opt_lpstestreshaped = np.zeros((lpstest.shape[0],lpsdata.shape[1],lpsdata.shape[2]))
    for i in range(predict_opt_lpstestreshaped.shape[0]):
        predict_opt_lpstestreshaped[i,:] = prediction[0,:].reshape(100,lpsdata.shape[2])
    structuralsimilarity = ssim(lpstest[randsample],predict_opt_lpstestreshaped[randsample])
    cost = 1 - structuralsimilarity
    

    print(cost)
    #error = 0
    costex = 0
    #for i in range(100):
    #    for j in range(240):
    #        error = (predict_opt_lpstestreshaped[randsample,i,j] - lpstest_scaled[randsample,i,j])
    #        costex = costex + error
    cost = cost + costex
    print(cost)
    L1p = L1popt
    L2p = L2popt
    L1v = L1vopt
    L2v = L2vopt
    
    optimization_plot_lps_vs_prediction_lucretia(lpstest,predict_opt_lpstestreshaped,x,y,randsample)
    itera += 1
#plot_model_history(history)

#predict_lpsreshaped = np.zeros((lpstrain.shape[0],lpsdata.shape[1],lpsdata.shape[2]))
#for i in range(predict_lps_train.shape[0]):
#    predict_lpsreshaped[i,:] = predict_lps_train[i,:].reshape(lpsdata.shape[1],lpsdata.shape[2])
# Reshape predictions on the test set
#predict_lpstestreshaped = np.zeros((lpstest.shape[0],lpsdata.shape[1],lpsdata.shape[2]))
#for i in range(predict_lpstestreshaped.shape[0]):
#    predict_lpstestreshaped[i,:] = predict_lps_test[i,:].reshape(lpsdata.shape[1],lpsdata.shape[2])
    
#for i in range(5):
#    plot_lps_vs_prediction_lucretia(lpstest,predict_lpstestreshaped,x,y)  

# Make a histogram of the score
#score = np.zeros(lpstrainreshaped.shape[0])
#for n in range(lpstrainreshaped.shape[0]):
#    trueval = lpstrainreshaped[n,:];
#    predval = predict_lps_train[n,:];
#    rmse = ((trueval-predval)**2).sum()
#  norm = ((trueval-trueval.mean())**2).sum()
#   score[n]= 1-rmse/norm;
#plt.hist(score)
#plt.show()
#print(np.mean(score))