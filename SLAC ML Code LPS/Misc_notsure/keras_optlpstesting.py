#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue July 30 12:24:39 2019
@author: malverson
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
lpsdata = lps['tcav_lps_ROI'];
# Scandata is in the following columns [L1p,L2p,L1v,L2v,Qi,bc11pkI,bc14pkI,IPpkI,bc11.centroidx,bc14.centroidx]
scandata = scandata['scandata2bunch_filtered'];
#Reduce scandata to L1 phase, L2 phase, L1 voltage, and L2 voltage for optimization purposes
scandata=scandata[:,:4]
randsample = np.random.randint(0,1555)
#Create variables needed to visualize LPS
dt = 1; # Time resolution From Alberto's TCAV axis conversion [um/pixel]
dE = 1;# Energy resolution [MeV/pix]
x = (np.arange(lpstrain.shape[1])-np.round(lpstrain.shape[1]/2))*dt;
y = (np.arange(lpstrain.shape[2])-np.round(lpstrain.shape[2]/2))*dE

#Cost function to be minimized by the optimizer function
def ssim_cost_function(vp,scandata,lpstestscaled,randsample,x,y,lpsdata,lpstest):
    settings_pv = np.zeros((4,1))
    settings_fv = np.zeros((4,1))
    #Rescaling the data between -1 and 1 to make it easier on model.predict
    for i in range(len(vp)): 
        settings_pv[i] = (vp[i]-min(scandata[:,i]))
        settings_fv[i] = settings_pv[i]/(max(scandata[:,i]) - min(scandata[:,i]))
    settings_qv = np.transpose(settings_fv)
    prediction = model.predict(settings_qv)
    #Rescale the (24000x1) array into a (100x240) array 
    predict_lpstestreshaped = np.zeros((lpstest.shape[0],lpsdata.shape[1],lpsdata.shape[2]))
    for i in range(predict_lpstestreshaped.shape[0]):
        predict_lpstestreshaped[i,:] = prediction[0,:].reshape(100,lpsdata.shape[2])
    #Calculate the structural similarity - a value of 1 implies the exact same picture  
    structuralsimilarity = ssim(lpstest[randsample],predict_lpstestreshaped[randsample])
    #Try to get the cost as close to 0 as possible by maximize structural similarity
    cost = 1 - structuralsimilarity
    ###########################################################
    costex = 0
    error = 0
    for k in range(100):
        for l in range(240):
            error = ((lpstest[randsample,k,l] - predict_lpstestreshaped[randsample,k,l])**2)/24000
            costex = costex + error
    cost = cost + costex
    ###########################################################
    return cost

#Initial cost value and machine setting values
cost = 2
L1p = 50
L2p = 50
L1v = 50
L2v = 50
itera = 0

#While loop to run fmin several times until optimal parameters are found
while cost > 0.004:
    Lpv = [L1p,L2p,L1v,L2v]
    #lambda function to act as a placeholder for input values that are not being changed
    costlb = lambda pv12: ssim_cost_function(pv12,scandata,lpstest_scaled,randsample,x,y,lpsdata,lpstest)
    [L1popt,L2popt,L1vopt,L2vopt] = scipy.optimize.fmin(costlb,Lpv)
    Lpvopt = [L1popt,L2popt,L1vopt,L2vopt]
    settings_opt_pv = np.zeros((4,1))
    settings_opt_fv = np.zeros((4,1))
    #Rescaling optimal settings between -1 and 1
    for i in range(len(Lpvopt)):
        settings_opt_pv[i] = (Lpvopt[i]-min(scandata[:,i]))
        settings_opt_fv[i] = settings_opt_pv[i]/(max(scandata[:,i]) - min(scandata[:,i]))
    settings_opt_qv = np.transpose(settings_opt_fv)
    prediction = model.predict(settings_opt_qv)
    #Rescaling optimal prediction from a (24000x1) array to a (100x240)
    predict_opt_lpstestreshaped = np.zeros((lpstest.shape[0],lpsdata.shape[1],lpsdata.shape[2]))
    for i in range(predict_opt_lpstestreshaped.shape[0]):
        predict_opt_lpstestreshaped[i,:] = prediction[0,:].reshape(100,lpsdata.shape[2])
    #Minimize cost by maximizing structural similarity
    structuralsimilarity = ssim(lpstest[randsample],predict_opt_lpstestreshaped[randsample])
    cost = 1 - structuralsimilarity
    ###########################################################
    costex = 0
    error = 0
    for k in range(100):
        for l in range(240):
            error = ((lpstest[randsample,k,l] - predict_opt_lpstestreshaped[randsample,k,l])**2)/24000
            costex = costex + error
    cost = cost + costex
    ###########################################################
    #Set new values for the machine settings
    L1p = L1popt
    L2p = L2popt
    L1v = L1vopt
    L2v = L2vopt
    
    #visualize progress
    optimization_plot_lps_vs_prediction_lucretia(lpstest,predict_opt_lpstestreshaped,x,y,randsample)
    itera += 1