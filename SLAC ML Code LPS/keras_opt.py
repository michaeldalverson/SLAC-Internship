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