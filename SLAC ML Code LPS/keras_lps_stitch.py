# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:16:09 2019

@author: MIchael Alverson
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

randsample = np.random.randint(0,1555)
scan_date = '3_29';
path = "/Users/MIchael Alverson/Desktop/SLAC ML Code/LPS Code/"
scandata = sio.loadmat(path+'scandata2bunch_filtered.mat',squeeze_me=True); 
lps = sio.loadmat(path+'twobunch_2019_'+scan_date+'_LPS_ROI.mat',squeeze_me=True); 
sys.path.append(path)
from plotting_functions_twobunch import *
from plotting_lps_fcns import *
from sklearn import preprocessing
from keras_lps_leftstitch import keras_lps_leftstitchfun
from keras_lps_rightstitch import keras_lps_rightstitchfun
from plotting_lps_fcns import *
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
lpsleft = lpsdata[:,:,:120]
lpsright = lpsdata[:,:,120:]

dt = 1; # Time resolution From Alberto's TCAV axis conversion [um/pixel]
dE = 1;# Energy resolution [MeV/pix]
x = (np.arange(lpsleft.shape[1])-np.round(lpsleft.shape[1]/2))*dt;
y = (np.arange(lpsleft.shape[2])-np.round(lpsleft.shape[2]/2))*dE

predict_lpsleft,lpsleft_true,ns = keras_lps_leftstitchfun(lpsleft,scandata)
predict_lpsright,lpsright_true = keras_lps_rightstitchfun(lpsright,scandata,ns)

lps_true = np.dstack((lpsleft_true,lpsright_true))
#predict_lps = np.vstack((predict_lpsleft,predict_lpsright))
##################################
predict_lpstestreshapedleft = np.zeros((lpsleft_true.shape[0],lpsdata.shape[1],int(lpsdata.shape[2]/2)))
for i in range(predict_lpstestreshapedleft.shape[0]):
    predict_lpstestreshapedleft[i,:] = predict_lpsleft[i,:].reshape(lpsdata.shape[1],int(lpsdata.shape[2]/2))
predict_lpstestreshapedright = np.zeros((lpsright_true.shape[0],lpsdata.shape[1],int(lpsdata.shape[2]/2)))
for i in range(predict_lpstestreshapedright.shape[0]):
    predict_lpstestreshapedright[i,:] = predict_lpsright[i,:].reshape(lpsdata.shape[1],int(lpsdata.shape[2]/2))
##################################
predict_lpstestreshaped = np.dstack((predict_lpstestreshapedleft,predict_lpstestreshapedright))
plot_lps_vs_prediction_lucretia(lps_true,predict_lpstestreshaped,x,y)

numer = 0
denom = 0
numersum = 0
denomsum = 0
randsample = np.random.randint(0,1555)
for j in range(100):
        for k in range(240):

            numer = (predict_lpstestreshaped[randsample,j,k] - lps_true[randsample,j,k])**2
            denom = (lps_true[randsample,j,k] - lps_true[:,j,k].mean())**2
            numersum = numer + numersum
            denomsum = denomsum + denom
accuracy_lpsstitch = 1 - (numersum/denomsum)