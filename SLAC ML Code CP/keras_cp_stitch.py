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

path = '/Users/cemma/Documents/Work/FACET-II/Lucretia_sims/ML_Two_bunch/'
sys.path.append(path)
from keras_Iz_largescan_predictor_left import left_prediction
from keras_Iz_largescan_predictor_right import right_prediction
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
Iz_left = Iz[:,:135]
Iz_right = Iz[:,135:]
#Iz = currprofile['currprofile_weighted']; 
def trunc_norm(mu,sigma,ntrunc,nsamples):
    import scipy.stats as stats
    lower, upper = -ntrunc*sigma, ntrunc*sigma
    X = stats.truncnorm(
                (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    out = X.rvs(nsamples)
    return out
total = np.zeros((1,9))
#total_left = np.zeros((1,9))
#total_right = ((1,9))
#All of the code can be run several times
for s in range(1):
    #This is combining the left and right predictions
    #The shot number is collected with left_prediction and used as input for the right_prediction 
    [total_left,Iz_test_scaled_left,predict_Iz_test_left, idx_left,ns] = left_prediction(Iz_left,scandata,Iz)
    [total_right, Iz_test_scaled_right, predict_Iz_test_right] = right_prediction(Iz_right,scandata,idx_left,Iz,ns)

    Iz_test_scaled = np.hstack((Iz_test_scaled_left,Iz_test_scaled_right))
    predict_Iz_test = np.hstack((predict_Iz_test_left,Iz_test_scaled_right))
#Iz_test_scaled = Iz_test_scaled_left + (Iz_test_scaled_right)
#Iz_test_unscaled = np.append(Iz_test_scaled_left,Iz_test_scaled_right)
#Iz_test_scaled = np.zeros((1555,270))
#k = 0
#for i in range(1555):
#    for j in range(270):
#        
##        Iz_test_scaled[i,j] = Iz_test_unscaled[k]
#        k += 1
#predict_Iz_untest = np.append(predict_Iz_test_left,predict_Iz_test_right)
#predict_Iz_test = np.zeros((1555,270))
#k = 0
#for i in range(1555):
#    for j in range(270):
        
 #       predict_Iz_test[i,j] = predict_Iz_untest[k]
  #      k += 1
#####
          #Can add noise values
    noise_values = [0.001,0.001,0.0001,0.0001,0.001,0.001,0.01,0.001,0.001]
    nsims=scandata.shape[0];
    X = np.empty([nsims,9])
    for i in range(X.shape[1]):
        noise = trunc_norm(0,noise_values[i],2,nsims);
        X[:,i]=scandata[:,i]+noise;  
        accuracy = 0
#plot_model_history(history)
    for i in range(1):
        preset_plot_2bunch_prediction_vs_lucretia(Iz_test_scaled,predict_Iz_test,np.max(Iz),ns[i])
        #Finding the R^2 associated with the stitch case
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
    accuracy = 1 - (numersum/denomsum)
    total[0,s] = accuracy
    print(total)