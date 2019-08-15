#This code is just so the ranges of the diagnostics could be visualized
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

L1p = scandata[:,0]
plt.hist(L1p)
plt.figure()
L2p = scandata[:,1]
plt.hist(L2p)
plt.figure()
L1v = scandata[:,2]
plt.hist(L1v)
plt.figure()
L2v = scandata[:,3]
plt.hist(L2v)
plt.figure()
Qi = scandata[:,4]
plt.hist(Qi)
plt.figure()
bc11pkI = scandata[:,5]
plt.hist(bc11pkI)
plt.figure()
bc14pkI = scandata[:,6]
plt.hist(bc14pkI)
plt.figure()
IppkI = scandata[:,7]
plt.hist(IppkI)
plt.figure()
bc11_centroidx = scandata[:,8]
plt.hist(bc11_centroidx)
plt.figure()
#bc14_centroidx = scandata[:,9]
#plt.hist(bc14_centroidx)
#plt.figure()

print(min(L1p))
print(max(L1p))
print(min(L2p))
print(max(L2p))
print(min(L1v))
print(max(L1v))
print(min(L2v))
print(max(L2v))
print(min(Qi))
print(max(Qi))
print(min(bc11pkI))
print(max(bc11pkI))
print(min(bc14pkI))
print(max(bc14pkI))
print(min(IppkI))
print(max(IppkI))
print(min(bc11_centroidx))
print(max(bc11_centroidx))
