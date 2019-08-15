#This code is a variation of keras_cp_stitch that also includes the double bunch case for comparison

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
for t in range(1):
    [total_left,Iz_test_scaled_left,predict_Iz_test_left, idx_left,ns] = left_prediction(Iz_left,scandata,Iz)
    [total_right, Iz_test_scaled_right, predict_Iz_test_right] = right_prediction(Iz_right,scandata,idx_left,Iz,ns)

    Iz_test_scaled_stitch = np.hstack((Iz_test_scaled_left,Iz_test_scaled_right))
    predict_Iz_test_stitch = np.hstack((predict_Iz_test_left,predict_Iz_test_right))

    noise_values = [0.001,0.001,0.0001,0.0001,0.001,0.001,0.01,0.001,0.001]
    nsims=scandata.shape[0];
    X = np.empty([nsims,9])
    for i in range(X.shape[1]):
        noise = trunc_norm(0,noise_values[i],2,nsims);
        X[:,i]=scandata[:,i]+noise;  
        accuracy = 0
#plot_model_history(history)
    for i in range(9):
        [ns, tvector, curprof_stitch, predicted_curprof_stitch] = comparison_plot_2bunch_prediction_vs_lucretia(Iz_test_scaled_stitch,predict_Iz_test_stitch,np.max(Iz),ns)
    accuracy_stitch = 0;
    numer = 0
    denom = 0
    numersum = 0
    denomsum = 0
    for i in range(1555):
        for j in range(270):

            numer = (predict_Iz_test_stitch[i,j] - Iz_test_scaled_stitch[i,j])**2
            denom = (Iz_test_scaled_stitch[i,j] - Iz_test_scaled_stitch[:,j].mean())**2
            numersum = numer + numersum
            denomsum = denomsum + denom
    accuracy_stitch = 1 - (numersum/denomsum)
    total[0,t] = accuracy_stitch
    print(total)
    #if t == 0:
     #   sio.savemat(path + 'predicted_curprof_stitch1.mat',mdict = {'predicted_curprof_stitch':predicted_curprof_stitch})
      #  sio.savemat(path + 'curprof_stitch1.mat',mdict = {'curprof_stitch':curprof_stitch})
    #if t == 1:
     #   sio.savemat(path + 'predicted_curprof_stitch2.mat',mdict = {'predicted_curprof_stitch':predicted_curprof_stitch})
      #  sio.savemat(path + 'curprof_stitch2.mat',mdict = {'curprof_stitch':curprof_stitch})
    #if t == 2:
     #   sio.savemat(path + 'predicted_curprof_stitch3.mat',mdict = {'predicted_curprof_stitch':predicted_curprof_stitch})
      #  sio.savemat(path + 'curprof_stitch3.mat',mdict = {'curprof_stitch':curprof_stitch})
#######################################################################################################################################
    X = np.empty([nsims,9])
    for i in range(X.shape[1]):
        noise = trunc_norm(0,noise_values[i],2,nsims);
        X[:,i]=scandata[:,i]+noise;     

# Now choose a number of random training and test shots
    ntrain = int(np.round(nsims*0.8));
    ntest = int(np.round(nsims*0.2));
# Randomly index your shots for traning and test sets
    idxtrain = idx_left[0:ntrain];
    idxtest = idx_left[ntrain:ntrain+ntest];
# Normalize the current
    Iz_scaled = Iz/np.max(Iz) 
    Iz_train_scaled = Iz_scaled[idxtrain,:]
    Iz_test_scaled_double = Iz_scaled[idxtest,:]

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
#%%#Set up the Tensorflow environment to use only one thread.
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
    epochs      = 300;
    history = model.fit(X_train_scaled, Iz_train_scaled, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(X_test_scaled,Iz_test_scaled_double))
    print('Time to perform fit [mins] =  {0:.3f}' .format((time.time() - start_time)/60))
#%% Predict on training and validation set and plot performance
    predict_Iz_train_double = model.predict(X_train_scaled)
    predict_Iz_test_double = model.predict(X_test_scaled)
    #plot_model_history(history)
    for i in range(1):
        [ns,tvector,curprof_double,predicted_curprof_double] = comparison_plot_2bunch_prediction_vs_lucretia(Iz_test_scaled_double,predict_Iz_test_double,np.max(Iz),ns)

# Make a histogram of the score
    score = np.zeros(Iz_test_scaled_double.shape[0])
    for n in range(Iz_test_scaled_double.shape[0]):
        trueval = Iz_test_scaled_double[n,:];
        predval = predict_Iz_test_double[n,:];
        rmse = ((trueval-predval)**2).sum()
        norm = ((trueval-trueval.mean())**2).sum()
        score[n]= 1-rmse/norm;
    plt.hist(score)
    plt.show()
    print(np.mean(score))

    accuracy_double = 0;
    numer = 0
    denom = 0
    numersum = 0
    denomsum = 0
    for i in range(1555):
        for j in range(270):

            numer = (predict_Iz_test_double[i,j] - Iz_test_scaled_double[i,j])**2
            denom = (Iz_test_scaled_double[i,j] - Iz_test_scaled_double[:,j].mean())**2
            numersum = numer + numersum
            denomsum = denomsum + denom
    accuracy_double = 1 - (numersum/denomsum)
    print(accuracy)
    sio.savemat(path + 'Iz_test_scaled.mat',mdict = {'Iz_test_scaled_double':Iz_test_scaled_double})
    sio.savemat(path + 'predict_Iz_test_double.mat',mdict = {'predict_Iz_test_double':predict_Iz_test_double})
    sio.savemat(path + 'predict_Iz_test_stitch.mat',mdict = {'predict_Iz_test_stitch':predict_Iz_test_stitch})
     