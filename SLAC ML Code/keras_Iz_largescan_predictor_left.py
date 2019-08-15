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
#Iz = currprofile['currprofile_weighted']; 
Iz = Iz[:,:135]
scandata = scandata[:,:135]
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
    #kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkjkj
    #kjadsflkja

nsims=scandata.shape[0];
total = []
#noise_values = [1000,1000,1000,1000,1000,1000,1000,1000,1000]
#noise_values = [0.005, 0.005, 0.00005, 0.00005, 0.0012459472, 0.133385908, 0.3305109657, 0.00001216562905, 0.00008658611114]
#increase_values = [.11*2,.11*2,.0011*2,.0011*2,0.0274108384*2,2.934489976*2,0.0002676438391*2,0.001904894*2]

    #noise_values = [0.001,0.001,0.0001,0.0001,0.001,0.001,0.01,0.001,0.001]

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
epochs      = 1;
history = model.fit(X_train_scaled, Iz_train_scaled, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(X_test_scaled,Iz_test_scaled))
print('Time to perform fit [mins] =  {0:.3f}' .format((time.time() - start_time)/60))
#%% Predict on training and validation set and plot performance
predict_Iz_train = model.predict(X_train_scaled)
predict_Iz_test = model.predict(X_test_scaled)
plot_model_history(history)
for i in range(1):
    plot_2bunch_prediction_vs_lucretia(Iz_test_scaled,predict_Iz_test,np.max(Iz))

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

accuracy = 0;
ns = int(Iz_test_scaled[:,0].shape+Iz_test_scaled[:,0].shape*(0.5*(2*np.random.rand(1,1))-1))
for j in range(1550):
    for i in range(135):
    
        temp = ((predict_Iz_test[j,i] - Iz_test_scaled[j,i])**2)/(270*1550)
        accuracy = accuracy + temp
print(accuracy)
#print(k)
#noise_values[3] = noise_values[3] + increase_values[3]
total.append(accuracy)
print(noise_values)
#%% Save and/or load stuff if you want
    # serialize model to JSON
#model_json = model.to_json()
#with open(path+"ML_predictions/keras_models/keras_Iz_model.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#model.save_weights(path+"ML_predictions/keras_models/keras_Iz_model.h5")
#print("Saved model to disk") 
#%% Load the model from file once training is done if you want to make new predictions    
## load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
# 
## evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
 #%% This is for the inverse model
#for i in range(4):
#    plt.scatter(predict_linac_settings_train[:,i],X_train_scaled[:,i]);plt.show()