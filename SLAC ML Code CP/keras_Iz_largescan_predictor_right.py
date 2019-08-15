def right_prediction(Iz,scandata,idx,Iz4max,ns):

    import numpy as np
    import keras
    from keras import optimizers
    from keras.models import Sequential
    from keras.layers import Dense
    import scipy.io as sio
    import matplotlib.pyplot as plt
    import sys, time
    from sklearn import preprocessing
    from plotting_functions_rightbunch import plot_model_history, plot_2bunch_prediction_vs_lucretia, preset_plot_right_prediction_vs_lucretia
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

    noise_values = [0.001,0.001,0.0001,0.0001,0.001,0.001,0.01,0.001,0.001]

    X = np.empty([nsims,9])
    for i in range(X.shape[1]):
        noise = trunc_norm(0,noise_values[i],2,nsims);
        X[:,i]=scandata[:,i]+noise;     

# Now choose a number of random training and test shots
    ntrain = int(np.round(nsims*0.8));
    ntest = int(np.round(nsims*0.2));
# Randomly index your shots for traning and test sets
    idxtrain = idx[0:ntrain];
    idxtest = idx[ntrain:ntrain+ntest];
# Normalize the current
    Iz_scaled = Iz/np.max(Iz4max) 
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
    epochs      = 300;
    history = model.fit(X_train_scaled, Iz_train_scaled, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(X_test_scaled,Iz_test_scaled))
    print('Time to perform fit [mins] =  {0:.3f}' .format((time.time() - start_time)/60))
#%% Predict on training and validation set and plot performance
    predict_Iz_train = model.predict(X_train_scaled)
    predict_Iz_test = model.predict(X_test_scaled)
    plot_model_history(history)
    for i in range(1):
        preset_plot_right_prediction_vs_lucretia(Iz_test_scaled,predict_Iz_test,np.max(Iz4max),ns[i])

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
    numer = 0
    denom = 0
    numersum = 0
    denomsum = 0
    for i in range(1555):
        for j in range(135):

            numer = (predict_Iz_test[i,j] - Iz_test_scaled[i,j])**2
            denom = (Iz_test_scaled[i,j] - Iz_test_scaled[:,j].mean())**2
            numersum = numer + numersum
            denomsum = denomsum + denom
    accuracy = 1 - (numersum/denomsum)
    print(accuracy)
#print(k)
#noise_values[3] = noise_values[3] + increase_values[3]
    total.append(accuracy)
    return total, Iz_test_scaled, predict_Iz_test