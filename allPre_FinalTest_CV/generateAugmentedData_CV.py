###*** Master Thesis Project Anomaly detection and forecasting in time series sensor data for Paper Machine

###*** Author Majid Aminian, Department of Data Science in Beuth Hochschule Berlin


### Import Libraries
import os
import sys
import gc
import random
import time
import pandas as pd
import numpy as np
pd.options.display.max_rows = None
pd.set_option('display.max_columns', 500)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=np.inf)

os.environ['PYTHONHASHSEED'] = '0'
random.seed(12345)
np.random.seed(42)


###***Start tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(1234)
###**** End tensorflow.keras


import utils.augmentation as augDTW

from pathlib import Path
pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')

import warnings
warnings.filterwarnings("ignore")




def jitterFun(x, sigma=0.01):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)



###Generate Synthetic data with DTW Approach to make a balanced dataset for preventing overfit on majority class(normals data as label zeros)
def generateAugDTW(yX,  foldNum, addAug=True , numberOfAug=14):

    ### for approach without adding Augmented data
    if addAug == False:
        return 0


    datestr = time.strftime("%y%m%d_%H%M%S")
    print(f"\n Start running time Data Augmentation With DTW Approach_Fold_{foldNum}: {datestr} ,++++++++++++++++++++++++++++++:\n")


    yXtrain = yX

    MSEAllAugedPerFold = np.empty([0, 3])

    xAuged = np.empty([0, yXtrain.shape[1] - 1])

    yXtrain = yXtrain[np.where(yXtrain[:, 0] == 1)]



    Xtrain = yXtrain[:, 1:]

    Xtrain = Xtrain.reshape((-1, Xtrain.shape[0], Xtrain.shape[1]))



    for i in range(1, numberOfAug):

        npTimeWarping_X = augDTW.time_warp(Xtrain, sigma = 0.02 + (0.0000001 + (i / 10000000) ) )[0]



        xAuged = np.append(xAuged, npTimeWarping_X, axis=0)




        ###Calculate Mean Square Error difference between Xtrain data(entered data) and synthetic data that generated with DTW approach
        mse = np.mean(np.power(Xtrain - npTimeWarping_X, 2), axis=1)

        print(f"\nData Augmentation MSE Label 1==> mean: {np.round(np.float(mse.mean()), 6)}, min: {np.round(np.float(mse.min()), 6)}, max: {np.round(np.float(mse.max()), 6)}")



        ###Calculate MSE all synthetic generated data
        MSEAllAugedPerFold = np.append(MSEAllAugedPerFold, [[mse.mean(), mse.min(), mse.max()]], axis=0)





    ytrain = np.repeat(np.array([[1]]), xAuged.shape[0], axis=0)
    yX_TimeWarping = np.concatenate((ytrain, xAuged), axis=1)



    print(f"\n \n Generated this amount: {yX_TimeWarping.shape[0]} ,Synthetic data With DTW Approach_Fold_{foldNum}\n")




    print(f"\nMSE All generated Augmented data Fold_{foldNum}:\n"
          f"mean: {np.round(np.float(MSEAllAugedPerFold[:, 0].mean(axis=0)), 6)} \n"
          f"min:  {np.round(np.float(MSEAllAugedPerFold[:, 1].mean(axis=0)), 6)} \n"
          f"max:  {np.round(np.float(MSEAllAugedPerFold[:, 2].mean(axis=0)), 6)} \n"
          )



    ###Make empty content of variables in order to prevent duplication for next CV iterations
    xAuged = np.empty([0, yX.shape[1] - 1])
    MSEAllAugedPerFold = np.empty([0, 3])



    datestr = time.strftime("%y%m%d_%H%M%S")
    print(f"\n End running time Data Augmentation With DTW Approach_Fold_{foldNum}: {datestr} ,++++++++++++++++++++++++++++++. \n")




    return  yX_TimeWarping




###Generate Synthetic data to make a balanced dataset for preventing overfit on majority class(normals data as label zeros)
def generateAugAE(yX,makeNoiseByJitter,foldNum, addAug=True,numberOfAug=14):

        ### for approach without adding Augmented data
        if addAug==False:
            return 0




        datestr = time.strftime("%y%m%d_%H%M%S")
        print(f"Start running time Data Augmentation_Fold_{foldNum}: {datestr} ,++++++++++++++++++++++++++++++:\n")


        MSEAllAugedPerFold = np.empty([0, 3])

        xPred=np.empty([0,yX.shape[1]-1])

        yXPredAuged = np.empty([0, yX.shape[1]])



        X = yX[np.where(yX[:, 0] == 1)]

        forAug = X[:,1:]




        ###Generate Synthetic data in number of user-predefined number
        for i in range(1,numberOfAug):

            if i>1 and makeNoiseByJitter==True:
                forAug = jitterFun(X[:, 1:], sigma=0.0000001 + i / 10000000)


            input_X = forAug



            ###****Hyperparameters of final Augmentation model
            epochs= 1000
            batch = 5
            lr = .001

            FitShuffle = True




            ###Just one time, Train on Actual breaks(shifted breaks that are as label one), then generate synthetic data in the number of user-predefined number by getting data from modelAE.predict(input_X) function
            if i==1:

                modelAE = Sequential()
                modelAE.add(Dense(input_X.shape[1]*2, activation='tanh',input_dim=input_X.shape[1]))
                modelAE.add(Dense(input_X.shape[1]*1.6, activation='tanh'))
                modelAE.add(Dense(input_X.shape[1]*1.3, activation='tanh'))
                modelAE.add(Dense(input_X.shape[1], activation='tanh'))



                adam = optimizers.Adam(lr)
                modelAE.compile(optimizer=adam, loss='mse')




                if foldNum == 1:
                    print("\nHyperparameters of Data Augmentation model:")
                    print(f"epochs: {epochs}, batch: {batch}, lr: {lr}, FitShuffle: {FitShuffle} \n ")
                    print("\nData Augmentation model.summary(): \n")
                    print(modelAE.summary())




                ###Fit modelAE on input_X that menas both xtrain and ytrain are one data(input_X), inspired by Autoencoder(AE) concepts
                modelAE.fit(input_X, input_X,
                            epochs=epochs,
                            batch_size=batch,
                            verbose=0, use_multiprocessing=True,
                            shuffle=FitShuffle)




            ###In order to generate next synthetic data in next iteration(s), every time put predicted(synthetic generated data) as input_X for next iteration(s)
            forAug = modelAE.predict(input_X)




            ###Calculate Mean Square Error difference between input_X data(entered data) and predicted data that generated with modelAE
            mse = np.mean(np.power(input_X - modelAE.predict(input_X), 2), axis=1)
            print(f"\nData Augmentation MSE Label 1==> mean: {np.round(np.float(mse.mean()),6)}, min: {np.round(np.float(mse.min()),6)}, max: {np.round(np.float(mse.max()),6)}")#, lr1{ir}


            ###Calculate MSE all synthetic generated data
            MSEAllAugedPerFold=np.append(MSEAllAugedPerFold,[[mse.mean(), mse.min(), mse.max()]],axis=0)




            ###Append synthetic generated data to xPred numpy variable
            xPred = np.append(xPred, modelAE.predict(input_X), axis=0)





            ###Create labels value one in the number of generated data and concatenate to xPred
            yAuged = np.repeat(np.array([[1]]), xPred.shape[0], axis=0)
            yX_Auged = np.concatenate((yAuged, xPred), axis=1)
            yXPredAuged = np.append(yXPredAuged, yX_Auged, axis=0)
            xPred = np.empty([0, yX.shape[1] - 1])





        ###Delete model after finishing all generated synthetic data because in TensorFlow, when we are using several models in a loop, some graphs and layers' name will still be kept

        del modelAE
        gc.collect()
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()




        print(f"\nMSE All generated Augmented data Fold_{foldNum}:\n"
              f"mean: {np.round(np.float(MSEAllAugedPerFold[:, 0].mean(axis=0)),6)} \n"
              f"min:  {np.round(np.float(MSEAllAugedPerFold[:, 1].mean(axis=0)),6)} \n"
              f"max:  {np.round(np.float(MSEAllAugedPerFold[:, 2].mean(axis=0)),6)} \n"
              )




        datestr = time.strftime("%y%m%d_%H%M%S")
        print(f"End running time Data Augmentation_Fold_{foldNum}: {datestr} ,++++++++++++++++++++++++++++++. \n")



        return yXPredAuged
