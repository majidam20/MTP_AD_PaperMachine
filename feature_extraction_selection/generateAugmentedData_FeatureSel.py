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

# os.environ['PYTHONHASHSEED'] = '0'
# random.seed(12345)
# np.random.seed(42)

###***Start tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam

#tf.random.set_seed(1234)
###**** End tensorflow.keras

from pathlib import Path
pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')

def jitterFun(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def generateAugAE(yX,makeNoiseByJitter,foldNum=0,model=0,jitterNum4Lbl2=20):

        datestr = time.strftime("%y%m%d_%H%M%S")
        print(f"Start running time Data Augmentation_Fold_{foldNum}: {datestr} ,++++++++++++++++++++++++++++++:\n")


        jitters=np.empty([0,yX.shape[1]])
        yPred=np.empty([0,yX.shape[1]-1])

        yXPred2 = np.empty([0, yX.shape[1]])

        li=0

        X2 = yX[np.where(yX[:, 0] == 1)]
        forAug = X2[:,1:]

        lr1 = np.arange(.001, .01, .001).tolist()
        ep = np.arange(500, 1000, 50).tolist()

        #for ir,jp in zip(lr1, ep):
        for i in range(jitterNum4Lbl2):


            if li==1 and makeNoiseByJitter==True:
                forAug = jitterFun(X2[:, 1:], sigma=0.0000001 + i / 1000000)

            input_X = forAug

            ###****Hyperparameters of final Augmentation model
            epochs = 10#00
            batch = 16
            lr = .0001

            neurons1 = 590
            neurons2 = 500

            flagFitShuffle = True

            if i==0:
                li+= 1
                model = Sequential()
                model.add(Dense(neurons1, activation='tanh',input_dim=input_X.shape[1]))
                model.add(Dense(neurons2, activation='tanh'))
                model.add(Dense(400, activation='tanh'))
                #model.add(Dense(neurons2, activation='tanh'))#AE
                model.add(Dense(input_X.shape[1], activation='tanh'))

                adam = optimizers.Adam(lr)
                model.compile(optimizer=adam, loss='mse')

                print("\nHyperparameters of Data Augmentation model:")
                print(f"epochs: {epochs}, batch: {batch}, lr: {lr}, flagFitShuffle: {flagFitShuffle} \n ")
                print("\nData Augmentation model.summary(): \n")

                print(model.summary())
                model.fit(input_X, input_X,
                            epochs=epochs,
                            batch_size=batch,
                            verbose=0, use_multiprocessing=True,
                            shuffle=flagFitShuffle)


            yPred=np.append(yPred , model.predict(input_X),axis=0)
            forAug=yPred

            mse = np.mean(np.power(input_X - model.predict(input_X), 2), axis=1)


            print(f"\nData Augmentation MSE Label 1==> mean: {mse.mean()}, min: {mse.min()}, max: {mse.max()} \n")#, lr1{ir}

            y2 = np.repeat(np.array([[1]]), yPred.shape[0], axis=0)
            yX2 = np.concatenate((y2, yPred), axis=1)
            yXPred2 = np.append(yXPred2, yX2, axis=0)
            yPred = np.empty([0, yX.shape[1] - 1])


        datestr = time.strftime("%y%m%d_%H%M%S")
        print(f"End running time Data Augmentation_Fold_{foldNum}: {datestr} ,++++++++++++++++++++++++++++++. \n")

        return yXPred2
