import os
import sys
import gc
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
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
# sys.path.append("..")

def jitterFun(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def GenerateAug_NN_Rolling(yX, foldNum, flagLbl2=False, model=0, jitterNum4Lbl1=10, jitterNum4Lbl2=20):
    datestr = time.strftime("%y%m%d_%H%M%S")
    print(f"Start running time Data Augmentation_Fold={foldNum}: {datestr} ,-------------------------- \n")

    # yX=np.concatenate((ytrain,xtrain),axis=1)#1+295==>>256

    # AugedNN=np.empty([0,yX.shape[1]])
    jitter = 0
    jitters = np.empty([0, yX.shape[1]])
    yPred = np.empty([0, yX.shape[1] - 1])

    yXPred1 = np.empty([0, yX.shape[1]])
    yXPred2 = np.empty([0, yX.shape[1]])

    forAug = 0
    li = 0
    # X2 = yX[np.where(yX[:,0]==2)]
    X1 = 0
    X2 = 0

    NumLbls = 2
    # if flagLbl2==True:
    #     NumLbls=3

    for Label in range(1, 3):  ### Iterate over label types that are 1,2
        if Label == 1:  # For Label1
            X1 = yX[np.where(yX[:, 0] == 1)]
        else:  # For Label2
            X2 = yX[np.where(yX[:, 0] == 2)]

        for iterLabel in range(1, 3):  ### Iterate over Number of desired aug for each label types that are 1,2
            if iterLabel == 1 and Label == 1:  # First Iteration For Label1
                forAug = X1
                li = 1
                continue
            elif iterLabel == 2 and Label == 1:  # Second Iteration For Label1
                for jitterIter4Lbl1 in range(1, jitterNum4Lbl1):
                    jitter = np.concatenate((X1[:, 0].reshape([X1.shape[0], 1]),
                                             jitterFun(X1[:, 1:], sigma=0.000001 + jitterIter4Lbl1 / 100000)), axis=1)
                    jitters = np.append(jitters, jitter, axis=0)
                forAug = jitters
                jitter = 0
                jitters = np.empty([0, yX.shape[1]])
                li = 0

            if iterLabel == 1 and Label == 2:  # First Iteration For Label2
                forAug = X2
                li = 1
                continue
            elif iterLabel == 2 and Label == 2:  # Second Iteration For Label2
                for jitterIter4Lbl2 in range(1, jitterNum4Lbl2):
                    jitter = np.concatenate(
                        (X2[:, 0].reshape([X2.shape[0], 1]),
                         jitterFun(X2[:, 1:], sigma=0.000001 + jitterIter4Lbl2 / 100000)), axis=1)
                    jitters = np.append(jitters, jitter, axis=0)
                forAug = jitters
                jitter = 0
                jitters = np.empty([0, yX.shape[1]])
                li = 0

            input_X = forAug[:, 1:]
            # input_yActual = dfActual.iloc[:rowCounts0, 0].values

            epochs = 1300  # 7#3#50#100  # 3#10#50#100#3
            batch = 32
            lr = 0.0001

            flagFitShuffle = True

            if NumLbls == 2:
                neurons1 = 177
                neurons2 = 150
            if NumLbls == 3:
                neurons1 = 590
                neurons2 = 450

            if Label == 1 and li == 2 and foldNum == 1:
                print("\n Data Augmentation Hyperparameters:")
                print(
                    f"epochs: {epochs}, batch: {batch}, lr: {lr}, neurons1: {neurons1}, neurons2: {neurons2}, flagFitShuffle: {flagFitShuffle} \n ")

            if li == 1:
                epochs += 700

            del model
            gc.collect()
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()

            model = Sequential()
            model.add(Dense(neurons1, activation='tanh',
                            input_dim=input_X.shape[1]))  # ,input_shape=(input_XActual.shape[1],),
            model.add(Dense(neurons2, activation='tanh'))  # ,kernel_initializer=initializer
            model.add(Dense(neurons2, activation='tanh'))
            model.add(Dense(neurons1, activation='tanh'))  # input_X.shape[1]
            model.add(Dense(input_X.shape[1], activation='tanh'))

            adam = optimizers.Adam(lr)
            model.compile(optimizer=adam, loss='mse')

            if Label == 1 and li == 2 and foldNum == 1:
                print("\n Hyperparameters:")

                print("\n Data Augmentation model.summary(): \n")
                print(model.summary())

            # print(model.summary())

            # cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",
            #                                save_best_only=True,
            #                                verbose=0)

            # tb = TensorBoard(log_dir='./logs',
            #                 histogram_freq=0,
            #                 write_graph=True,
            #                 write_images=True)

            NN_AE_history = model.fit(input_X, input_X,
                                      epochs=epochs,
                                      batch_size=batch,
                                      # validation_data=(xValidActual, xValidActual),
                                      verbose=0, use_multiprocessing=True,
                                      shuffle=flagFitShuffle)  # .history,shuffle=False

            yPred = np.append(yPred, model.predict(input_X), axis=0)

            mse = np.mean(np.power(input_X - model.predict(input_X), 2), axis=1)

            print(
                f"Data Augmentation MSE Label{Label}, iter{iterLabel}==> mean: {mse.mean()}, min: {mse.min()}, max: {mse.max()}")

            if Label == 1:
                y1 = np.repeat(np.array([[1]]), yPred.shape[0], axis=0)
                yX1 = np.concatenate((y1, yPred), axis=1)
                yXPred1 = np.append(yXPred1, yX1, axis=0)
                yPred = np.empty([0, yX.shape[1] - 1])

            if Label == 2:
                y2 = np.repeat(np.array([[2]]), yPred.shape[0], axis=0)
                yX2 = np.concatenate((y2, yPred), axis=1)
                yXPred2 = np.append(yXPred2, yX2, axis=0)
                yPred = np.empty([0, yX.shape[1] - 1])

    AugedNN12 = np.concatenate((yXPred1, yXPred2), axis=0)

    datestr = time.strftime("%y%m%d_%H%M%S")
    print(f"End running time Data Augmentation_Fold={foldNum}: {datestr} ,-------------------------- \n")

    return AugedNN12
