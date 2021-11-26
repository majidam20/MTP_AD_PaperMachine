import os
import sys
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
np.random.seed(42)
random.seed(12345)
###Start sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import recall_score, auc, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split,TimeSeriesSplit,cross_val_score,KFold,cross_validate,GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import preprocessing
### End sklearn

###***Start tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(1234)
###**** End tensorflow.keras

from pathlib import Path
pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')
pathData = pathMainCodes + "/data/paperMachine/"
pathDataAuged = pathMainCodes + "/data/paperMachine/auged/"


7#LSTM model tests
#LSTM model tests

datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Start running time: {datestr}")

#dfpShifted5ForAug = pd.read_csv(pathData+"dfpShifted5ForAug_1To5_201204_192153.csv",header=None)
dfpShifted5ForAug = pd.read_csv(pathData+"dfpShifted5_ForAug_AllTrainTest_201204_161806.csv",header=None)#shuffled
#dfpShifted5JitterAuged = pd.read_csv(pathData+"dfpShifted5JitterAuged_1To5_201204_192153.csv",header=None)#.8
# dfpShifted5JitterAuged = pd.read_csv(pathData+"dfpShifted5_ForAug_withAugedFawaz_OneZero_1To5_201221_002448.csv",header=None)# Fawaz
#dfpShifted5JitterAuged = pd.read_csv(pathData+"dfpShifted5_ForAug_withAugedJitter_OneZero_1To5_201220_013915.csv",header=None)#Jitter
#dfpShifted5JitterAuged = pd.read_csv(pathData+"dfpShifted5_ForAug_withAugedFawaz4_OneZero_1To5_201221_011005.csv",header=None)# Fawaz*4
#dfpShifted5JitterAuged = pd.read_csv(pathData+"dfpShifted5_AugedMagnitude_OneZero_1To5_201222_154819.csv",header=None)#Magnitude
#dfpShifted5JitterAuged = pd.read_csv(pathData+"dfpShifted5_AugedFawazAll_OneZero_1To5_201223_172909.csv",header=None)# FawazAll


dfpShifted5ForAug = dfpShifted5ForAug.loc[dfpShifted5ForAug[0]==1]
#dfpShifted5JitterAuged = dfpShifted5JitterAuged.loc[dfpShifted5JitterAuged[0]==1]

dfActual=dfpShifted5ForAug
#dfJAuged=dfpShifted5JitterAuged

rowCounts0 = len(dfActual)
#rowCounts1 = len(dfJAuged)

input_XActual = dfActual.iloc[:rowCounts0, 1:].values  # converts the df to a numpy array
input_yActual = dfActual.iloc[:rowCounts0, 0].values

# input_XJAuged = dfJAuged.iloc[:rowCounts1, 1:].values  # converts the df to a numpy array
# input_yJAuged = dfJAuged.iloc[:rowCounts1, 0].values

# print('First instance of y = 1 in the original data')
# print(df.iloc[(np.where(np.array(input_y) == 1)[0][0]-5):(np.where(np.array(input_y) == 1)[0][0]+1), ])

###5 To 1
input_X=np.reshape(input_XActual,(int(input_XActual.shape[0]),1,input_XActual.shape[1]))
input_y=np.reshape(input_yActual,(int(input_yActual.shape[0]),1,1))
input_y=input_y.reshape(input_y.shape[0],input_y.shape[1])[:,0]


print(f'shape input_XActual: {input_XActual.shape}')
print(f'shape input_yActual: {input_yActual.shape}')


print(f'shape input_X: {input_X.shape}')
print(f'shape input_y: {input_y.shape}')


#ytest=yJAuged.reshpe(yJAuged.shape[0],-1)

# print('For the same instance of y = 1, we are keeping past 5 samples in the 3D predictor array, X.')
# print(pd.DataFrame(np.concatenate(X[np.where(np.array(y) == 1)[0][0]], axis=0 )))

timesteps = input_X.shape[1]  # equal to the lookback
n_features = input_X.shape[2]  # 59 number of features

epochs = 1000
batch = 32
lr = 0.001
neurons=128
flagFitShuffle=False

print("Hyperparameters:")
print(f"epochs: {epochs}, batch: {batch}, lr: {lr}, neurons: {neurons}, flagFitShuffle: {flagFitShuffle} \n ")

lstm_autoencoder = Sequential()
# Encoder
lstm_autoencoder.add(LSTM(n_features, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
lstm_autoencoder.add(LSTM(neurons, activation='relu', return_sequences=False))
lstm_autoencoder.add(RepeatVector(timesteps))
# Decoder
lstm_autoencoder.add(LSTM(neurons, activation='relu', return_sequences=True))
lstm_autoencoder.add(LSTM(n_features, activation='relu', return_sequences=True))
lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

adam = optimizers.Adam(lr)
lstm_autoencoder.compile(loss='mse', optimizer=adam)
print(lstm_autoencoder.summary())
# cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",
#                                save_best_only=True,
#                                verbose=0)

# tb = TensorBoard(log_dir='./logs',
#                 histogram_freq=0,
#                 write_graph=True,
#                 write_images=True)

lstm_autoencoder_history = lstm_autoencoder.fit(input_X, input_X,
                                                epochs=epochs,
                                                batch_size=batch,
                                                #validation_data=(xValidActual, xValidActual),
                                                verbose=1, use_multiprocessing=True,shuffle=flagFitShuffle)#.history,shuffle=False

yPred = lstm_autoencoder.predict(input_X)

yPred=np.reshape(yPred,(int(yPred.shape[0]),yPred.shape[2]))
input_X=np.reshape(input_X,(int(input_X.shape[0]),input_X.shape[2]))

mse = np.mean(np.power(input_X - yPred, 2), axis=1)

datestr = time.strftime("%y%m%d_%H%M%S")
pd.DataFrame(yPred).to_csv(pathDataAuged+'jitterFor_NN/'+f'dfpShifted5_Auged_LSTM_Ones_5To1_peyman_{datestr}.csv',index=None,header=None)
#pd.DataFrame(yPred).to_csv(pathDataAuged+'jitterFor_NN/'+f'dfpShifted5_Auged_NN_Ones_5To1_jitter_{datestr}.csv',index=None,header=None)

print(f"MSE==> mean: {mse.mean()}, min: {mse.min()}, max: {mse.max()}")

datestr = time.strftime("%y%m%d_%H%M%S")
print(f"End running time: {datestr}")
###*** Temp Codes
