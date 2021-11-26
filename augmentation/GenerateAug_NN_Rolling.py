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

from pathlib import Path
pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')
pathData = pathMainCodes + "/data/paperMachine/"
pathDataAuged = pathMainCodes + "/data/paperMachine/auged/"
pathDataAuged_jitter = pathMainCodes + "/data/paperMachine/auged/jitterFor_NN/jitters/"
pathData_NewFromWeb = pathMainCodes + "/data/paperMachine/paperMachine_NewFromWeb/"
pathData_Rolling=pathData +"shifted12345_Rolling_Auged/"

7###NN model tests

datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Start running time: {datestr}")

print("Please enter your desired shiftedNumber in order to continue code running: ")
shiftedNumber=5#int(input())
print(f"shiftedNumber: {shiftedNumber}")

pathData_Rolling_Auged=pathData_Rolling+f"auged_Shifted{shiftedNumber}_Rolling/"
pathData_AllAguedLabel2ForFinalPrediction=pathData_Rolling_Auged+"AllAguedLabel2ForFinalPrediction/"
pathData_AllAguedLabel2ForNoRandomSelected=pathData_AllAguedLabel2ForFinalPrediction+"ForNoRandomSelected/"
pathData_ForNoRandomSel_AugedBiggerMSE=pathData_AllAguedLabel2ForNoRandomSelected+"MoreGeneralAugedWithBiggerMSE/"


#dfpShifted2_Rolling5To1 = pd.read_csv(pathData_Rolling+"dfpShifted1_Rolling5To1_210110_214400.csv",header=None)#dfpShifted5_Rolling5To1_210110_214459
#dfpShifted2_Rolling5To1 = pd.read_csv(pathData_Rolling_Auged+"dfpShifted1_Rolling5To1_ForAugedNN_5Jitters_210115_230818.csv",header=None)
#dfpShifted5_Rolling5To1_ForAugedNN_5Jitters_210115_191948
#dfpShifted2_Rolling5To1 = pd.read_csv(pathData_Rolling+"dfpShifted2_Rolling5To1_210110_214153.csv",header=None)

#dfpShifted5_Rolling5To1 = pd.read_csv(pathData_Rolling_Auged+"dfpShifted5_ForLabel2_AfterRolling_Rolling5To1_Shuffled_210119_214109.csv",header=None)
# dfpShifted5_Rolling5To1 = pd.read_csv(pathData_Rolling_Auged+"dfpShifted5_ForLabel2_AfterRolling_Rolling5To1_Shuffled_5Jitters_210119_220724.csv",header=None)

# dfpShifted5_Rolling5To1 = pd.read_csv(pathData_Rolling_Auged+"dfpShifted5_ForLabel2_AfterRolling_Rolling5To1_Shuffled_5Jitters_FromAugedNN_210119_222750.csv",header=None)


######### For Solution NoLess5 For Rolling label2
# dfpShifted5_Rolling5To1 = pd.read_csv(pathData_Rolling_Auged+"dfpShifted5_ForLabel2_AfterRolling_Rolling5To1_210124_163135.csv",header=None)
# dfpShifted5_Rolling5To1 = pd.read_csv(pathData_Rolling_Auged+"dfpShifted5_Label2_Rolling5To1_5JittersForAug_210124_171054.csv",header=None)
# dfpShifted5_Rolling5To1 = pd.read_csv(pathData_Rolling_Auged+"dfpShifted5_Label2_Rolling5To1_5JittersForAug_210124_171534.csv",header=None)
# dfpShifted5_Rolling5To1 = pd.read_csv(pathData_Rolling_Auged+"dfpShifted5_Label2_Rolling5To1_5JittersForAug_210124_171903.csv",header=None)


######### For Solution NoLess5 then RandomSelected For Rolling label2
# # dfpShifted5_Rolling5To1 = pd.read_csv(pathData_AllAguedLabel2ForNoRandomSelected+"dfpShifted5_ForLabel2_NoRandomSelected_Rolling5To1_210127_183054.csv",header=None)
#
# # dfpShifted5_Rolling5To1 = pd.read_csv(pathData_AllAguedLabel2ForNoRandomSelected+"dfpShifted5_ForLabel2_NoRandomSelected_Rolling5To1_210128_210616.csv",header=None)
#
# dfpShifted5_Rolling5To1 = pd.read_csv(pathData_AllAguedLabel2ForNoRandomSelected+"dfpShifted5_Label2_NoRandomSelected_5To1_18JittersForAug_210129_004204.csv",header=None)


######### For Solution NoLess5 then RandomSelected then generate general auged with bigger MSE For Rolling label2
# dfpShifted5_Rolling5To1 = pd.read_csv(pathData_ForNoRandomSel_AugedBiggerMSE+"dfpShifted5_ForLabel2_NoRandomSelected_Rolling5To1_210128_210616.csv",header=None)

dfpShifted5_Rolling5To1 = pd.read_csv(pathData_ForNoRandomSel_AugedBiggerMSE+"dfp5_Lbl2_NoRandomSel_5To1_18JittersForAug_210129_004204.csv",header=None)




#dfpShifted2_Rolling5To1 = dfpShifted2_Rolling5To1.loc[dfpShifted2_Rolling5To1[0]==1]
dfpShifted5_Rolling5To1 = dfpShifted5_Rolling5To1.loc[dfpShifted5_Rolling5To1[0]==2]

####### Create New DS with Just label2 for making easy in jittering step
###pd.DataFrame(dfpShifted5_Rolling5To1).to_csv(pathData_Rolling_Auged+f"dfpShifted5_Labels2ForJittering_5To1_{datestr}.csv",index=None,header=None)

# dfpShifted5_Rolling5To1.to_csv(pathData_AllAguedLabel2ForNoRandomSelected+f"dfpShifted5_Labels2ForJittering_5To1_NoRandomSelected_{datestr}.csv",index=None,header=None)
#########

print(dfpShifted5_Rolling5To1.shape)
#dfpShifted5JitterAuged = dfpShifted5JitterAuged.loc[dfpShifted5JitterAuged[0]==0]

dfActual=dfpShifted5_Rolling5To1
#dfJAuged=dfpShifted5JitterAuged

rowCounts0 = len(dfActual)
#rowCounts1 = len(dfJAuged)

input_XActual = dfActual.iloc[:rowCounts0, 1:].values  # converts the df to a numpy array
input_yActual = dfActual.iloc[:rowCounts0, 0].values


#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-10000, 20000))
#input_XActual = min_max_scaler.fit_transform(input_XActual)


epochs = 1#3#10#50#100#3
batch = 32
lr = 0.001
neurons=128
flagFitShuffle=True

print("\n Hyperparameters:")
print(f"epochs: {epochs}, batch: {batch}, lr: {lr}, neurons: {neurons}, flagFitShuffle: {flagFitShuffle} \n ")

adam = optimizers.Adam(lr)

# class ExampleRandomNormal(tf.keras.initializers.Initializer):
#
#     def __init__(self, mean, stddev):
#       self.mean = mean
#       self.stddev = stddev
#
#     def __call__(self, shape, dtype=None):
#       return tf.random.normal(shape, mean=self.mean, stddev=self.stddev, dtype=dtype)
#
#     def get_config(self):  # To support serialization
#       return {'mean': self.mean, 'stddev': self.stddev}

# initializer = ExampleRandomNormal(-1, 1)# tf.keras.initializers.RandomUniform(-1, 1)
# config = initializer.get_config()
# initializer = tf.keras.initializers.from_config(config)

model = Sequential()
model.add(Dense(input_XActual.shape[1], activation='tanh',input_dim=input_XActual.shape[1]))#,input_shape=(input_XActual.shape[1],),
model.add(Dense(neurons, activation='tanh'))#,kernel_initializer=initializer
model.add(Dense(neurons, activation='tanh'))
model.add(Dense(input_XActual.shape[1], activation='tanh'))

model.compile(optimizer=adam, loss='mse')

print(model.summary())


# cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",
#                                save_best_only=True,
#                                verbose=0)

# tb = TensorBoard(log_dir='./logs',
#                 histogram_freq=0,
#                 write_graph=True,
#                 write_images=True)

NN_autoencoder_history = model.fit(input_XActual, input_XActual,
                                                epochs=epochs,
                                                batch_size=batch,
                                                #validation_data=(xValidActual, xValidActual),
                                                verbose=1, use_multiprocessing=True,shuffle=flagFitShuffle)#.history,shuffle=False
yPred = model.predict(input_XActual)

# W_Input_Hidden = model.layers[0].get_weights()[0]
# W_Output_Hidden = model.layers[1].get_weights()[0]
#
# B_Input_Hidden = model.layers[0].get_weights()[1]
# B_Output_Hidden = model.layers[1].get_weights()[1]
#
# W_Input_Hidden = model.layers[2].get_weights()[0]
# W_Output_Hidden = model.layers[3].get_weights()[0]
#
# B_Input_Hidden = model.layers[2].get_weights()[1]
# B_Output_Hidden = model.layers[3].get_weights()[1]
#
# yPred = model.predict(input_XActual)
#
# W_Input_Hidden = model.layers[0].get_weights()[0]
# W_Output_Hidden = model.layers[1].get_weights()[0]
#
# B_Input_Hidden = model.layers[0].get_weights()[1]
# B_Output_Hidden = model.layers[1].get_weights()[1]
#
# W_Input_Hidden = model.layers[2].get_weights()[0]
# W_Output_Hidden = model.layers[3].get_weights()[0]
#
# B_Input_Hidden = model.layers[2].get_weights()[1]
# B_Output_Hidden = model.layers[3].get_weights()[1]

datestr = time.strftime("%y%m%d_%H%M%S")

yPred=pd.DataFrame(yPred)
yPred.to_csv(pathData_ForNoRandomSel_AugedBiggerMSE+f"dfp5_ForLbl2_AfterRol_NoRandomSel_5To1_AugedNN_{datestr}.csv",index=None,header=None)

mse = np.mean(np.power(input_XActual - yPred, 2), axis=1)

print(f"MSE==> mean: {mse.mean()}, min: {mse.min()}, max: {mse.max()}")

datestr = time.strftime("%y%m%d_%H%M%S")
print(f"End running time: {datestr}")

