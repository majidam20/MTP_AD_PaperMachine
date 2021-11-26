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
pathDataAuged_jitter = pathMainCodes + "/data/paperMachine/auged/jitterFor_NN/jitters/"
pathData_NewFromWeb = pathMainCodes + "/data/paperMachine/paperMachine_NewFromWeb/"
pathData_Rolling=pathData +"shifted12345_Rolling_Auged/"


###++++++++++++++++++++++++++++++++++++++++
###***Helful Web pages tutorials
###https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial
###https://www.kaggle.com/shivamb/how-autoencoders-work-intro-and-usecases
###https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
###++++++++++++++++++++++++++++++++++++++++

1###NN Autoencoder model tests
###Start Autoencoder part++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Start running time: {datestr}")

print("Please enter your desired shiftedNumber in order to continue code running: ")
shiftedNumber=5#int(input())
print(f"shiftedNumber: {shiftedNumber}")

pathData_Rolling_Auged=pathData_Rolling+f"auged_Shifted{shiftedNumber}_Rolling/"
pathData_AllAguedLabel2ForFinalPrediction=pathData_Rolling_Auged+"AllAguedLabel2ForFinalPrediction/"
pathData_AllAguedLabel2ForNoRandomSelected=pathData_AllAguedLabel2ForFinalPrediction+"ForNoRandomSelected/"
pathData_ForNoRandomSel_AugedBiggerMSE=pathData_AllAguedLabel2ForNoRandomSelected+"MoreGeneralAugedWithBiggerMSE/"


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


epochs = 1#3#10#50#100#3
batch = 32
lr = 0.001
neurons=128
flagFitShuffle=True

print("\n Hyperparameters:")
print(f"epochs: {epochs}, batch: {batch}, lr: {lr}, neurons: {neurons}, flagFitShuffle: {flagFitShuffle} \n ")

adam = optimizers.Adam(lr)

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

datestr = time.strftime("%y%m%d_%H%M%S")

yPred=pd.DataFrame(yPred)
yPred.to_csv(pathData_ForNoRandomSel_AugedBiggerMSE+f"dfp5_ForLbl2_AfterRol_NoRandomSel_5To1_AugedNN_{datestr}.csv",index=None,header=None)

mse = np.mean(np.power(input_XActual - yPred, 2), axis=1)

print(f"MSE==> mean: {mse.mean()}, min: {mse.min()}, max: {mse.max()}")


###++++++++++++++++++++++++++++++++++++++++++++++++++++
#######Start Roc auc curve part###################################

l=[]
for i in yPred:
    if i<=.5:
        l.append(0)
    else:
        l.append(1)

yPred=l

false_pos_rate, true_pos_rate, thresholds = roc_curve(ytest, yPred)
roc_auc = auc(false_pos_rate, true_pos_rate, )
print(f"roc_auc: {roc_auc}")
# print(f"thresholds: {thresholds}")

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f' % roc_auc)
plt.plot([0, 1], [0, 1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.show()



precision_rt, recall_rt, threshold_rt = precision_recall_curve(ytest, yPred)
pr_re_auc = auc(recall_rt, precision_rt)
#lr_f1 = f1_score(ytest, y_pred_class)
# summarize scores
#print(f'f1: {lr_f1} , pr_re_auc: {pr_re_auc}')
print('pr_re_auc=%.3f' % (pr_re_auc))
#
plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
#plt.show()

datestr = time.strftime("%y%m%d_%H%M%S")
print(f"End running time: {datestr}")
###End Autoencoder part++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
2### Start Example of Normal Neural Network Test(Notice this is not an Autoencoder)
7###NN model tests
datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Start running time: {datestr}")
#df = pd.read_csv(pathData + "dfpShifted5_ForAug_withAuged_1To5_201209_233059.csv",header=None)#dfpShifted5_ForAug_withAuged_1To5_201216_235923
#df = pd.read_csv(pathData + "dfpShifted5_ForAug_withAuged_1To5_201216_235923.csv",header=None)#dfpShifted5_ForAug_withAuged_1To5_201216_235923
#df=pd.read_csv(pathData + "dfpShifted5_ForAug_withAuged_OneZero_1To5_201220_011751.csv",header=None)
#df=pd.read_csv(pathData + "dfpShifted5_ForAug_withAugedJitter_OneZero_1To5_201220_013915.csv",header=None)#dfpAllScale_2RowsDel
#df=pd.read_csv(pathData + "dfpShifted5_ForAug_AllAugedLSTM_1To5_201224_023655.csv",header=None)#
#df=pd.read_csv(pathData + "dfpShifted5_ForAug_withAuged_NN_5To1_peyman_201225_021551.csv",header=None)
#df=pd.read_csv(pathData + "dfpShifted5_ForAug_withAuged_NN_1To5_peyman_201226_194414.csv",header=None)
df=pd.read_csv(pathData + "dfpShifted5ForAug_1To5_201204_192153.csv",header=None)


# df=pd.read_csv(pathData + "dfpShifted5_ForAug_201201_202734_AllTested_Correct_NT_NH.csv",header=None)# Actual data()
#df=pd.read_csv(pathData + "dfpShifted5_ForAug_AllTrainTest_201204_161806.csv",header=None)#Actual data just shuffeled()
#df=pd.read_csv(pathData + "dfpShifted5_ForAug_AllAugedJitter_NN_201228_203629.csv",header=None)
#df=pd.read_csv(pathData+"paperMachine_NewFromWeb/"+ "dfpRaw1_Scale_201229_003740.csv",header=None)#Raw data and just scaled
# def minMax(x):
#     return pd.Series(index=['min','max'],data=[x.min(),x.max()])
# df.apply(minMax)

#df  # [:200]
DATA_SPLIT_PCT=.2
rowCounts = len(df)
input_X = df.loc[:rowCounts, 1:].values  # converts the df to a numpy array
input_y = df.loc[:rowCounts, 0].values

xtrain, xtest,ytrain,ytest= train_test_split(input_X, input_y,shuffle=False, test_size=DATA_SPLIT_PCT, random_state=42)
#xtrain, xtest,ytrain,ytest= train_test_split(input_X, input_y, test_size=DATA_SPLIT_PCT, random_state=42,stratify=input_y)
#xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=DATA_SPLIT_PCT, random_state=42,stratify=ytrain)

print(f"xtrain: {np.shape(xtrain)}, ytrain: {np.shape(ytrain)}")
#print(f"xvalid: {np.shape(xvalid)}, yvalid: {np.shape(yvalid)}")
print(f"xtest: {np.shape(xtest)}, ytest: {np.shape(ytest)}")

epochs = 1000#00#150
batch = 32
lr = 0.0001
neurons=input_X.shape[1]

flagFitShuffle=False

print("Hyperparameters:")
print(f"epochs: {epochs}, batch: {batch}, lr: {lr}, neurons: {neurons}, flagFitShuffle: {flagFitShuffle} \n ")

model = Sequential()
model.add(Dense(256, activation='tanh', input_dim=input_X.shape[1]))#, input_dim=input_X.shape[1]
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

adam = optimizers.Adam(lr)
# cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",save_best_only=True,verbose=0)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# fit model
model.fit(xtrain, ytrain, batch_size=batch, epochs=epochs
          #,validation_data=(xvalid,yvalid)
          , verbose=1, use_multiprocessing=True,shuffle=flagFitShuffle)  # .history#,shuffle=True

yPred = model.predict(xtest, verbose=1)
l=[]
for i in yPred:
    if i<=.5:
        l.append(0)
    else:
        l.append(1)

yPred=l

false_pos_rate, true_pos_rate, thresholds = roc_curve(ytest, yPred)
roc_auc = auc(false_pos_rate, true_pos_rate, )
print(f"roc_auc: {roc_auc}")
# print(f"thresholds: {thresholds}")

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f' % roc_auc)
plt.plot([0, 1], [0, 1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.show()



precision_rt, recall_rt, threshold_rt = precision_recall_curve(ytest, yPred)
pr_re_auc = auc(recall_rt, precision_rt)
#lr_f1 = f1_score(ytest, y_pred_class)
# summarize scores
#print(f'f1: {lr_f1} , pr_re_auc: {pr_re_auc}')
print('pr_re_auc=%.3f' % (pr_re_auc))
#
plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
#plt.show()

scores = model.evaluate(xtest, ytest, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))



# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')

# define thresholds
# thresholds = np.arange(yPred.min(), yPred.max(), 0.00001)#min=0.22957686, ,max=0.22973779
# # evaluate each threshold
# scores = [f1_score(ytest, to_labels(yPred, t)) for t in thresholds]
# # get best threshold
# ix = np.argmax(scores)
# print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
#
# y_pred_class = [1 if e > thresholds[ix] else 0 for e in yPred]

target_names = ['Normal 0', 'Anomalous 1']
print(classification_report(ytest, yPred, target_names=target_names))

print(f"xtrain: {np.shape(xtrain)}, ytrain: {np.shape(ytrain)}")
#print(f"xvalid: {np.shape(xvalid)}, yvalid: {np.shape(yvalid)}")
print(f"xtest: {np.shape(xtest)}, ytest: {np.shape(ytest)}")

tn, fp, fn, tp = confusion_matrix(ytest, yPred,labels=[0,1]).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

conf_matrix = confusion_matrix(ytest, yPred,labels=[0,1])
LABELS = ["Normal", "Anomalous"]

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
#plt.show()

### End Example of Normal Neural Network Test(Notice this is not an Autoencoder)