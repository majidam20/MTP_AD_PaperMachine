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
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils import class_weight
### End sklearn

###***Start tensorflow.keras
import tensorflow as tf
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(1234)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2, l1_l2
###**** End tensorflow.keras

#sys.path.append("..")
#import mainClass as mcFile

from mainClassFile import mainClass
import preprocessRollingLabel2_NN as aug


from pathlib import Path
pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')

pathDataforShifted5 = pathMainCodes + "/data/paperMachine/forCV/forShifted5/"
pathDataRolLbl1 = pathMainCodes + "/data/paperMachine/forCV/Shifted5_Rol_Lbl1/"


pathSavingPlotsShifted5=pathMainCodes + "/reports/forShifted5/"
#####End Import Libraries


############ Start Running codes




datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Main Start running time: {datestr},******************************:")

accPerFold = []
lossPerFold = []
dfLossEpochTrVal=pd.DataFrame()
df_ytest_yPred=pd.DataFrame()
df_ytest_yPredProb=pd.DataFrame()
dfPrReF1=pd.DataFrame()

#dfActual = pd.read_csv(pathDataforShifted5+"dfpShifted5_ForAug_201201_202734_AllTested_Correct_NT_NH.csv",header=None)
#dfActual = pd.read_csv(pathDataforShifted5+"dfpShifted5ForAug_1To5_FromAllTrainTest_201204_192153.csv",header=None)

#dfActual = pd.read_csv(pathDataRolLbl1+"dfpShifted5_Rolling5To1_210110_214459.csv",header=None)
# dfActual = pd.read_csv(pathDataRolLbl1+"dfpShifted5_Rolling_210110_191921.csv",header=None)

dfActual = pd.read_csv(pathDataforShifted5+"dfpShifted5_NoLess5_210216_182303.csv",header=None)

#dfActual=dfActual[:10000]

yX=dfActual.values
X = yX[:, 1:]  # converts the df to a numpy array
y = yX[:, 0]

# neg, pos = np.bincount(y.astype(int))
# total = neg + pos
# print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
#     total, pos, 100 * pos / total))
# init_bias = np.log([pos / neg])
print(f"\n   Shape of Actual raw dataset: {yX.shape}")
print(f"   Number of Actual labeled 0: {len(y[np.where(y==0)])}")
print(f"   Number of Actual labeled 1: {len(y[np.where(y==1)])} \n")

dataSplitTrainTestPCT=.3
dataSplitValTestPCT=.5

train_test_split_Shuffle=True
flagFitShuffle =True# True
flagSeed=True
numberOfSplits=5

skf = StratifiedKFold(n_splits=numberOfSplits,shuffle=True)#5
#skf = KFold(n_splits=5,shuffle=False)#5
model=0
mc=mainClass()

for foldNum, (trainIndex, testIndex) in enumerate(skf.split(X,y),start=1):

    #print("TRAIN:", trainIndex, "TEST:", testIndex)

    yXtrain, yXtest = yX[trainIndex], yX[testIndex]
    #ytrain, ytest = y[trainIndex], y[testIndex]

    lookback = 5  # Equivalent to 10 min of past data.

    # Temporalize the data

    yXtrain = mc.temporalize(X=yXtrain[:, 1:], y=yXtrain[:, 0], lookback=lookback)
    numberOfActualLbl1=len(yXtrain[np.where(yXtrain[:, 0] == 1)])

    AugedNN=aug.GenerateAug_NN_Rolling(yXtrain,foldNum,flagLbl2=False,jitterNum4Lbl1=7,jitterNum4Lbl2=10)###***Generate synthetic data


    datestrfoldNum = time.strftime("%y%m%d_%H%M%S")
    print(f"\n Start running time Fold_{foldNum}: {datestrfoldNum} ,--------------------------: \n")

    Actual_AugedNN=np.concatenate((yXtrain,AugedNN),axis=0)
    #yXtrain=Actual_AugedNN

    yXtrain1, yXtrain2 = train_test_split(Actual_AugedNN, shuffle=train_test_split_Shuffle,
                                                          test_size=dataSplitTrainTestPCT, random_state=42,
                                                          stratify=Actual_AugedNN[:,0])  # stratify=input_y

    yXtrain = np.concatenate((yXtrain1, yXtrain2), axis=0)

    yXvalid, yXtest = train_test_split(yXtest, shuffle=train_test_split_Shuffle,
                                          test_size=dataSplitValTestPCT, random_state=42,
                                          stratify=yXtest[:, 0])  # stratify=input_y

    print(f"\n    Shape of data to give final model in fold_{foldNum}: ")
    print(f"\n    xtrain: {np.shape(yXtrain[:,1:])}, ytrain: {np.shape(yXtrain[:,0])}")
    print(f"    xvalid: {np.shape(yXvalid[:,1:])}, yvalid: {np.shape(yXvalid[:,0])}")
    print(f"    xtest:  {np.shape(yXtest[:,1:])},  ytest:  {np.shape(yXtest[:,0])} \n")

    print(f"\n   Number of Final label 0 in yXtrain_Fold_{foldNum}: {len(yXtrain[np.where(yXtrain[:, 0] == 0)])}")

    print(f"   Number of Final label 1 in yXtrain_BeforeAddingAugedData_Fold_{foldNum}: "
          f"{len(Actual_AugedNN[np.where(Actual_AugedNN[:, 0] == 1)]) - len(AugedNN[np.where(AugedNN[:, 0] == 1)])}")

    print(f"   Number of Final label 1 in yXtrain_AfterAddingAugedData_Fold_{foldNum}: "
          f"{len(Actual_AugedNN[np.where(Actual_AugedNN[:, 0] == 1)]) - numberOfActualLbl1}")

    ytrain = yXtrain  [:,0]
    yvalid = yXvalid  [:,0]
    ytest  = yXtest   [:,0]


    xtrain = yXtrain  [:,1:]
    xvalid = yXvalid  [:,1:]
    xtest  = yXtest   [:,1:]

    epochs = 30#00#100#0#60#400#60#30#30# 150  # 0  # 100#300#60#300#10#200#00#150
    batch = 256
    lr = 0.001

    #flagR1 = True
    flagR1=False
    r1 = .1
    r2 = .1
    d1 = .2

    if foldNum==1:
        print("\n Hyperparameters of final model:")
        print(f"epochs: {epochs}, batch: {batch}, lr: {lr}, flagFitShuffle: {flagFitShuffle}, numberOfSplits:{numberOfSplits}"
              f", dataSplitTrainTestPCT: {dataSplitTrainTestPCT}, dataSplitValTestPCT: {dataSplitValTestPCT}"
              f", train_test_split_Shuffle: {train_test_split_Shuffle}, flagSeed: {flagSeed}\n ")

        pathSavingPlotsPerRunning = pathSavingPlotsShifted5 + datestr #+ "_" + modelname
        if not os.path.exists(pathSavingPlotsPerRunning):
            os.makedirs(pathSavingPlotsPerRunning)

    del model
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    model = Sequential()
    model.add(Dense(177, activation='tanh', input_dim=xtrain.shape[1]#,initial_bias=init_bias,#180
                    #,kernel_regularizer = l1(r1) if flagR1 else l2(r2),
                    ))  # , input_dim=xtrain.shape[1]
    #model.add(Dropout(d1))

    model.add(Dense(150, activation='tanh',#,initial_bias=init_bias,#150
                    #kernel_regularizer=l1(r1) if flagR1 else l2(r2),
                    # ,# kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    #bias_regularizer=l1(r2),
                    #activity_regularizer=l1(r1) if flagR1 else l2(r2),
                    #activity_regularizer=l1(r2)
                    ))
    # model.add(Dropout(d1))

    model.add(Dense(90, activation='tanh'#,initial_bias=init_bias,#90
                    #kernel_regularizer=l1(r1) if flagR1 else l2(r2),
                    # bias_regularizer=l1(r2),
                    #activity_regularizer = l1(r1) if flagR1 else l2(r2)
                    # activity_regularizer=l2(r2)
                    ))
    # model.add(Dropout(d1))

    # model.add(Dense(295, activation='tanh'#,initial_bias=init_bias,
    #                 #kernel_regularizer=l1(r1) if flagR1 else l2(r2),
    # #                 # bias_regularizer=l1(r2),
    # #                 activity_regularizer=l1(r1) if flagR1 else l2(r2)
    # #                 # activity_regularizer=l2(r2)
    #                 ))
    # model.add(Dropout(d1))

    model.add(Dense(1, activation='sigmoid'))
    # model.add(Dense(3, activation='softmax'))

    adam = optimizers.Adam(lr)#lr
    # cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",save_best_only=True,verbose=0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)

    if foldNum == 1:
        print("\n Final test model.summary(): \n")
        print(model.summary())

        print(f"\n model.get_config: {str(model.get_config())} \n")

    # fit model
    history1 = model.fit(xtrain, ytrain, batch_size=batch, epochs=epochs
                         , validation_data=(xvalid, yvalid)
                         , verbose=1, use_multiprocessing=True,
                         shuffle=flagFitShuffle).history  ### ,shuffle=True#,callbacks=[es]


    dfLossEpochTrVal=df_ytest_yPred.append(pd.concat((pd.Series(history1['loss']), pd.Series(history1['val_loss'])), axis=1),ignore_index=True)

    mc.pltLossVal(history1['loss'],history1['val_loss'],foldNum,epochs,pathSavingPlotsPerRunning,flagSeed,ylim=.5)


    # plt.figure()
    # plt.plot(history1['loss'], linewidth=2, label='Train',color="goldenrod")  # OR accuracy
    # plt.plot(history1['val_loss'], linewidth=2, label='Validation',color="brown")  # OR val_accuracy
    # plt.legend(loc='upper right', bbox_to_anchor=(1.13, 1.13))
    # plt.title(f'Model loss Fold_{foldNum}')
    # plt.ylabel('Loss')
    # plt.ylim(0,1)#.2
    # plt.xlabel('Epoch')
    # plt.savefig(pathSavingPlotsPerRunning +"/" + f"loss&valLoss_Fold_{foldNum}_Epochs{epochs}_flagSeed{flagSeed}.png", dpi=300, format='png')
    # plt.show()

    yPred = model.predict(xtest, verbose=1)
    yPredProb=model.predict(xtest, verbose=1)
    ###Definition of loss function
    l = []
    for i in yPred:
        if i < .5:
            l.append(0)
        else:
            l.append(1)

    yPred = l

    # # apply threshold to positive probabilities to create labels
    # def to_labels(pos_probs, threshold):
    # 	return (pos_probs >= threshold).astype('int')
    #
    # # define thresholds
    # thresholds = np.arange(yPred.min(), yPred.max(), 0.01)#min=0.22957686, ,max=0.22973779
    # # evaluate each threshold
    # scores = [f1_score(ytest, to_labels(yPred, t)) for t in thresholds]
    # # get best threshold
    # ix = np.argmax(scores)
    # print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
    #
    # y_pred_class = [1 if e > thresholds[ix] else 0 for e in yPred]
    # yPred=y_pred_class

    LABELS = ['Normal 0', 'Anomalous 1']
    print(f"\nclassification_report_Fold_{foldNum}:")
    print(classification_report(ytest, yPred, target_names=LABELS))

    df_ytest_yPred=df_ytest_yPred.append(pd.concat((pd.Series(ytest),pd.Series(yPred)),axis=1),ignore_index=True)
    df_ytest_yPredProb=df_ytest_yPredProb.append(pd.concat((pd.Series(ytest),pd.Series(yPredProb.reshape(-1,))),axis=1),ignore_index=True)

    mc.pltPrRe(ytest, yPredProb,foldNum,epochs,pathSavingPlotsPerRunning,flagSeed)
    # precision_rt, recall_rt, threshold_rt = precision_recall_curve(ytest, yPred)
    # pr_re_auc = auc(recall_rt, precision_rt)
    # print('precision_recall_aucAnomalous= %.3f' % (pr_re_auc),"\n")
    #
    # mc.pltPrRe(threshold_rt,precision_rt[1:],recall_rt[1:],foldNum,epochs,pathSavingPlotsPerRunning,flagSeed)
    # plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=2, color="blue")
    # plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=2, color="green")
    # plt.title(f'Precision and recall for different threshold values in Fold_{foldNum}')
    # plt.xlabel('Threshold')
    # plt.ylabel('Precision/Recall')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(pathSavingPlotsPerRunning +"/" + f"Precision&Recall_Threshold_Fold_{foldNum}_Epochs{epochs}_flagSeed{flagSeed}.png", dpi=300, format='png')
    # plt.show()

    mc.printConfMatrix(ytest,yPred, foldNum, labelsValues=[0, 1])
    # print("True Negatives: ", tn)
    # print("False Positives: ", fp)
    # print("False Negatives: ", fn)
    # print("True Positives: ", tp, "\n")

    mc.pltConfMatrix(ytest, yPred,LABELS,foldNum,epochs,pathSavingPlotsPerRunning,flagSeed,figsizeValues=(6, 6), labelsValues=[0, 1])
    # conf_matrix = confusion_matrix(ytest, yPred, labels=[0, 1])
    # #LABELS = ["Normal", "Anomalous"]
    # plt.figure(figsize=(6, 6))
    # sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d",cmap='YlGnBu');
    # plt.title(f"Confusion matrix in Fold_{foldNum}")
    # plt.ylabel('True class')
    # plt.xlabel('Predicted class')
    # plt.savefig(pathSavingPlotsPerRunning + "/" + f"ConfusionMatrix_Fold_{foldNum}_Epochs{epochs}_flagSeed{flagSeed}.png",dpi=300, format='png')
    # plt.show()

    #print(f"accuracy_score_Fold_{foldNum}:\n {accuracy_score(ytest, yPred, normalize=True)} \n")


    cr = pd.DataFrame(classification_report(ytest, yPred, target_names=LABELS, output_dict=True))
    dfPrReF1 = dfPrReF1.append(cr.iloc[:3, :2])

    datestr = time.strftime("%y%m%d_%H%M%S")
    print(f"End running time Fold_{foldNum}: {datestr} ,--------------------------. \n")


print(f"Result Average All Folds is such as below ,$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$: \n")

dfLossEpochTrVal.columns=['loss','val_loss']
df_ytest_yPred.columns=['ytest','yPred']
df_ytest_yPredProb.columns=['ytest','yPredProb']


# mc.pltLossVal(dfLossEpochTrVal['loss'].tolist(),dfLossEpochTrVal['val_loss'].tolist(),foldNum,epochs,pathSavingPlotsPerRunning,flagSeed,ylim=.5,AllFold=True)

mc.pltPrRe(df_ytest_yPredProb['ytest'].tolist(), df_ytest_yPredProb['yPredProb'].values, foldNum, epochs, pathSavingPlotsPerRunning, flagSeed,AllFold=True)

mc.printConfMatrix(df_ytest_yPred['ytest'].tolist(), df_ytest_yPred['yPred'].tolist(), foldNum, labelsValues=[0, 1],AllFold=True)

mc.pltConfMatrix(df_ytest_yPred['ytest'].tolist(), df_ytest_yPred['yPred'].tolist(),LABELS,foldNum,epochs,pathSavingPlotsPerRunning, labelsValues=[0, 1],AllFold=True)


dfPrReF1=pd.DataFrame([np.round(dfPrReF1[dfPrReF1.index=='precision'].mean(),2),np.round(dfPrReF1[dfPrReF1.index=='recall'].mean(),2),np.round(dfPrReF1[dfPrReF1.index=='f1-score'].mean(),2)],index=['precision','recall','f1-score'])

print(f"\nclassification_report_AllFolds:\n {dfPrReF1} \n")


datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Main End running time: {datestr}, ******************************.")
