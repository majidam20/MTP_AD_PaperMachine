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
import preprocessRollingLabel2_NN as aug


from pathlib import Path
pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')


pathDataRolLbl2 = pathMainCodes + "/data/paperMachine/forCV/forRolLbl2/"

pathSavingPlotsRolLbl2=pathMainCodes + "/reports/forCVRolLbl2/"
#####End Import Libraries


############ Start Running codes

def temporalize(X, y, lookback):
    output_X = []
    output_y = []

    for i in range(-2, len(X) - 5 - 1):
        t = []
        for j in range(1, 5 + 1):
            t.append(X[i + j + 1, :])

        lookback = 4### because i start from zero and 4th(0,1,2,3,4) value is first label for first block
        i+=2
        output_X.append(t)
        output_y.append(y[i + lookback ])
    y2=np.repeat(np.array(output_y),5)
    y2=y2.reshape(y2.shape[0],1)
    X2=np.array(output_X).reshape(len(output_X)*5,59)
    y2X2=np.concatenate((y2, X2), axis=1)
    return y2X2


datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Main Start running time : {datestr}")

accPerFold = []
lossPerFold = []
dfPrReF1=pd.DataFrame()

dfActual = pd.read_csv(pathDataRolLbl2+"Shifted5_NoLess5_AfterRol_Lbl2_5To1_210124_163135.csv",header=None)


#dfActual=dfActual[:5000]

yX=dfActual.values
X = yX[:, 1:]  # converts the df to a numpy array
y = yX[:, 0]

# zero,one, two = np.bincount(y.astype(int))
# total = neg + pos
# print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
#     total, pos, 100 * pos / total))

# yWeights = to_categorical(y)
# from sklearn.utils.class_weight import compute_class_weight
# y_integers = np.argmax(yWeights, axis=1)
# class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
# class_weights_yc = dict(enumerate(class_weights))



print(f"\n Number of Actual labeled 0: {len(y[np.where(y==0)])}")
print(f"Number of Actual labeled 1: {len(y[np.where(y==1)])}")
print(f"Number of Actual labeled 2: {len(y[np.where(y==2)])} \n")

dataSplitPCT=.3
dataSplitValTestPCT=.5

train_test_split_Shuffle=True
flagFitShuffle = True
flagSeed=True

p1=""

skf = StratifiedKFold(n_splits=5,shuffle=True)#5
#skf = KFold(n_splits=5,shuffle=False)#5
model=0

for foldNum, (trainIndex, testIndex) in enumerate(skf.split(X,y),start=1):

    #print("TRAIN:", trainIndex, "TEST:", testIndex)

    yXtrain, yXtest = yX[trainIndex], yX[testIndex]
    #ytrain, ytest = y[trainIndex], y[testIndex]

    lookback = 5  # Equivalent to 10 min of past data.

    # Temporalize the data
    yXtrain = temporalize(X=yXtrain[:, 1:], y=yXtrain[:, 0], lookback=lookback)

    AugedNN=aug.GenerateAug_NN_Rolling(yXtrain,foldNum,flagLbl2=False,jitterNum4Lbl1=3,jitterNum4Lbl2=10)###***Generate synthetic data


    datestrfoldNum = time.strftime("%y%m%d_%H%M%S")
    print(f"\n Start running time Fold={foldNum}: {datestrfoldNum} ,-------------------------- \n")

    Actual_AugedNN=np.concatenate((yXtrain,AugedNN),axis=0)


    yXtrain1, yXtrain2 = train_test_split(Actual_AugedNN, shuffle=train_test_split_Shuffle,
                                                          test_size=dataSplitPCT, random_state=42,
                                                          stratify=Actual_AugedNN[:,0])  # stratify=input_y

    yXtrain = np.concatenate((yXtrain1, yXtrain2), axis=0)

    yXvalid, yXtest = train_test_split(yXtest, shuffle=train_test_split_Shuffle,
                                          test_size=dataSplitValTestPCT, random_state=42,
                                          stratify=yXtest[:, 0])  # stratify=input_y

    print(f"\n Number of Final yXtrain_Fold={foldNum} labeled 0: {len(yXtrain[np.where(yXtrain[:, 0] == 0)])}")
    print(f"Number of Final yXtrain_Fold={foldNum} labeled 1: {len(yXtrain[np.where(yXtrain[:, 0] == 1)])}")
    print(f"Number of Final yXtrain_Fold={foldNum} labeled 2: {len(yXtrain[np.where(yXtrain[:, 0] == 2)])} \n")



    ytrain = to_categorical(yXtrain  [:,0] )
    yvalid = to_categorical(yXvalid  [:,0] )
    ytest  = to_categorical(yXtest   [:,0] )

    # class_weights_yc0 = class_weight.compute_class_weight('balanced', np.unique(yXtrain  [:,0]), yXtrain  [:,0])
    #
    # from sklearn.utils.class_weight import compute_class_weight
    #
    # y_integers = np.argmax(ytrain, axis=1)
    # class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    # class_weights_yc = dict(enumerate(class_weights))
    #
    # class_weights_yc=np.array(list(class_weights_yc.values()))

    xtrain = yXtrain  [:,1:]
    xvalid = yXvalid  [:,1:]
    xtest  = yXtest   [:,1:]


    neurons = xtrain.shape[1]

    epochs = 3#100#30#30# 150  # 0  # 100#300#60#300#10#200#00#150
    batch = 32
    lr = 0.001

    flagR1 = True
    #flagR1=False
    r1 = .1
    r2 = .1
    d1 = .2

    if foldNum==1:
        print("\n Hyperparameters:")
        print(f"epochs: {epochs}, batch: {batch}, lr: {lr}, neurons: {neurons}, flagFitShuffle: {flagFitShuffle} , train_test_split_Shuffle: {train_test_split_Shuffle}, flagSeed: {flagSeed}\n ")

        print(f"\n xtrain: {np.shape(xtrain)}, ytrain: {np.shape(ytrain)}")
        print(f"xvalid: {np.shape(xvalid)}, yvalid: {np.shape(yvalid)}")
        print(f"xtest: {np.shape(xtest)}, ytest: {np.shape(ytest)} \n")

        #####p1 = os.path.join(str(pathCurrent.parent.parent), "Results", "Results_001_class_oppys", "bestModels", "")
        pathSavingPlotsPerRunning = pathSavingPlotsRolLbl2 + datestr #+ "_" + modelname
        if not os.path.exists(pathSavingPlotsPerRunning):
            os.makedirs(pathSavingPlotsPerRunning)


    del model
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    model = Sequential()
    model.add(Dense(590, activation='tanh', input_dim=xtrain.shape[1]
                    #,kernel_regularizer = l1(r1) if flagR1 else l2(r2),
                    ))  # , input_dim=xtrain.shape[1]
    #model.add(Dropout(d1))

    model.add(Dense(500, activation='tanh',
                     #kernel_regularizer=l1(r1) if flagR1 else l2(r2),
                    # ,# kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    # bias_regularizer=l1(r2),
                    #activity_regularizer=l1(r1) if flagR1 else l2(r2)
                    # activity_regularizer=l1(r2)
                    ))
    # model.add(Dropout(d1))

    model.add(Dense(400, activation='tanh',
                   #kernel_regularizer=l1(r1) if flagR1 else l2(r2),
                    # bias_regularizer=l1(r2),
                    #activity_regularizer = l1(r1) if flagR1 else l2(r2)
                    # activity_regularizer=l2(r2)
                    ))
    # model.add(Dropout(d1))

    # model.add(Dense(32, activation='tanh',
    #                 kernel_regularizer=l1(r1) if flagR1 else l2(r2),
    # #                 # bias_regularizer=l1(r2),
    # #                 activity_regularizer=l1(r1) if flagR1 else l2(r2)
    # #                 # activity_regularizer=l2(r2)
    #                 ))
    # model.add(Dropout(d1))

    # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))

    adam = optimizers.Adam(lr)#lr
    # cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",save_best_only=True,verbose=0)
    # model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)

    if foldNum == 1:
        print("\n Final test model.summary(): \n")
        print(model.summary())

        print(f"\n model.get_config: {str(model.get_config())}")

    # fit model
    history1 = model.fit(xtrain, ytrain, batch_size=batch, epochs=epochs #,class_weight = class_weights_yc
                         , validation_data=(xvalid, yvalid)
                         , verbose=1, use_multiprocessing=True,
                         shuffle=flagFitShuffle).history  # ,shuffle=True#,callbacks=[es]

    plt.figure()
    plt.plot(history1['loss'], linewidth=2, label='Train')  # OR accuracy
    plt.plot(history1['val_loss'], linewidth=2, label='Validation')  # OR val_accuracy
    plt.legend(loc='upper right', bbox_to_anchor=(1.13, 1.13))
    plt.title(f'Model loss Fold={foldNum}')
    plt.ylabel('Loss')
    plt.ylim(0,.5)
    plt.xlabel('Epoch')

    #plt.savefig(pathSavingPlotsPerRunning +"/" + f"loss&val_loss_Fold={foldNum}_Epochs{epochs}_flagSeed{flagSeed}.png", dpi=300, format='png')
    #plt.show()

    yPred = model.predict(xtest, verbose=1)

    print(f"confusion_matrix_Fold={foldNum}:\n {confusion_matrix(ytest.argmax(axis=1), yPred.argmax(axis=1))} \n")

    # Creating multilabel confusion matrix
    mlbConfusion = multilabel_confusion_matrix(ytest.argmax(axis=1), yPred.argmax(axis=1))
    print(f"multilabel_confusion_matrix_Fold={foldNum}:\n {mlbConfusion} \n")

    print(f"accuracy_score_Fold={foldNum}:\n {accuracy_score(ytest.argmax(axis=1), yPred.argmax(axis=1), normalize=True)} \n")
    print(f"accuracy_score_Fold={foldNum}:\n {accuracy_score(ytest.argmax(axis=1), yPred.argmax(axis=1), normalize=False)} \n")

    print(f"classification_report_Fold={foldNum}:\n {classification_report(ytest.argmax(axis=1), yPred.argmax(axis=1))} \n")

    cr=pd.DataFrame(classification_report(ytest.argmax(axis=1), yPred.argmax(axis=1),output_dict=True))
    dfPrReF1=dfPrReF1.append(cr.iloc[:3,:3])
    # Predicting test images
    # preds = np.where(yPred < 0.5, 0, 1)

    mlbClasses = [0, 1, 2]
    # Plot confusion matrix
    plt.figure(figsize=(14, 8))
    for j, (label, matrix) in enumerate(zip(mlbClasses, mlbConfusion)):
        plt.subplot(f'23{j + 1}')
        labels = [f'Not_{label}', label]
        sns.heatmap(matrix, annot=True, square=True, fmt='d', cbar=False, cmap='Blues',
                    cbar_kws={'label': 'My Colorbar'},  # , fmt = 'd'
                    xticklabels=labels, yticklabels=labels, linecolor='black', linewidth=1)

        plt.ylabel('Actual class')
        plt.xlabel(f'Predicted class_Fold={foldNum}')
        plt.title(labels[0])

    plt.tight_layout()
    #plt.savefig(pathSavingPlotsPerRunning +"/" + f"ConfusionMatrix_Fold={foldNum}_Epochs{epochs}_flagSeed{flagSeed}.png", dpi=300, format='png')
    #plt.show()
    datestr = time.strftime("%y%m%d_%H%M%S")
    print(f"End running time Fold={foldNum}: {datestr} ,-------------------------- \n")

dfPrReF1=pd.DataFrame([np.round(dfPrReF1[dfPrReF1.index=='precision'].mean(),2),np.round(dfPrReF1[dfPrReF1.index=='recall'].mean(),2),np.round(dfPrReF1[dfPrReF1.index=='f1-score'].mean(),2)],index=['precision','recall','f1-score'])

print(f"\nclassification_report_AllFolds:\n {dfPrReF1}")

datestr = time.strftime("%y%m%d_%H%M%S")
print(f"End running time: {datestr}")
