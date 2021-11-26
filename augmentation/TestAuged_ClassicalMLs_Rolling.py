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
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.utils import to_categorical
### End sklearn

from pathlib import Path
pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')
pathData = pathMainCodes + "/data/paperMachine/"
pathDataAuged = pathMainCodes + "/data/paperMachine/auged/"
pathDataAuged_jitter_NN = pathMainCodes + "/data/paperMachine/auged/jitterFor_NN/"
pathData_Rolling=pathData +"shifted12345_Rolling_Auged/"
pathData_Rolling=pathData +"shifted12345_Rolling_Auged/"

print("Please enter your desired shiftedNumber in order to continue code running: ")
shiftedNumber=5#int(input())
print(f"shiftedNumber: {shiftedNumber}")

pathData_Rolling_Auged=pathData_Rolling+f"auged_Shifted{shiftedNumber}_Rolling/"


7#LSTM model tests

datestr = time.strftime("%y%m%d_%H%M%S")
print(f"\n Start running time: {datestr} \n")

#df = pd.read_csv(pathData + "dfpShifted5_ForAug_withAuged_1To5_201209_233059.csv",header=None)#dfpShifted5_ForAug_withAuged_1To5_201216_235923
#df = pd.read_csv(pathData + "dfpShifted5_ForAug_withAuged_1To5_201216_235923.csv",header=None)#dfpShifted5_ForAug_withAuged_1To5_201216_235923
#df=pd.read_csv(pathData + "dfpShifted5_ForAug_withAuged_OneZero_1To5_201220_011751.csv",header=None)
#df=pd.read_csv(pathData + "dfpShifted5_ForAug_withAugedJitter_OneZero_1To5_201220_013915.csv",header=None)#dfpAllScale_2RowsDel
#df=pd.read_csv(pathData + "dfpShifted5_ForAug_AllAugedLSTM_1To5_201224_023655.csv",header=None)#
#df=pd.read_csv(pathData + "dfpShifted5ForAug_1To5_201204_192153.csv",header=None)
#df=pd.read_csv(pathData + "dfpShifted5_ForAug_withAuged_NN_5To1_peyman_201225_021551.csv",header=None)
#df=pd.read_csv(pathData + "dfpShifted5_ForAug_withAuged_NN_1To5_peyman_201226_194414.csv",header=None)
#df  # [:200]

#dfActual=pd.read_csv(pathData_Rolling + "dfpShifted5_Rolling_210110_191921.csv",header=None)#dfpShifted5_Rolling5To1_210110_214459

#dfActual=pd.read_csv(pathData_Rolling_Auged + "dfpShifted5_Rolling5To1_210110_214459.csv",header=None)#dfpShifted5_Rolling5To1_210110_214459
#df1=pd.read_csv(pathData_Rolling_Auged + "dfpShifted5_Rolling5To1_AugedNN_210115_183020.csv",header=None)#dfpShifted5_Rolling5To1_AugedNN_210115_183020
#df2=pd.read_csv(pathData_Rolling_Auged + "dfpShifted5_Rolling5To1_AugedNN_210115_192544.csv",header=None)#dfpShifted5_Rolling5To1_AugedNN_210115_192544
#dfAll=pd.concat([dfActual,df1,df2],axis=0)

dfActual=pd.read_csv(pathData_Rolling_Auged + "dfpShifted5_ForLabel2_AfterRolling_Rolling5To1_Shuffled_210119_214109.csv",header=None)
df1=pd.read_csv(pathData_Rolling_Auged + "dfpShifted5_ForLabel2_AfterRolling_Rolling5To1_Shuffled_AugedNN_210119_214744.csv",header=None)
df2=pd.read_csv(pathData_Rolling_Auged + "dfpShifted5_ForLabel2_AfterRolling_Rolling5To1_Shuffled_AugedNNFrom5Jitters_210119_221642.csv",header=None)
df3=pd.read_csv(pathData_Rolling_Auged + "dfpShifted5_ForLabel2_AfterRolling_Rolling5To1_Shuffled_AugedNNFrom5Jitters_210119_224034.csv",header=None)


#dftest=dfActual
dfAll=pd.concat([dfActual,df1,df2,df3],axis=0)
#dfAll=dfActual
#dfAll.to_csv(pathData+f'dfAll_allAuged_Rolling{datestr}.csv',index=None,header=None)


print(f"Number of labeled 0: {len(dfAll.loc[dfAll[0]==0])}")
print(f"Number of labeled 1: {len(dfAll.loc[dfAll[0]==1])}")
print(f"Number of labeled 2: {len(dfAll.loc[dfAll[0]==2])}")

#dfAll=dfActual
rowCounts = len(dfAll)
#rowCounts = 25
input_X = dfAll.iloc[:rowCounts, 1:].values  # converts the df to a numpy array
input_y = dfAll.iloc[:rowCounts, 0].values

# print('First instance of y = 1 in the original data')
# print(df.iloc[(np.where(np.array(input_y) == 1)[0][0]-5):(np.where(np.array(input_y) == 1)[0][0]+1), ])

# # print('For the same instance of y = 1, we are keeping past 5 samples in the 3D predictor array, X.')
# # print(pd.DataFrame(np.concatenate(X[np.where(np.array(y) == 1)[0][0]], axis=0 )))

### 1 To 5
# input_X=np.reshape(input_X,(int(input_X.shape[0]/5),5,input_X.shape[1]))
# input_y=np.reshape(input_y,(int(input_y.shape[0]/5),5,1))
# input_y=input_y.reshape(input_y.shape[0],input_y.shape[1])[:,0]


###5 To 1
# input_X=np.reshape(input_X,(int(input_X.shape[0]),1,input_X.shape[1]))
# input_y=np.reshape(input_y,(int(input_y.shape[0]),1,1))
# input_y=input_y.reshape(input_y.shape[0],input_y.shape[1])[:,0]

DATA_SPLIT_PCT=.2
train_test_split_Shuffle=True
###*** shuffle=False exactly equal to hard index split
xtrain, xtest,ytrain,ytest= train_test_split(input_X, input_y,shuffle=True, test_size=DATA_SPLIT_PCT, random_state=42)

#xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=DATA_SPLIT_PCT, random_state=42)

###*** Default is shuffle=True, so when df is 1To5 must not to do due to Timeseries concept
###*** Default is shuffle=True, so when df is 5To1 is better to use with stratify=input_y
# xtrain, xtest,ytrain,ytest= train_test_split(input_X, input_y, test_size=DATA_SPLIT_PCT, random_state=42)#,stratify=input_y)

print(f'shape input_X: {input_X.shape}')
print(f'shape input_y: {input_y.shape}')

# one hot encode
ytrain=to_categorical(ytrain)
#yvalid=to_categorical(yvalid)
ytest=to_categorical(ytest)
# ytrain=ytrain.reshape((ytrain.shape[0],1))
# ytest=ytest.reshape((ytest.shape[0],1))
#
# ytrain=OneHotEncoder().fit_transform(ytrain.astype(str))
# ytest=OneHotEncoder().fit_transform(ytest.astype(str))

print(f"xtrain: {np.shape(xtrain)}, ytrain: {np.shape(ytrain)}")
#print(f"xvalid: {np.shape(xvalid)}, yvalid: {np.shape(yvalid)}")
print(f"xtest: {np.shape(xtest)}, ytest: {np.shape(ytest)}")

###input_X.shape[0]/5
n_features = input_X.shape[1]  # 59 or 295 number of features

flagFitShuffle=True

print("\n Hyperparameters:")
print(f"flagFitShuffle: {flagFitShuffle} , train_test_split_Shuffle: {train_test_split_Shuffle}\n ")

# fit model
rfc = RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state = 42)
rfc.fit(xtrain, ytrain)

yPred = rfc.predict(xtest)
#yPred = rfc.predict_proba(xtest)


print(f"confusion_matrix:\n {confusion_matrix(ytest.argmax(axis=1), yPred.argmax(axis=1))} \n")

# Creating multilabel confusion matrix
mlbConfusion = multilabel_confusion_matrix(ytest.argmax(axis=1), yPred.argmax(axis=1))
print(f"multilabel_confusion_matrix:\n {mlbConfusion} \n")

print(f"accuracy_score:\n {accuracy_score(ytest.argmax(axis=1), yPred.argmax(axis=1), normalize=False)} \n")

print(f"classification_report:\n {classification_report(ytest.argmax(axis=1), yPred.argmax(axis=1))} \n")

# Predicting test images
#preds = np.where(yPred < 0.5, 0, 1)

mlbClasses=[0,1,2]
# Plot confusion matrix
fig = plt.figure(figsize = (14, 8))
for i, (label, matrix) in enumerate(zip(mlbClasses, mlbConfusion)):
    plt.subplot(f'23{i+1}')
    labels = [f'Not_{label}', label]
    sns.heatmap(matrix, annot = True, square = True, fmt = 'd', cbar = False, cmap = 'Blues',cbar_kws = {'label': 'My Colorbar'},
                xticklabels = labels, yticklabels = labels, linecolor = 'black', linewidth = 1)

    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    plt.title(labels[0])

plt.tight_layout()
plt.show()


# l=[]
# for i in yPred:
#     if i<=.5:
#         l.append(0)
#     else:
#         l.append(1)
#
# yPred=l
#
# y_pred_class=yPred
# target_names = ['Normal 0', 'Anomalous 1']
# print(classification_report(ytest, y_pred_class, target_names=target_names))
#
# print(f"xtrain: {np.shape(xtrain)}, ytrain: {np.shape(ytrain)}")
# #print(f"xvalid: {np.shape(xvalid)}, yvalid: {np.shape(yvalid)}")
# print(f"xtest: {np.shape(xtest)}, ytest: {np.shape(ytest)}")
#
# tn, fp, fn, tp = confusion_matrix(ytest, y_pred_class,labels=[0,1]).ravel()
# print("True Negatives: ",tn)
# print("False Positives: ",fp)
# print("False Negatives: ",fn)
# print("True Positives: ",tp)

datestr = time.strftime("%y%m%d_%H%M%S")
print(f"End running time: {datestr}")