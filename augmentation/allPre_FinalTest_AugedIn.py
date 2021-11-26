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

# os.environ['PYTHONHASHSEED'] = '0'
# np.random.seed(42)
# random.seed(12345)

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
#tf.random.set_seed(1234)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2, l1_l2
###**** End tensorflow.keras

#sys.path.append("..")
from augmentation.mainClassFile import mainClass


from pathlib import Path
pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')

pathDataforShifted5 = pathMainCodes + "/data/paperMachine/forCV/forShifted5/"
pathDataRolLbl1 = pathMainCodes + "/data/paperMachine/forCV/Shifted5_Rol_Lbl1/"

pathSavingPlotsShifted5=pathMainCodes + "/reports/forShifted5/"

shiftedNumber=5

1###Raw dataset just with One row as break, scale and drop some columns
# dfpRaw1 = pd.read_csv(pathData+"paperMachine_NewFromWeb/"+"processminer_single1.csv",header=None)
# dfpRaw1 = dfpRaw1.drop([0,29,62], axis=1)
# dfpRaw1.columns=[i for i in range(dfpRaw1.shape[1])]
# dfpRaw1=dfpRaw1.drop([0],axis=0)
# dfpRaw1=dfpRaw1.reset_index(drop=True)#inplace=True,
# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
# x_scaled = min_max_scaler.fit_transform(dfpRaw1.iloc[:,1:].values)
# dfpRaw1_Scale = pd.DataFrame(x_scaled)
# dfpRaw1_Scale=pd.concat([dfpRaw1.iloc[:,0],dfpRaw1_Scale],axis=1)
# datestr = time.strftime("%y%m%d_%H%M%S")
# dfpRaw1_Scale.to_csv(pathData+f'dfpRaw1_Scale_{datestr}.csv',index=None,header=None)
#
# print("Scaled Done")


2###Create dataset with signle 1 as labeled and remove two rows after each labeled 1, Then scaled data
##*
#dfpd = pd.read_csv("data/paperMachine/processminer-rare-event-mts - tag-map.csv")# columns' descriptions
#dfpAll = pd.read_csv(pathMainCodes+"/data/paperMachine/paperMachine_NewFromWeb/processminer_rep1_4min_desc_aug.csv")# with all repetetive breaks(1 as labeled)

# dfpAll.rename(columns={'y': 'label'}, inplace=True)
# dfpAll["time"]= pd.to_datetime(dfpAll["time"])
# dfpAll['time']=dfpAll['time'].dt.strftime('%m-%d %H:%M')
# dfpAll = dfpAll.drop(['x28', 'x61'], axis=1)
# Main focused dataset that i work on that
##*
##**
# dfpAll["label"]=dfpAll["label"].loc[(dfpAll["label"].shift()!= 1)]
# dfpAll = dfpAll[dfpAll['label'].notna()]
# dfpAll.reset_index(drop=True,inplace=True)

# dfpAll=dfpAll.drop(dfpAll.loc[dfpAll['label'] == 1].index+1)
# dfpAll.reset_index(drop=True,inplace=True)
##**

##***
# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
# x_scaled = min_max_scaler.fit_transform(dfpAll.iloc[:,2:].values)
# dfpAllScale = pd.DataFrame(x_scaled)
# dfpAllScale=pd.concat([dfpAll.iloc[:,[0,1]],dfpAllScale],axis=1)
##***

##****
# dfpAllScale.to_csv(pathData+'dfpAllScale_2RowsDel.csv',index=None)##################################
# dfp=dfp.reset_index()
# dfp.index=dfp["time"]
# dfp.drop(['time','index'], axis=1, inplace=True)
# dfpNorm=dfp[dfp["label"]==0]# filtered Normal data
# dfpAnorm=dfp[dfp["label"]!=0]# filtered Abnormal data
# dfpAnorm.reshape((dfpAnorm.shape[0], dfpAnorm.shape[1], 1))
##****
#End of creation dfpAllScale.csv


3### dropped breaks that distance with previous breaks is less than Five
# dfpLess = pd.read_csv(pathData_Rolling_Auged+"dfpAllScale_2RowsDel.csv",header=None)
# #dfpLess=dfpLess[:2300]
# dfpLess=dfpLess.drop([0],axis=1)
# dfpLess=dfpLess.drop([0],axis=0)
# dfpLess=dfpLess.reset_index(drop=True)
# dfpLess.columns=[i for i in range(dfpLess.shape[1])]
# dfpLess = dfpLess.astype({0: float})
# dfpLess = dfpLess.astype({0: int})
#
# il=[0,0]
# ones=[[0]]
# ll=[]
# # ci=0
# # c5one=True
# for i in range(len(dfpLess)):
#     #if i < len(dfpLess) - 6:
#         if dfpLess.iloc[i, 0] == 1:
#             il.append(i)
#             if i-il[-2]<=5:
#                 mm=np.max(np.where(dfpLess.iloc[:i, 0].values==1))
#                 ones.append(dfpLess.iloc[i:mm:-1].index.values.tolist())
#
# for i in range(1,len(ones)):
#     for j in range(len(ones[i])):
#         ll.append(ones[i][j])
#
# ll.sort()
#
# dfpNoLess5=dfpLess.drop(ll,axis=0)
# dfpNoLess5=dfpNoLess5.reset_index(drop=True)
# datestr = time.strftime("%y%m%d_%H%M%S")
# dfpNoLess5.to_csv(pathData_Rolling_Auged+f"dfpAllScale_2RowsDel_NoLess5_{datestr}.csv",index=None,header=None)


4###start curve_shift, Shift data corresponds to lookback length
###***start curve_shift ##########################################################
# sign = lambda x: (1, -1)[x < 0]
# def curve_shift(df, shift_by):
#     vector = df['label'].copy()
#     for s in range(abs(shift_by)):
#         tmp = vector.shift(sign(shift_by))
#         tmp = tmp.fillna(0)
#         vector += tmp
#     labelcol = 'label'
#     # Add vector to the df
#     df.insert(loc=0, column=labelcol + 'tmp', value=vector)
#     # Remove the rows with labelcol == 1.
#     df = df.drop(df[df[labelcol] == 1].index)
#     # Drop labelcol and rename the tmp col as labelcol
#     df = df.drop(labelcol, axis=1)
#     df = df.rename(columns={labelcol + 'tmp': labelcol})
#     # Make the labelcol binary
#     df.loc[df[labelcol] > 0, labelcol] = 1
#
#     return df
# # end curve_shift#############################################################

###Call Shifting function then shift data corresponds to lookback length
# ###*** Shift data corresponds to lookback length
# dfpNoLess5 = pd.read_csv(pathDataforShifted5+"dfpAllScale_2RowsDel_NoLess5_210123_160259.csv",header=None)
# #dfpNoLess5=dfpNoLess5[:1000]
#
# #dfpnt = dfpNoLess5.drop(["time"], axis=1, inplace=True)
#
# shiftedNumber=5
# dfpNoLess5.rename(columns={0: 'label'}, inplace=True)
# dfpShifted5NoLess5 = curve_shift(dfpNoLess5, shift_by = -1*shiftedNumber)
# dfpShifted5NoLess5 = dfpShifted5NoLess5.astype({"label": int})
# dfpShifted5NoLess5.rename(columns={'label': 0}, inplace=True)
# dfpShifted5NoLess5=dfpShifted5NoLess5.reset_index(drop=True)
#
# datestr = time.strftime("%y%m%d_%H%M%S")
# dfpShifted5NoLess5.to_csv(pathDataforShifted5+f"dfpShifted{shiftedNumber}_NoLess5_{datestr}.csv",index=None,header=None)
# print(f"Creation dfpShifted_{shiftedNumber} is Done!!!")
#################################################################################


5### Rolling shifted(1,2,3,4,5) DSs
# dfpShifted5 = pd.read_csv(pathData_Rolling_Auged+"dfpShifted5_NoLess5_210123_170853.csv",header=None)#
# flagFirstOnes=True
# firstones=[]
# ii=0
# for i in range(len(dfpShifted5)):
#     ii+=1
#     if dfpShifted5.iloc[i,0] == 1 and ii>4:
#         ii=0
#         firstones.append(i)
#         a=0
#
# print(firstones)
#
# def generateRol(X, y, lookback):
#     output_X = []
#     output_y = []
#
#     t = []
#     for j in range(1, 5 + 1):
#         t.append(X[i + j + 1, :])
#
#     lookback = 4
#     i += 2
#     if  (i + lookback) in firstones: # label2 done!!!
#             output_X.append(t)
#             output_y.append(2)  # label2
#             print(f"i as label 2: {i}")
#
#         if (i + lookback) not in firstones and y[i + lookback]==1:
#             output_X.append(t)
#             output_y.append(y[i + lookback])
#
#         if (i + lookback) not in firstones and y[i + lookback]==0:
#             output_X.append(t)
#             output_y.append(y[i + lookback])
#
#     z1 = np.repeat(np.array(output_y), 5)
#     z1 = z1.reshape(z1.shape[0], 1)
#     z2 = np.array(output_X).reshape(len(output_X) * 5, 59)
#     z3 = np.concatenate((z1, z2), axis=1)
#     # return np.squeeze(np.array(output_X)), np.array(output_y)
#     return z3
#
#     # for i in range(-2, len(X) - 5 - 1):
#     #     t = []
#     #     for j in range(1, 5 + 1):
#     #         t.append(X[i + j + 1, :])
#     #
#     #     lookback = 4### because i start from zero and 4th(0,1,2,3,4) value is first label for first block
#     #     i+=2
#     #     output_X.append(t)
#     #     output_y.append(y[i + lookback ])
#     # y2=np.repeat(np.array(output_y),5)
#     # y2=y2.reshape(y2.shape[0],1)
#     # X2=np.array(output_X).reshape(len(output_X)*5,59)
#     # y2X2=np.concatenate((y2, X2), axis=1)
#     # return y2X2
#
# dfpShifted5 = pd.read_csv(pathDataforShifted5+"dfpShifted5_NoLess5_210216_182303.csv",header=None)
#
# input_X = dfpShifted5.iloc[:, 1:].values  # converts the df to a numpy array
# input_y = dfpShifted5.iloc[:,0].values
#
# lookback = 5  # Equivalent to 10 min of past data.
#
# # generateRol data
# yX = generateRol(X = input_X, y = input_y, lookback = lookback)
# dfpShifted5Rolling=pd.DataFrame(yX)
#
# datestr = time.strftime("%y%m%d_%H%M%S")
# dfpShifted5Rolling.to_csv(pathDataforShifted5+f"dfpShifted{shiftedNumber}_Rolling_{datestr}.csv",index=None,header=None)
# print(f"Creation dfpShifted{shiftedNumber}_Rolling_{datestr}.csv is Done!!!")


6###Start Randomly select amount of Twos after rolling
# dfpShifted5RollingCycles5 = pd.read_csv(pathData_AllAguedLabel2ForNoRandomSelected+"dfpShifted5_2_NoLess5_ForLabel2_AfterRolling_210124_155122.csv",header=None)
# dfp5=dfpShifted5RollingCycles5###[:3000]
#
# firstTwos=[]
# ii=0
# for i in range(len(dfp5)):
#     ii+=1
#     if dfp5.iloc[i,0] == 2 and ii>4:
#         ii=0
#         firstTwos.append(i)
#
# ### Randomly select amount of Twos after rolling and before 5To1 and Augmentation step
# firstTwosRandom=random.sample(firstTwos,20)
# print(f"Leb firstTwosRandom: {len(firstTwosRandom)}")
# print(f"firstTwosRandom: {firstTwosRandom}")
#
#
# #dfp5=dfp5[:1500]
# li=[]### could be less than 20 labels2 because in if condition(any()) i filtered labels2 that its 20 rows before are zeros
# liSorted=[]
# dfp5RandomSelected=pd.DataFrame()
#
# for i in range(21,len(dfp5)-28):
#         #if dfp5.iloc[i,0]==2 and dfp5.iloc[i-1,0]==0 and i in firstTwosRandom:#and i==0:###confused classifier
#         if dfp5.iloc[i, 0] == 2 and any(dfp5.iloc[i - 1:i-21:-1, 0]) == 0 and i in firstTwosRandom:  # and i==0:
#             dfp5RandomSelected = dfp5RandomSelected.append(dfp5.iloc[i-1:i -21:-1, :])
#             dfp5RandomSelected = dfp5RandomSelected.append(dfp5.iloc[i:i+25,:])
#
#             li.append(dfp5.iloc[i-1:i -21:-1].index.values.tolist())### Inorder to avoid confusion in label 2 and 0 i have to drop cycles that label 2 is inside them IMPORTANT POINT labels 2 already was zero and were label cycles zeros.
#             li.append(dfp5.iloc[i:i+25].index.values.tolist())### select cycle label2 and labels 1 after this cycle 2
# #h=0
# for i in range(len(li)):
#    #h+=len(li[i])
#    for j in range(len(li[i])):
#        liSorted.append(li[i][j])
#
# liSorted.sort()
# print(f"Leb liSorted: {len(liSorted)}")
# print(f"In order to delete liSorted rows : {liSorted}")
#
#
# dfp5ForNoRandomSelected=dfp5.drop(liSorted,axis=0)
# #df2=df2.reset_index(drop=True)
#
# datestr = time.strftime("%y%m%d_%H%M%S")
#
# # dfp5ForNoRandomSelected.to_csv(pathData_AllAguedLabel2ForNoRandomSelected+f'dfpShifted{shiftedNumber}_ForLabel2_AfterRolling_NoRandomSelected_{datestr}.csv',index=None,header=None)
#
# dfp5RandomSelected.to_csv(pathData_AllAguedLabel2ForNoRandomSelected+f'dfpShifted{shiftedNumber}_ForLabel2_AfterRolling_Added20zerosBeforeRandomSelected_{datestr}.csv',index=None,header=None)
#########End Randomly select amount of Twos after rolling

7###Transform 5 rows to 1 row
######### Start Transform 5 rows to 1 row just ones,
# dfpShifted5NoRandomSelected = pd.read_csv(pathData_AllAguedLabel2ForNoRandomSelected+"dfpShifted5_ForLabel2_AfterRolling_NoRandomSelected_210128_194133.csv",header=None)
#
# dfpShifted5RandomSelected = pd.read_csv(pathData_AllAguedLabel2ForNoRandomSelected+"dfpShifted5_ForLabel2_AfterRolling_Added20zerosBeforeRandomSelected_210129_021057.csv",header=None)
#
#
# shiftedNumber=5

# AllAugedLSTM=pd.concat([
# AugedLSTM1,
# AugedLSTM2,
# AugedLSTM3,
# AugedLSTM4
# ],axis=0)


# datestr = time.strftime("%y%m%d_%H%M%S")
# print(f"Start time running: {datestr}")

#dfpShifted1RollingCycles5=dfpShifted1RollingCycles5[:300]

# y=dfpShifted5NoRandomSelected.iloc[:,0].values
# X=dfpShifted5NoRandomSelected.iloc[:,1:].values
#
# y=y.reshape(int(y.shape[0]/5),5)
# X=X.reshape(int(X.shape[0]/5),X.shape[1]*5)
# #y=y.reshape(int(y.shape[0]/5),5)#,y.shape[1]
# #y=y.reshape(y.shape[0],1)#[:,0]
# #a=np.concatenate((y[:,0].reshape(y.shape[0],1),X),axis=1)
# y=y[:,0].reshape(y.shape[0],1)
#
# yX=np.concatenate((y,X),axis=1)
# dfpShifted5NoRandomSelected5To1=pd.DataFrame(yX)


###############
# y=dfpShifted5RandomSelected.iloc[:,0].values
# X=dfpShifted5RandomSelected.iloc[:,1:].values
#
# y=y.reshape(int(y.shape[0]/5),5)
# X=X.reshape(int(X.shape[0]/5),X.shape[1]*5)
# #y=y.reshape(int(y.shape[0]/5),5)#,y.shape[1]
# #y=y.reshape(y.shape[0],1)#[:,0]
# #a=np.concatenate((y[:,0].reshape(y.shape[0],1),X),axis=1)
# y=y[:,0].reshape(y.shape[0],1)
#
# yX=np.concatenate((y,X),axis=1)
# dfpShifted5RandomSelected5To1=pd.DataFrame(yX)
# for i in range(len(dfp5)):
#     l.append(dfp5.iloc[i, :])
#     if np.mod(len(l),5)==0 and len(l)>0:
#         print(f"Value i: {i}")
#         l = np.reshape(l, (1, np.shape(l)[0] * np.shape(l)[1]))
#         dfpShifted1_Rolling5To1 = dfpShifted1_Rolling5To1.append(pd.DataFrame(l))
#         l=[]

#dfpShifted1_Rolling5To1=pd.DataFrame(np.concatenate([y.reshape(y.shape[0],1),dfpShifted1_Rolling5To1.values],axis=1))

#dfpShifted1_Rolling5To1=pd.concat([pd.DataFrame(y),dfpShifted1_Rolling5To1],axis=1)

# datestr = time.strftime("%y%m%d_%H%M%S")
# dfpShifted1_Rolling5To1.to_csv(pathData_Rolling+f'dfpShifted{shiftedNumber}_Rolling5To1_{datestr}.csv',index=None,header=None)
# dfpShifted5_Rolling5To1.to_csv(pathData_Rolling_Auged+f'dfpShifted{shiftedNumber}_ForLabel2_AfterRolling_Rolling5To1_{datestr}.csv',index=None,header=None)

# dfpShifted5NoRandomSelected5To1.to_csv(pathData_AllAguedLabel2ForNoRandomSelected+f'dfpShifted{shiftedNumber}_ForLabel2_NoRandomSelected_Rolling5To1_{datestr}.csv',index=None,header=None)

# dfpShifted5RandomSelected5To1.to_csv(pathData_AllAguedLabel2ForNoRandomSelected+f'dfpShifted{shiftedNumber}_ForLabel2_Added20zerosBeforeRandomSelected_Rolling5To1_{datestr}.csv',index=None,header=None)
#
# # print(f"Creation of dfpShifted{shiftedNumber}_Rolling5To1 is Done!!!")
# print(f"Creation of dfpShifted{shiftedNumber}_ForLabel2_AfterRolling_Rolling5To1 is Done!!!")
# print(f"Start time running: {datestr}")
######### End Transform 5 rows to 1 row just ones,
#################################################################################

8###Final test model
#
# from mainClassFile import mainClass
# import preprocessRolLbls_CV as aug
#
#
# from pathlib import Path
# pathCurrrent = Path.cwd()
# pathMainCodes = Path.cwd().parent
# pathCurrrent = str(pathCurrrent).replace("\\", '/')
# pathMainCodes = str(pathMainCodes).replace("\\", '/')
#
# pathDataforShifted5 = pathMainCodes + "/data/paperMachine/forCV/forShifted5/"
# pathDataRolLbl1 = pathMainCodes + "/data/paperMachine/forCV/Shifted5_Rol_Lbl1/"
#
# pathSavingPlotsShifted5=pathMainCodes + "/reports/forShifted5/"
#
# baseFileName=os.path.basename(__file__).replace(".py", '')
# #####End Import Libraries
#
#
# ############ Start Running codes
#
#
# datestr = time.strftime("%y%m%d_%H%M%S")
# print(f"Main Start running time: {datestr},******************************:")
# print(f"\nName of corresponding python code file: {baseFileName} \n")
#
# accPerFold = []
# lossPerFold = []
# dfLossEpochTrVal=pd.DataFrame()
# df_ytest_yPred=pd.DataFrame()
# df_ytest_yPredProb=pd.DataFrame()
# dfPrReF1=pd.DataFrame()
#
# ###Tested Datasets
# #dfActual = pd.read_csv(pathDataforShifted5+"dfpShifted5_ForAug_201201_202734_AllTested_Correct_NT_NH.csv",header=None)
# #dfActual = pd.read_csv(pathDataforShifted5+"dfpShifted5ForAug_1To5_FromAllTrainTest_201204_192153.csv",header=None)
#
# #dfActual = pd.read_csv(pathDataRolLbl1+"dfpShifted5_Rolling5To1_210110_214459.csv",header=None)
# # dfActual = pd.read_csv(pathDataRolLbl1+"dfpShifted5_Rolling_210110_191921.csv",header=None)
#
# dfActual = pd.read_csv(pathDataforShifted5+"dfpShifted5_NoLess5_210216_182303.csv",header=None)
#
# #dfActual=dfActual[:10000]
#
# yX=dfActual.values
# X = yX[:, 1:]  # converts the df to a numpy array
# y = yX[:, 0]
#
# # neg, pos = np.bincount(y.astype(int))
# # total = neg + pos
# # print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
# #     total, pos, 100 * pos / total))
# # init_bias = np.log([pos / neg])
#
# print(f"\n   Shape of Actual raw dataset: {yX.shape}")
# print(f"   Number of Actual labeled 0: {len(y[np.where(y==0)])}")
# print(f"   Number of Actual labeled 1: {len(y[np.where(y==1)])} \n")
#
# dataSplitTrainTestPCT=.3
# dataSplitValTestPCT=.5
#
# train_test_split_Shuffle=True
# flagFitShuffle =True# True
# flagSeed=True
# numberOfSplits=5
#
# skf = StratifiedKFold(n_splits=numberOfSplits,shuffle=True)#5
# #skf = KFold(n_splits=5,shuffle=False)#5
# model=0
# mc=mainClass()
#
# AugedWithJittering = False
# if AugedWithJittering == False:
#     baseFileName = baseFileName + "_AugedNN"
# else:
#     baseFileName = baseFileName + "_AugedJitter"
#
# doRolling=True
# if doRolling==False:
#     baseFileName=baseFileName.replace("Rol", '')
#     baseFileName += "_WithoutRolling"
#
# print(f"\nbaseFileName of output Folders and Plots name: {baseFileName} \n")
#
# for foldNum, (trainIndex, testIndex) in enumerate(skf.split(X,y),start=1):
#
#     #print("TRAIN:", trainIndex, "TEST:", testIndex)
#
#     yXtrain, yXtest = yX[trainIndex], yX[testIndex]
#     #ytrain, ytest = y[trainIndex], y[testIndex]
#
#     lookback = 5  # Equivalent to 10 min of past data.
#
#     # Generate Rolling the data
#     if doRolling==True:
#         yXtrain = mc.generateRol(X=yXtrain[:, 1:], y=yXtrain[:, 0], lookback=lookback)
#
#     numberOfActualLbl1=len(yXtrain[np.where(yXtrain[:, 0] == 1)])
#
#     ###***Generate synthetic data with NN
#     AugedNN=aug.GenerateAug_NN_Rolling(yXtrain,foldNum,AugedWithJittering,flagLbl2=False,jitterNum4Lbl1=7,jitterNum4Lbl2=10)
#
#
#     datestrfoldNum = time.strftime("%y%m%d_%H%M%S")
#     print(f"\n Start running time Fold_{foldNum}: {datestrfoldNum} ,--------------------------: \n")
#
#     Actual_AugedNN=np.concatenate((yXtrain,AugedNN),axis=0)
#     #yXtrain=Actual_AugedNN
#
#     yXtrain1, yXtrain2 = train_test_split(Actual_AugedNN, shuffle=train_test_split_Shuffle,
#                                                           test_size=dataSplitTrainTestPCT, random_state=42,
#                                                           stratify=Actual_AugedNN[:,0])  # stratify=input_y
#
#     yXtrain = np.concatenate((yXtrain1, yXtrain2), axis=0)
#
#     yXvalid, yXtest = train_test_split(yXtest, shuffle=train_test_split_Shuffle,
#                                           test_size=dataSplitValTestPCT, random_state=42,
#                                           stratify=yXtest[:, 0])  # stratify=input_y
#
#     print(f"\n    Shape of data to give final model in fold_{foldNum}: ")
#     print(f"\n    xtrain: {np.shape(yXtrain[:,1:])}, ytrain: {np.shape(yXtrain[:,0])}")
#     print(f"    xvalid: {np.shape(yXvalid[:,1:])}, yvalid: {np.shape(yXvalid[:,0])}")
#     print(f"    xtest:  {np.shape(yXtest[:,1:])},  ytest:  {np.shape(yXtest[:,0])} \n")
#
#     print(f"\n   Number of Final label 0 in yXtrain_Fold_{foldNum}: {len(yXtrain[np.where(yXtrain[:, 0] == 0)])}")
#
#     print(f"   Number of Final label 1 in yXtrain_BeforeAddingAugedData_Fold_{foldNum}: "
#           f"{len(Actual_AugedNN[np.where(Actual_AugedNN[:, 0] == 1)]) - len(AugedNN[np.where(AugedNN[:, 0] == 1)])}")
#
#     print(f"   Number of Final label 1 in yXtrain_AfterAddingAugedData_Fold_{foldNum}: "
#           f"{len(Actual_AugedNN[np.where(Actual_AugedNN[:, 0] == 1)]) - numberOfActualLbl1}")
#
#     ytrain = yXtrain  [:,0]
#     yvalid = yXvalid  [:,0]
#     ytest  = yXtest   [:,0]
#
#
#     xtrain = yXtrain  [:,1:]
#     xvalid = yXvalid  [:,1:]
#     xtest  = yXtest   [:,1:]
#
#     epochs = 60#2#0#30#0#100#0#60#400#60#30#30# 150  # 0  # 100#300#60#300#10#200#00#150
#     batch = 32#256
#     lr = 0.001
#
#     #flagR1 = True
#     flagR1=False
#     r1 = .1
#     r2 = .1
#     d1 = .2
#
#     if foldNum==1:
#         print("\n Hyperparameters of final model:")
#         print(f"epochs: {epochs}, batch: {batch}, lr: {lr}, flagFitShuffle: {flagFitShuffle}, numberOfSplits:{numberOfSplits}"
#               f", dataSplitTrainTestPCT: {dataSplitTrainTestPCT}, dataSplitValTestPCT: {dataSplitValTestPCT}"
#               f", train_test_split_Shuffle: {train_test_split_Shuffle}, flagSeed: {flagSeed}\n ")
#
#         ###Create new folder for each run
#         pathSavingPlotsPerRunning = pathSavingPlotsShifted5 + datestr+"_"+baseFileName #+ "_" + modelname
#         if not os.path.exists(pathSavingPlotsPerRunning):
#             os.makedirs(pathSavingPlotsPerRunning)
#
#     del model
#     gc.collect()
#     tf.keras.backend.clear_session()
#     tf.compat.v1.reset_default_graph()
#
#     model = Sequential()
#     model.add(Dense(177, activation='tanh', input_dim=xtrain.shape[1]#,initial_bias=init_bias,#180
#                     #,kernel_regularizer = l1(r1) if flagR1 else l2(r2),
#                     ))  # , input_dim=xtrain.shape[1]
#     #model.add(Dropout(d1))
#
#     model.add(Dense(150, activation='tanh',#,initial_bias=init_bias,#150
#                     #kernel_regularizer=l1(r1) if flagR1 else l2(r2),
#                     # ,# kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
#                     #bias_regularizer=l1(r2),
#                     #activity_regularizer=l1(r1) if flagR1 else l2(r2),
#                     #activity_regularizer=l1(r2)
#                     ))
#     # model.add(Dropout(d1))
#
#     model.add(Dense(90, activation='tanh'#,initial_bias=init_bias,#90
#                     #kernel_regularizer=l1(r1) if flagR1 else l2(r2),
#                     # bias_regularizer=l1(r2),
#                     #activity_regularizer = l1(r1) if flagR1 else l2(r2)
#                     # activity_regularizer=l2(r2)
#                     ))
#     # model.add(Dropout(d1))
#
#     # model.add(Dense(295, activation='tanh'#,initial_bias=init_bias,
#     #                 #kernel_regularizer=l1(r1) if flagR1 else l2(r2),
#     # #                 # bias_regularizer=l1(r2),
#     # #                 activity_regularizer=l1(r1) if flagR1 else l2(r2)
#     # #                 # activity_regularizer=l2(r2)
#     #                 ))
#     # model.add(Dropout(d1))
#
#     model.add(Dense(1, activation='sigmoid'))
#     # model.add(Dense(3, activation='softmax'))
#
#     adam = optimizers.Adam(lr)#lr
#     # cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",save_best_only=True,verbose=0)
#
#     model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
#     #model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
#     # es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
#
#     if foldNum == 1:
#         print("\n Final test model.summary(): \n")
#         print(model.summary())
#
#         print(f"\n model.get_config: {str(model.get_config())} \n")
#
#     # fit model
#     history1 = model.fit(xtrain, ytrain, batch_size=batch, epochs=epochs
#                          , validation_data=(xvalid, yvalid)
#                          , verbose=1, use_multiprocessing=True,
#                          shuffle=flagFitShuffle).history  ### ,shuffle=True#,callbacks=[es]
#
#
#     dfLossEpochTrVal=df_ytest_yPred.append(pd.concat((pd.Series(history1['loss']), pd.Series(history1['val_loss'])), axis=1),ignore_index=True)
#
#     mc.pltLossVal(history1['loss'],history1['val_loss'],foldNum,epochs,pathSavingPlotsPerRunning,baseFileName,flagSeed,ylim=.5)
#
#
#     # plt.figure()
#     # plt.plot(history1['loss'], linewidth=2, label='Train',color="goldenrod")  # OR accuracy
#     # plt.plot(history1['val_loss'], linewidth=2, label='Validation',color="brown")  # OR val_accuracy
#     # plt.legend(loc='upper right', bbox_to_anchor=(1.13, 1.13))
#     # plt.title(f'Model loss Fold_{foldNum}')
#     # plt.ylabel('Loss')
#     # plt.ylim(0,1)#.2
#     # plt.xlabel('Epoch')
#     # plt.savefig(pathSavingPlotsPerRunning +"/" + f"loss&valLoss_Fold_{foldNum}_Epochs{epochs}_flagSeed{flagSeed}.png", dpi=300, format='png')
#     # plt.show()
#
#     yPred = model.predict(xtest, verbose=1)
#     yPredProb=model.predict(xtest, verbose=1)
#     ###Definition of loss function
#     l = []
#     for i in yPred:
#         if i < .5:
#             l.append(0)
#         else:
#             l.append(1)
#
#     yPred = l
#
#     # # apply threshold to positive probabilities to create labels
#     # def to_labels(pos_probs, threshold):
#     # 	return (pos_probs >= threshold).astype('int')
#     #
#     # # define thresholds
#     # thresholds = np.arange(yPred.min(), yPred.max(), 0.01)#min=0.22957686, ,max=0.22973779
#     # # evaluate each threshold
#     # scores = [f1_score(ytest, to_labels(yPred, t)) for t in thresholds]
#     # # get best threshold
#     # ix = np.argmax(scores)
#     # print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
#     #
#     # y_pred_class = [1 if e > thresholds[ix] else 0 for e in yPred]
#     # yPred=y_pred_class
#
#     LABELS = ['Normal 0', 'Anomalous 1']
#     print(f"\nclassification_report_Fold_{foldNum}:")
#     print(classification_report(ytest, yPred, target_names=LABELS))
#
#     df_ytest_yPred=df_ytest_yPred.append(pd.concat((pd.Series(ytest),pd.Series(yPred)),axis=1),ignore_index=True)
#     df_ytest_yPredProb=df_ytest_yPredProb.append(pd.concat((pd.Series(ytest),pd.Series(yPredProb.reshape(-1,))),axis=1),ignore_index=True)
#
#     mc.pltPrRe(ytest, yPredProb,foldNum,epochs,pathSavingPlotsPerRunning,baseFileName,flagSeed)
#     # precision_rt, recall_rt, threshold_rt = precision_recall_curve(ytest, yPred)
#     # pr_re_auc = auc(recall_rt, precision_rt)
#     # print('precision_recall_aucAnomalous= %.3f' % (pr_re_auc),"\n")
#     #
#     # mc.pltPrRe(threshold_rt,precision_rt[1:],recall_rt[1:],foldNum,epochs,pathSavingPlotsPerRunning,flagSeed)
#     # plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=2, color="blue")
#     # plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=2, color="green")
#     # plt.title(f'Precision and recall for different threshold values in Fold_{foldNum}')
#     # plt.xlabel('Threshold')
#     # plt.ylabel('Precision/Recall')
#     # plt.legend()
#     # plt.tight_layout()
#     # plt.savefig(pathSavingPlotsPerRunning +"/" + f"Precision&Recall_Threshold_Fold_{foldNum}_Epochs{epochs}_flagSeed{flagSeed}.png", dpi=300, format='png')
#     # plt.show()
#
#     mc.printConfMatrix(ytest,yPred, foldNum, labelsValues=[0, 1])
#     # print("True Negatives: ", tn)
#     # print("False Positives: ", fp)
#     # print("False Negatives: ", fn)
#     # print("True Positives: ", tp, "\n")
#
#     mc.pltConfMatrix(ytest, yPred,LABELS,foldNum,epochs,pathSavingPlotsPerRunning,baseFileName,flagSeed,figsizeValues=(6, 6), labelsValues=[0, 1])
#     # conf_matrix = confusion_matrix(ytest, yPred, labels=[0, 1])
#     # #LABELS = ["Normal", "Anomalous"]
#     # plt.figure(figsize=(6, 6))
#     # sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d",cmap='YlGnBu');
#     # plt.title(f"Confusion matrix in Fold_{foldNum}")
#     # plt.ylabel('True class')
#     # plt.xlabel('Predicted class')
#     # plt.savefig(pathSavingPlotsPerRunning + "/" + f"ConfusionMatrix_Fold_{foldNum}_Epochs{epochs}_flagSeed{flagSeed}.png",dpi=300, format='png')
#     # plt.show()
#
#     #print(f"accuracy_score_Fold_{foldNum}:\n {accuracy_score(ytest, yPred, normalize=True)} \n")
#
#
#     cr = pd.DataFrame(classification_report(ytest, yPred, target_names=LABELS, output_dict=True))
#     dfPrReF1 = dfPrReF1.append(cr.iloc[:3, :2])
#
#     datestr = time.strftime("%y%m%d_%H%M%S")
#     print(f"End running time Fold_{foldNum}: {datestr} ,--------------------------. \n")
#
#
# print(f"Result Average All Folds is such as below ,$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$: \n")
#
# dfLossEpochTrVal.columns=['loss','val_loss']
# df_ytest_yPred.columns=['ytest','yPred']
# df_ytest_yPredProb.columns=['ytest','yPredProb']
#
#
# # mc.pltLossVal(dfLossEpochTrVal['loss'].tolist(),dfLossEpochTrVal['val_loss'].tolist(),foldNum,epochs,pathSavingPlotsPerRunning,flagSeed,ylim=.5,AllFold=True)
#
# mc.pltPrRe(df_ytest_yPredProb['ytest'].tolist(), df_ytest_yPredProb['yPredProb'].values, foldNum, epochs, pathSavingPlotsPerRunning,baseFileName, flagSeed,AllFold=True)
#
# mc.printConfMatrix(df_ytest_yPred['ytest'].tolist(), df_ytest_yPred['yPred'].tolist(), foldNum, labelsValues=[0, 1],AllFold=True)
#
# mc.pltConfMatrix(df_ytest_yPred['ytest'].tolist(), df_ytest_yPred['yPred'].tolist(),LABELS,foldNum,epochs,pathSavingPlotsPerRunning,baseFileName, labelsValues=[0, 1],AllFold=True)
#
#
# dfPrReF1=pd.DataFrame([np.round(dfPrReF1[dfPrReF1.index=='precision'].mean(),2),np.round(dfPrReF1[dfPrReF1.index=='recall'].mean(),2),np.round(dfPrReF1[dfPrReF1.index=='f1-score'].mean(),2)],index=['precision','recall','f1-score'])
#
# print(f"\nclassification_report_AllFolds:\n {dfPrReF1} \n")
#
#

datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Main End running time: {datestr}, ******************************.")
