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
###**** End tensorflow.keras

tf.random.set_seed(1234)

from pathlib import Path
pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')
pathData = pathMainCodes + "/data/paperMachine/"
pathDataAuged = pathMainCodes + "/data/paperMachine/auged/"
pathDataAuged_jitter_NN = pathMainCodes + "/data/paperMachine/auged/jitterFor_NN/"

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



3###start curve_shift
###***start curve_shift ##########################################################
# sign = lambda x: (1, -1)[x < 0]
# def curve_shift(df, shift_by):
#     '''
#     This function will shift the binary labels in a dataframe.
#     The curve shift will be with respect to the 1s.
#     For example, if shift is -2, the following process
#     will happen: if row n is labeled as 1, then
#     - Make row (n+shift_by):(n+shift_by-1) = 1.
#     - Remove row n.
#     i.e. the labels will be shifted up to 2 rows up.
#
#     Inputs:
#     df       A pandas dataframe with a binary labeled column.
#              This labeled column should be named as 'label'.
#     shift_by An integer denoting the number of rows to shift.
#
#     Output
#     df       A dataframe with the binary labels shifted by shift.
#     '''
#
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
# end curve_shift#############################################################

#Shift data corresponds to lookback length
###*** Shift data corresponds to lookback length
# dfpt = pd.read_csv(pathData+"dfpAllScale_2RowsDel.csv")
# #dfpnt = dfpt.drop(["time"], axis=1)
# #dfp = dfpnt
# dfpShifted5 = curve_shift(dfpt, shift_by = -5)
# dfpShifted5 = dfpShifted5.astype({"label": int})
# dfpShifted5.to_csv(pathData+'dfpShifted5withTime.csv',index=None)
#################################################################################



4###Transform 5 rows to 1 row
###*** Transform 5 rows to 1 row in order to create files for augmentation. 5 Rows--> 1 Row
# dfpShifted5 = pd.read_csv(pathData+"dfpShifted5withTime.csv")
# #dfpShifted5=dfpShifted5[:300]
# #dfpShifted5.drop(['time'], axis=1, inplace=True)
# dfpShifted5_ForAug=pd.DataFrame()
# l=[]
# s=0
# firstOne=True
# ci=0
# c5one=True
# for i in range(len(dfpShifted5)):
#
#     if firstOne==False:
#         ci+=1
#         if ci==5:
#             firstOne=True
#             c5one = True
#             s=i
#             print(f's={s}, time={dfpShifted5.iloc[s,1]}')
#             ci=0
#
#     if i<len(dfpShifted5)-5:
#         if dfpShifted5.iloc[i,0]==1 and firstOne==True :
#
#             l=[]
#             if all(dfpShifted5.iloc[i:i+5,0])==1 and c5one==True:# problem
#                 c5one=False
#                 firstOne = False
#                 dfpShifted5_ForAug = dfpShifted5_ForAug.append(pd.DataFrame(np.reshape(np.array(dfpShifted5.iloc[i:i+5,:]), (1, np.shape(dfpShifted5.iloc[i:i+5,:])[0] * np.shape(dfpShifted5.iloc[i:i+5,:])[1]))))
#
#             else:
#                 countZeroes = dfpShifted5.iloc[i:i+5, 0].values.tolist().count(0)
#                 a = [countZeroes * dfpShifted5.iloc[s, :].tolist()]  # duplicate s as length number of countZeroes
#
#                 b = np.reshape(np.array(dfpShifted5.loc[((dfpShifted5.index >= i) & (dfpShifted5.index<i+5)) & (dfpShifted5.label == 1), :]),(-1,np.array(dfpShifted5.loc[((dfpShifted5.index >= i) & (dfpShifted5.index<i+5)) & (dfpShifted5.label == 1), :]).shape[0]*np.array(dfpShifted5.loc[((dfpShifted5.index >= i) & (dfpShifted5.index<i+5)) & (dfpShifted5.label == 1), :]).shape[1]))  # append ones to end of duplicated zeros(s)
#                 c = np.concatenate((a, b), axis=1)  # make one row
#                 dfpShifted5_ForAug = dfpShifted5_ForAug.append(pd.DataFrame(c))
#
#                 firstOne = False
#
#             for j in range(i-1,s,-1):
#                 firstOne = False
#                 l.append(dfpShifted5.iloc[j,:])
#                 if np.mod(len(l),5)==0 and len(l)>0:
#                     l = np.reshape(l, (1, np.shape(l)[0] * np.shape(l)[1]))
#                     dfpShifted5_ForAug = dfpShifted5_ForAug.append(pd.DataFrame(l))
#                     l=[]
#                     #l.append(dfpShifted5.iloc[j,:])
#
#
#
# #print(dfpShifted5_ForAug)
# datestr = time.strftime("%y%m%d_%H%M%S")
# dfpShifted5_ForAug.to_csv(pathData+f'dfpShifted5_ForAug_{datestr}.csv',index=None,header=None)
# print('Creation of file for augmentation was done: '+f'dfpShifted5_ForAug_{datestr}.csv')
#################################################################################



5###Filter one row zero labeled just before ones and add its ones
# df = pd.read_csv(pathData+"paperMachine_NewFromWeb/"+"dfpShifted5_ForAug_201201_202734_AllTested_Correct_NT_NH.csv",header=None)
# df2=pd.DataFrame()#dfpShifted5_ForAugOneZero
# for i in range(len(df)):
#     if df.iloc[i,0]==1 and i==0:#and i==0:
#         df2 = df2.append(df.iloc[i])
#     if df.iloc[i,0]==1 and df.iloc[:i,0].tail(1).values==0 and i!=0:
#         df2 = df2.append(df.iloc[:i].tail(1))
#         df2=df2.append(df.iloc[i])
#         print("0,1",i)
#     if df.iloc[i, 0] == 1 and df.iloc[:i,0].tail(1).values==1 and i != 0:  # and i==0:
#         df2 = df2.append(df.iloc[i])
#         print("1", i)
#
# print('Finished!!!')
# datestr = time.strftime("%y%m%d_%H%M%S")
# df2.to_csv(pathData+f'dfpShifted5_ForAugOneZero{datestr}.csv',index=None,header=None)
# print('Done!!!')
#################################################################################


6###Transform 5 rows to 1 row just ones
# AugedLSTM1 = pd.read_csv(pathDataAuged+"dfpShifted5_AugedLSTM_Ones_1To5_201224_000736.csv",header=None)
# AugedLSTM2 = pd.read_csv(pathDataAuged+"dfpShifted5_AugedLSTM_Ones_1To5_201224_001123.csv",header=None)
# AugedLSTM3 = pd.read_csv(pathDataAuged+"dfpShifted5_AugedLSTM_Ones_1To5_201224_002149.csv",header=None)
# AugedLSTM4 = pd.read_csv(pathDataAuged+"dfpShifted5_AugedLSTM_Ones_1To5_201224_002526.csv",header=None)
#
# AllAugedLSTM=pd.concat([
# AugedLSTM1,
# AugedLSTM2,
# AugedLSTM3,
# AugedLSTM4
# ],axis=0)
#
# AllAugedLSTM2=pd.DataFrame()
# l=[]
#
# for i in range(len(AllAugedLSTM)):
#     l.append(AllAugedLSTM.iloc[i, :])
#     if np.mod(len(l),5)==0 and len(l)>0:
#         l = np.reshape(l, (1, np.shape(l)[0] * np.shape(l)[1]))
#         AllAugedLSTM2 = AllAugedLSTM2.append(pd.DataFrame(l))
#         l=[]
#
# datestr = time.strftime("%y%m%d_%H%M%S")
# AllAugedLSTM2.to_csv(pathData+f'dfpShifted5_AllAugedLSTM_{datestr}.csv',index=None,header=None)
# print("AllAugedLSTM2 Finished!!!")
#################################################################################



7###Create shuffle data(All) for augmentation from _ForAug file.
###***Create shuffle data(All) for augmentation from _ForAug file. dfpShifted5_AllAugedLSTM_201224_004144
#
# dfpShifted5_ForAug = pd.read_csv(pathData+"dfpShifted5_ForAug_AllTrainTest_201204_161806.csv",header=None)
# #AllAugedLSTM = pd.read_csv(pathData+"dfpShifted5_AllAugedLSTM_201224_004144.csv",header=None)
# jitter1_NN = pd.read_csv(pathDataAuged_jitter_NN+"dfpShifted5_Auged_NN_Ones_5To1_peyman2_201227_195021.csv",header=None)
# jitter2_NN = pd.read_csv(pathDataAuged_jitter_NN+"dfpShifted5_Auged_NN_Ones_5To1_jitter_201228_201302.csv",header=None)
# jitter3_NN = pd.read_csv(pathDataAuged_jitter_NN+"dfpShifted5_Auged_NN_Ones_5To1_jitter_201228_201802.csv",header=None)
# jitter4_NN = pd.read_csv(pathDataAuged_jitter_NN+"dfpShifted5_Auged_NN_Ones_5To1_jitter_201228_201914.csv",header=None)
#
# dfpShifted5_AllAuged=pd.concat([
# jitter1_NN,
# jitter2_NN,
# jitter3_NN,
# jitter4_NN
# ],axis=0)
#
# dfpShifted5_AllAuged.insert(0,-1,1)
# dfpShifted5_AllAuged.columns=[i for i in range(dfpShifted5_AllAuged.shape[1])]
#
# dfpShifted5_ForAug_AllAugedJitter_NN=pd.concat([
# dfpShifted5_ForAug,
# dfpShifted5_AllAuged],axis=0)
#
# train, test = train_test_split(dfpShifted5_ForAug_AllAugedJitter_NN,test_size=0.3, random_state=42,shuffle=True,stratify=dfpShifted5_ForAug_AllAugedJitter_NN.iloc[:,0])
#
# datestr = time.strftime("%y%m%d_%H%M%S")
# AllTrainTest=pd.concat([train,test],axis=0)
# AllTrainTest.to_csv(pathData+f'dfpShifted5_ForAug_AllAugedJitter_NN_{datestr}.csv',index=None,header=None)
#
# # train.to_csv(pathData+f'dfpShifted5_ForAug_train_{datestr}.csv',index=None,header=None)
# # test.to_csv(pathData+f'dfpShifted5_ForAug_test_{datestr}.csv',index=None,header=None)
#
# print('Creation of  train and test files for augmentation was done!!!')
#################################################################################

8###Transform data from auged rows(1 row) to normal rows(5)
###*** Transform data from auged rows(1 row) to normal rows(5) for comparing results augmentation methods. 1 Row--> 5 rows
#dfpShifted5ForAug = pd.read_csv(pathData+"dfpShifted5_ForAug_AllTrainTest_201204_161806.csv",header=None)
# dfpShifted5_ForAug_AllAugedLSTM = pd.read_csv(pathData+"dfpShifted5_ForAug_AllAugedLSTM_201224_021009.csv",header=None)

#dfpShifted5ForAug_1To5=pd.DataFrame()
# dfpShifted5_ForAug_AllAugedLSTM_1To5=pd.DataFrame()

# DFs5To1=[dfpShifted5ForAug,dfpShifted5_ForAug_AllAugedLSTM]
# DFs1To5=[dfpShifted5ForAug_1To5,dfpShifted5_ForAug_AllAugedLSTM_1To5]

# DFs5To1=[dfpShifted5_ForAug_AllAugedLSTM]
# DFs1To5=[dfpShifted5_ForAug_AllAugedLSTM_1To5]
#
# dfi=-1
#
# for df in DFs5To1:
#     if len(DFs1To5)>1:
#         dfi=dfi+1
#         for i,row in df.iterrows():
#             DFs1To5[dfi]=DFs1To5[dfi].append(pd.concat([pd.Series(list(np.array([row[0]] * 5).astype(int))),
#                        pd.DataFrame(np.reshape(np.array(row[1:]), (5, int(row[1:].shape[0] / 5))))], axis=1))
#             print(i)
#     else:
#         dfi = dfi + 1
#         for i, row in df.iterrows():
#             DFs1To5[dfi] = DFs1To5[dfi].append(pd.concat([pd.Series(list(np.array([row[0]] * 5).astype(int))),
#                                                           pd.DataFrame(np.reshape(np.array(row[1:]),
#                                                                                   (5, int(row[1:].shape[0] / 5))))],
#                                                          axis=1))
#             print(i)
#
#
# datestr = time.strftime("%y%m%d_%H%M%S")

# DFs1To5[0].to_csv(pathData+f'dfpShifted5ForAug_1To5_{datestr}.csv',index=None,header=None)
# DFs1To5[1].to_csv(pathData+f'dfpShifted5JitterAuged_1To5_{datestr}.csv',index=None,header=None)

#dfpShifted5_ForAug_AllAugedLSTM=pd.concat([DFs1To5[0],DFs1To5[1]],axis=0)
# DFs1To5[0].to_csv(pathData+f'dfpShifted5_ForAug_AllAugedLSTM_1To5_{datestr}.csv',index=None,header=None)
# print('dfpShifted5_ForAug_AllAugedLSTM_1To5_ Finished!!!')
#################################################################################

9###Transform data from auged rows(1 row) to normal rows(5),dfpShifted5JitterAuged[0]==1.
###*** Transform data from auged rows(1 row) to normal rows(5) for comparing results augmentation methods. 1 Row--> 5 rows
# dfpShifted5ForAug = pd.read_csv(pathData+"dfpShifted5_ForAug_AllTrainTest_201204_161806.csv",header=None)
# # # dfpShifted5JitterAuged = pd.read_csv(pathData+"df_jitter_201204_162219.csv",header=None)
# # # dfpShifted5JitterAuged = pd.read_csv(pathData+"df_jitter2_201216_235245.csv",header=None)
# # # dfpShifted5JitterAuged = pd.read_csv(pathData+"df_jitter2_201216_235245.csv",header=None)#dfpShifted5_ForAugOneZero
# # dfpShifted5JitterAuged = pd.read_csv(pathDataAuged+"df_jitter2_OneZero_201220_013335.csv",header=None)#dfpShifted5_ForAugOneZero
# dfpShifted5ForAug = pd.read_csv(pathData+"dfpShifted5_ForAug_201201_202734_AllTested_Correct_NT_NH.csv",header=None)
# peyman1 = pd.read_csv(pathDataAuged+"dfpShifted5_Auged_NN_Ones_5To1_peyman_201225_011159.csv",header=None)
# peyman2 = pd.read_csv(pathDataAuged+"dfpShifted5_Auged_NN_Ones_5To1_peyman_201225_011440.csv",header=None)
# peyman3 = pd.read_csv(pathDataAuged+"dfpShifted5_Auged_NN_Ones_5To1_peyman_201225_011446.csv",header=None)
# peyman4 = pd.read_csv(pathDataAuged+"dfpShifted5_Auged_NN_Ones_5To1_peyman_201225_011501.csv",header=None)

# # df_Magnitude = pd.read_csv(pathDataAuged+"df_Magnitude_201219_205807.csv",header=None)#Magnitude
#
# # dfFawaz1 = dfFawaz1.loc[dfFawaz1[0]==1]
# # ###dfpShifted5_ForAug_withAuged=dfFawaz1
# # dfFawaz2 = dfFawaz2.loc[dfFawaz2[0]==1]
# # dfFawaz3 = dfFawaz3.loc[dfFawaz3[0]==1]
# # dfFawaz4 = dfFawaz4.loc[dfFawaz4[0]==1]
# # dfpShifted5_ForAug_withAuged=dfFawaz1
#
# # dfpShifted5_ForAug_withAuged = df_Magnitude.loc[df_Magnitude[0]==1]
#
# dfpShifted5_ForAug_withAuged=pd.concat([dfpShifted5ForAug,
# peyman1,
# peyman2,
# peyman3,
# peyman4],axis=0)
#
# train, test = train_test_split(dfpShifted5_ForAug_withAuged,test_size=0.3, random_state=42,shuffle=True,stratify=dfpShifted5_ForAug_withAuged.iloc[:,0])
# AllTrainTest=pd.concat([train,test],axis=0)
# datestr = time.strftime("%y%m%d_%H%M%S")
# AllTrainTest.to_csv(pathData+f'dfpShifted5_ForAug_withAuged_NN_5To1_peyman_{datestr}.csv',index=None,header=None)
# print('Done!!!')

# # # dfpShifted5_ForAug_withAuged=pd.concat([dfpShifted5ForAug,dfpShifted5JitterAuged],axis=0)
# # # # dfpShifted5_ForAug_withAuged=pd.concat([dfpShifted5ForAug,dfFawaz1,
# # # # dfFawaz2,
# # # # dfFawaz3,
# # # # dfFawaz4],axis=0)
# # #
# # # train, test = train_test_split(dfpShifted5_ForAug_withAuged,test_size=0.3, random_state=42,shuffle=True,stratify=dfpShifted5_ForAug_withAuged.iloc[:,0])
# # #
# # dfpShifted5_ForAug_withAuged=pd.concat([train,test],axis=0)
#
# dfpShifted5_ForAug_withAuged_1To5=pd.DataFrame()
#
# for i,row in dfpShifted5_ForAug_withAuged.iterrows():
#         dfpShifted5_ForAug_withAuged_1To5=dfpShifted5_ForAug_withAuged_1To5.append\
#             (pd.concat([pd.Series(list(np.array([row[0]] * 5).astype(int))),
#             pd.DataFrame(np.reshape(np.array(row[1:]), (5, int(row[1:].shape[0] / 5))))], axis=1))
#         print(i)
# #
# datestr = time.strftime("%y%m%d_%H%M%S")
# # # #dfpShifted5_ForAug_withAuged_1To5.to_csv(pathData+f'dfpShifted5_ForAug_withAuged_1To5_{datestr}.csv',index=None,header=None)
# # # #dfpShifted5_ForAug_withAuged_1To5.to_csv(pathData+f'dfpShifted5_ForAug_withAuged_OneZero_1To5_{datestr}.csv',index=None,header=None)
# # # dfpShifted5_ForAug_withAuged_1To5.to_csv(pathData+f'dfpShifted5_ForAug_withAugedJitter_OneZero_1To5_{datestr}.csv',index=None,header=None)
# # dfpShifted5_ForAug_withAuged_1To5.to_csv(pathData+f'dfpShifted5_ForAug_withAugedFawaz_OneZero_1To5_{datestr}.csv',index=None,header=None)
# #dfpShifted5_ForAug_withAuged_1To5.to_csv(pathData+f'dfpShifted5_ForAug_withAugedFawaz4_OneZero_1To5_{datestr}.csv',index=None,header=None)
# # dfpShifted5_ForAug_withAuged_1To5.to_csv(pathData+f'dfpShifted5_AugedMagnitude_OneZero_1To5_{datestr}.csv',index=None,header=None)
# # dfpShifted5_ForAug_withAuged_1To5.to_csv(pathData+f'dfpShifted5_AugedFawazAll_OneZero_1To5_{datestr}.csv',index=None,header=None)
# dfpShifted5_ForAug_withAuged_1To5.to_csv(pathData+f'dfpShifted5_ForAug_withAuged_NN_1To5_peman_{datestr}.csv',index=None,header=None)
#################################################################################

10###NN model tests
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

datestr = time.strftime("%y%m%d_%H%M%S")
print(f"End running time: {datestr}")
