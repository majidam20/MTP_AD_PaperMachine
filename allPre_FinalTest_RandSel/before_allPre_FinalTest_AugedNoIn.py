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

from pathlib import Path
pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')

pathDataAllPre_FinalTest = pathMainCodes + "/data/paperMachine/forAllPre_FinalTest/"

pathSavingPlotsAllPre_FinalTest=pathMainCodes + "/reports/forAllPre_FinalTest/"

shiftedNumber=5

1#### actualRaw_data-augmentation dataset has all repetetive breaks(1 as labeled), Here create DS with having just one ActualRaw row as break, Then scaled data and drop some unnecessary columns.

# Here, the raw data is being loaded
dfpRaw1 = pd.read_csv(pathDataAllPre_FinalTest+"actualRaw_data-augmentation.csv",header=None)


# Not wanted data has been dropped
dfpRaw2 = dfpRaw1.drop([0,29,62], axis=1)
dfpRaw2.columns=[i for i in range(dfpRaw2.shape[1])]
dfpRaw2=dfpRaw2.drop([0],axis=0)
dfpRaw2=dfpRaw2.reset_index(drop=True)#inplace=True,

# Covert from Pandas to Numpy array
dfpRaw2=dfpRaw2.values

# All data will be converted to float64
dfpRaw2=np.asarray(dfpRaw2).astype('float64')

# It is needed to extra "1" after breaks be deleted
flagFirstOne=False
li=[]
for i in range(dfpRaw2.shape[0]):
    if dfpRaw2[i,0]==1 and flagFirstOne==False:
        flagFirstOne=True
        continue
    if flagFirstOne==True and dfpRaw2[i,0]==1:
        li.append(i)
    if dfpRaw2[i,0]==0:
        flagFirstOne=False

dfpRaw2=np.delete(dfpRaw2, li, axis = 0)
#print("Repetitive Actual Raw labels One were deleted!!! \n")



min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
x_scaled = min_max_scaler.fit_transform(dfpRaw2[:,1:])
dfpRaw2Scaled=np.concatenate((dfpRaw2[:,0].reshape(-1,1),x_scaled),axis=1)
print("Scaling Actual Raw DS was Done!!! \n")

##****
###Two Rows after Actual Raw labels One will be deleted(it means noisy data after each break will be deleted)
indexLblOnes=np.where(dfpRaw2Scaled[:,0]==True)
l=[]
for i in range(1,3):
    for j in indexLblOnes:
        l.append(j+i)
l1=np.array(l)
l2=l1.reshape(l1.shape[0]*l1.shape[1],).tolist()

dfpRaw3=np.delete(dfpRaw2Scaled, l2, axis = 0)
print("Two Rows after Actual Raw labels One were deleted(it means noisy data after each break deleted)!!! \n")
##**

2### dropped breaks that distance with previous breaks is less than Five
il=[0,0]
ones=[[0]]
ll=[]

il=[0,0]
ones=[[0]]
ll=[]

for i in range(dfpRaw3.shape[0]):
        if dfpRaw3[i, 0] == 1:
            il.append(i)
            if i-il[-2]<=5:###il[-2] means second last number,,, so when difference length second last number(li[-2]) and than last number(li[-1]) is less than 5  rows then from i(last number in li[-1])  number to(not up to) second last number(li[-2]) will be added to ones(index rows that are less than 5 between two breaks)

                mm=np.max(np.where(dfpRaw3[:i, 0]==1))
                ones.append(np.arange(i,mm,-1).tolist())

for i in range(1,len(ones)):
    for j in range(len(ones[i])):
        ll.append(ones[i][j])


ll.sort()
dfpRaw4=np.delete(dfpRaw3,ll , axis = 0)
#dfpNoLess5=dfpRaw4.drop(ll,axis=0)
#dfpNoLess5=dfpNoLess5.reset_index(drop=True)
datestr = time.strftime("%y%m%d_%H%M%S")

print("Dropping breaks that distance with previous breaks is less than Five was Done!!! \n")



3#start curve_shift, Shift data corresponds to lookback length
###***start curve_shift ##########################################################
sign = lambda x: (1, -1)[x < 0]
def curve_shift(df, shift_by):
    vector = df['label'].copy()
    for s in range(abs(shift_by)):
        tmp = vector.shift(sign(shift_by))
        tmp = tmp.fillna(0)
        vector += tmp
    labelcol = 'label'
    # Add vector to the df
    df.insert(loc=0, column=labelcol + 'tmp', value=vector)
    # Remove the rows with labelcol == 1.
    df = df.drop(df[df[labelcol] == 1].index)
    # Drop labelcol and rename the tmp col as labelcol
    df = df.drop(labelcol, axis=1)
    df = df.rename(columns={labelcol + 'tmp': labelcol})
    # Make the labelcol binary
    df.loc[df[labelcol] > 0, labelcol] = 1

    return df
# end curve_shift#############################################################

##Call Shifting function then shift data corresponds to lookback length
shiftedNumber=5
dfpRaw5=pd.DataFrame(dfpRaw4)
dfpRaw5.rename(columns={0: 'label'}, inplace=True)
dfpShifted5NoLess5 = curve_shift(dfpRaw5, shift_by = -1*shiftedNumber)
dfpShifted5NoLess5 = dfpShifted5NoLess5.astype({"label": int})
dfpShifted5NoLess5.rename(columns={'label': 0}, inplace=True)
dfpShifted5NoLess5=dfpShifted5NoLess5.reset_index(drop=True)

dfpShifted5NoLess5=dfpShifted5NoLess5.values
datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Creation of dfpShifted_{shiftedNumber} was Done!!! \n")
#################################################################################


4### Generate Rolling data from shifted5 DS

firstOnes=[]

i=0
while i in range(dfpShifted5NoLess5.shape[0]):
    if i==dfpShifted5NoLess5.shape[0]-1:
        break
    if dfpShifted5NoLess5[i,0] == 0:
        i+=1
    if dfpShifted5NoLess5[i,0] == 1:
        firstOnes.append(i)
        i += 5

print(f"firstOnes of dfpShifted_{shiftedNumber} was Selected!!! \n")



#### Randomly select amount of Twos after rolling and before 5To1 and Augmentation step
firstOnesRandom=random.sample(firstOnes,20)

rowsBeforeAfterRanSel=[]
for i in range(dfpShifted5NoLess5.shape[0]):
    if  i  in firstOnesRandom:
        rowsBeforeAfterRanSel.append(np.arange(i,i+5))
        rowsBeforeAfterRanSel.append(np.arange(i-5,i))

rowsBeforeAfterRanSel=np.concatenate(rowsBeforeAfterRanSel, axis=0 )
print(f" rowsBeforeAfterRanSel firstOnes of dfpShifted_{shiftedNumber} was Selected!!! \n")


rowsBeforeAfterRanSel.sort()
dfpShifted5NoLess5RanSel=dfpShifted5NoLess5[rowsBeforeAfterRanSel,:]
dfpShifted5NoLess5NoRanSel=dfpShifted5NoLess5[np.delete(np.arange(dfpShifted5NoLess5.shape[0]),rowsBeforeAfterRanSel),:]

print(f"Creation dfpShifted{shiftedNumber}_BeforeRolling NoRanSel and NoRanSel, is Done!!! \n")
print(f"Clarification: NoRanSel For train purpose,  and NoRanSel For test purpose!!! \n")





firstOnesRanSel=[]
i=0
while i in range(dfpShifted5NoLess5RanSel.shape[0]):
    if i==dfpShifted5NoLess5RanSel.shape[0]-1:
        break
    if dfpShifted5NoLess5RanSel[i,0] == 0:
        i+=1
    if dfpShifted5NoLess5RanSel[i,0] == 1:
        firstOnesRanSel.append(i)
        i += 5





firstOnesNoRanSel=[]
i=0
while i in range(dfpShifted5NoLess5NoRanSel.shape[0]):
    if i==dfpShifted5NoLess5NoRanSel.shape[0]-1:
        break
    if dfpShifted5NoLess5NoRanSel[i,0] == 0:
        i+=1
    if dfpShifted5NoLess5NoRanSel[i,0] == 1:
        firstOnesNoRanSel.append(i)
        i += 5


#### GenerateRol data

def generateRol(X, y,firstOnes, lookback):
    output_X = []
    output_y = []

    for i in range(-2, len(X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            t.append(X[i + j + 1, :])

        lookback2 = 4
        i += 2

        if  (i + lookback2) in firstOnes: # label2 done!!!
                output_X.append(t)
                output_y.append(2)  # label2
                #print(f"i as label 2: {i}")

        if (i + lookback2) not in firstOnes and y[i + lookback2]==1:
            output_X.append(t)
            output_y.append(y[i + lookback2])

        if (i + lookback2) not in firstOnes and y[i + lookback2]==0:
                output_X.append(t)
                output_y.append(y[i + lookback2])

    y1 = np.repeat(np.array(output_y), 5)
    y2 = y1.reshape(y1.shape[0], 1)
    X = np.array(output_X).reshape(len(output_X) * 5, 59)
    yX = np.concatenate((y2, X), axis=1)
    return yX


input_X = dfpShifted5NoLess5RanSel[:,1:]
input_y = dfpShifted5NoLess5RanSel[:,0]
lookback = 5
#### generateRol data
yX1_RanSel = generateRol(X = input_X, y = input_y,firstOnes=firstOnesRanSel, lookback = lookback)


#Call function generateRol data
input_X = dfpShifted5NoLess5NoRanSel[:,1:]
input_y = dfpShifted5NoLess5NoRanSel[:,0]
yX1_NoRanSel = generateRol(X = input_X, y = input_y,firstOnes=firstOnesNoRanSel, lookback = lookback)

datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Creation dfpShifted{shiftedNumber}_Rolling NoRanSel and NoRanSel, is Done!!! \n")

4-1###
#######Start Randomly select amount of Twos after rolling
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

5##Transform 5 rows to 1 row
######### Start Transform 5 rows to 1 row just ones,

###RanSel Part
y1_RanSel=yX1_RanSel[:,0]
X1_RanSel=yX1_RanSel[:,1:]

y11_RanSel=y1_RanSel.reshape(int(y1_RanSel.shape[0]/5),5)
y111_RanSel=y11_RanSel[:,0].reshape(y11_RanSel[:,0].shape[0],1)
X11_RanSel=X1_RanSel.reshape(int(X1_RanSel.shape[0]/5),X1_RanSel.shape[1]*5)
yX2_RanSel=np.concatenate((y111_RanSel,X11_RanSel),axis=1)
yXfinal_RanSel=yX2_RanSel


###NoRanSel Part
y1_NoRanSel=yX1_NoRanSel[:,0]
X1_NoRanSel=yX1_NoRanSel[:,1:]

y11_NoRanSel=y1_NoRanSel.reshape(int(y1_NoRanSel.shape[0]/5),5)
y111_NoRanSel=y11_NoRanSel[:,0].reshape(y11_NoRanSel[:,0].shape[0],1)
X11_NoRanSel=X1_NoRanSel.reshape(int(X1_NoRanSel.shape[0]/5),X1_NoRanSel.shape[1]*5)
yX2_NoRanSel=np.concatenate((y111_NoRanSel,X11_NoRanSel),axis=1)
yXfinal_NoRanSel=yX2_NoRanSel

print(f"Creation 5 rows to 1 row, dfpShifted{shiftedNumber}_After_Rolling NoRanSel and NoRanSel , is Done!!! \n")
###############

5-1###
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



6###******************************Final model test
from mainClassFile_NoRanSel_RanSel import mainClass
import preprocessRolLbl2_CV as aug
mc=mainClass()

baseFileName=os.path.basename(__file__).replace(".py", '')

############# Start Running codes

datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Main Start running time: {datestr},******************************:")
print(f"\nName of corresponding python code file: {baseFileName} \n")


###Final datasets for passing to Final model
print(f"Clarification: NoRanSel For train purpose,  and NoRanSel For test purpose!!! \n")
#yX2_finalRanSel
#yX2_finalNoRanSel

yX_RanSel=yXfinal_RanSel
X_RanSel = yX_RanSel[:, 1:]
y_RanSel = yX_RanSel[:, 0]

yX_NoRanSel=yXfinal_NoRanSel
X_NoRanSel = yX_NoRanSel[:, 1:]
y_NoRanSel = yX_NoRanSel[:, 0]


print(f"\n   Shape of Actual RanSel   raw dataset: {yX_RanSel.shape}")
print(f"   Number of Actual  RanSel  labeled 0: {len(y_RanSel[np.where(y_RanSel==0)])}")
print(f"   Number of Actual  RanSel  labeled 1: {len(y_RanSel[np.where(y_RanSel==1)])} \n")
print(f"   Number of Actual  RanSel  labeled 2: {len(y_RanSel[np.where(y_RanSel==2)])} \n")

print(f"\n   Shape of Actual NoRanSel raw dataset: { yX_NoRanSel.shape}")
print(f"   Number of Actual  NoRanSel labeled 0: {len(y_NoRanSel[np.where(y_NoRanSel==0)])}")
print(f"   Number of Actual  NoRanSel labeled 1: {len(y_NoRanSel[np.where(y_NoRanSel==1)])} \n")
print(f"   Number of Actual  NoRanSel labeled 2: {len(y_NoRanSel[np.where(y_NoRanSel==2)])} \n")




dataSplitTrainTestPCT=.3
dataSplitValTestPCT=.5

train_test_split_Shuffle=True
flagFitShuffle =True


print(f"\nbaseFileName of output Folders and Plots name: {baseFileName} \n")


numberOfActualLbl2=len(yX_NoRanSel[np.where(yX_NoRanSel[:, 0] == 2)])


###***Generate synthetic data with NN
AugedNN=aug.GenerateAug_NN_Rolling(yX_NoRanSel,jitterNum4Lbl2=20)



datestrfoldNum = time.strftime("%y%m%d_%H%M%S")
print(f"\n Start running time ,--------------------------: \n")

Actual_AugedNN=np.concatenate((yX_NoRanSel,AugedNN),axis=0)

yXtrain1, yXtrain2 = train_test_split(Actual_AugedNN, shuffle=train_test_split_Shuffle,
                                                      test_size=dataSplitTrainTestPCT, random_state=42,
                                                      stratify=Actual_AugedNN[:,0])

yXtrain = np.concatenate((yXtrain1, yXtrain2), axis=0)


yXvalid, yXtest = train_test_split(yX_RanSel, shuffle=train_test_split_Shuffle,
                                      test_size=dataSplitValTestPCT, random_state=42,
                                      stratify=yX_RanSel[:, 0])



print(f"\n    Shape of data to give final model: ")
print(f"\n    xtrain: {np.shape(yXtrain[:,1:])}, ytrain: {np.shape(yXtrain[:,0])}")
print(f"    xvalid: {np.shape(yXvalid[:,1:])}, yvalid: {np.shape(yXvalid[:,0])}")
print(f"    xtest:  {np.shape(yXtest[:,1:])},  ytest:  {np.shape(yXtest[:,0])} \n")


print(f"\n   Number of Final label 0 in yXtrain: {len(yXtrain[np.where(yXtrain[:, 0] == 0)])}")
print(f"\n   Number of Final label 1 in yXtrain: {len(yXtrain[np.where(yXtrain[:, 0] == 1)])}")

print(f"   Number of Final label 2 in yXtrain_BeforeAddingAugedData: "
      f"{len(Actual_AugedNN[np.where(Actual_AugedNN[:, 0] == 2)]) - len(AugedNN[np.where(AugedNN[:, 0] == 2)])}")

print(f"   Number of Final label 2 in yXtrain_AfterAddingAugedData: "
      f"{len(Actual_AugedNN[np.where(Actual_AugedNN[:, 0] == 2)]) - numberOfActualLbl2}")




ytrain = to_categorical(yXtrain[:, 0])
yvalid = to_categorical(yXvalid[:, 0])
ytest = to_categorical(yXtest[:, 0])

xtrain = yXtrain  [:,1:]
xvalid = yXvalid  [:,1:]
xtest  = yXtest   [:,1:]




###****Hyperparameters of final model
epochs = 500
batch = 256
lr = 0.001


print("\n Hyperparameters of final model:")
print(f"epochs: {epochs}, batch: {batch}, lr: {lr}, flagFitShuffle: {flagFitShuffle}"
      f", dataSplitTrainTestPCT: {dataSplitTrainTestPCT}, dataSplitValTestPCT: {dataSplitValTestPCT}"
      f", train_test_split_Shuffle: {train_test_split_Shuffle}\n ")

###Create new folder for each run
# pathSavingPlotsPerRunning = pathSavingPlotsShifted5 + datestr+"_"+baseFileName #+ "_" + modelname
# if not os.path.exists(pathSavingPlotsPerRunning):
#         os.makedirs(pathSavingPlotsPerRunning)


model = Sequential()
model.add(Dense(590, activation='tanh', input_dim=xtrain.shape[1]))

model.add(Dense(500, activation='tanh'))

model.add(Dense(400, activation='tanh'))

model.add(Dense(3, activation='softmax'))

adam = optimizers.Adam(lr)

###cp = ModelCheckpoint(filepath=pathSavingPlotsPerRunning+ "/NN_Mlbls_classifier.h5",save_best_only=True,verbose=0)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


print("\n Final test model.summary(): \n")
print(model.summary())
print(f"\n model.get_config: {str(model.get_config())} \n")

# fit model
history1 = model.fit(xtrain, ytrain, batch_size=batch, epochs=epochs
                     , validation_data=(xvalid, yvalid)
                     , verbose=1, use_multiprocessing=True,
                     shuffle=flagFitShuffle).history



mc.pltLossVal(history1['loss'],history1['val_loss'],epochs,ylim=.5)


yPred = model.predict(xtest, verbose=1)
yPredProb=model.predict(xtest, verbose=1)


LABELS = ['Normal 0', 'Anomalous 1', 'Anomalous 5 step ahead']

print(f"classification_report:\n {classification_report(ytest.argmax(axis=1), yPred.argmax(axis=1))} \n")

print(f"confusion_matrix:\n {confusion_matrix(ytest.argmax(axis=1), yPred.argmax(axis=1))} \n")

mlbConfusion = multilabel_confusion_matrix(ytest.argmax(axis=1), yPred.argmax(axis=1))
print(f"multilabel_confusion_matrix:\n {mlbConfusion} \n")


# mlbClasses = [0, 1, 2]
#
# mc.printConfMatrixMlbls(ytest,yPred, labelsValues=mlbClasses)
#
# mc.pltConfMatrixMlbls(mlbConfusion,LABELS,epochs,figsizeValues=(6, 6), labelsValues=mlbClasses)
#
# mc.pltPrRe(ytest, yPredProb,epochs,baseFileName)


datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Main End running time: {datestr}, ******************************.")
