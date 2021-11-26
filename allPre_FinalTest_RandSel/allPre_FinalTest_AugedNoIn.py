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
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, KFold, cross_validate, \
    GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
# tf.random.set_seed(1234)
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

pathSavingPlotsAllPre_FinalTest = pathMainCodes + "/reports/forAllPre_FinalTest/"

shiftedNumber = 5

1  #### actualRaw_data-augmentation dataset has all repetetive breaks(1 as labeled), Here create DS with having just one ActualRaw row as break, Then scaled data and drop some unnecessary columns.

# Here, the raw data is being loaded
dfpRaw1 = pd.read_csv(pathDataAllPre_FinalTest + "actualRaw_data-augmentation.csv", header=None)

# Not wanted data has been dropped
dfpRaw2 = dfpRaw1.drop([0, 29, 62], axis=1)
dfpRaw2.columns = [i for i in range(dfpRaw2.shape[1])]
dfpRaw2 = dfpRaw2.drop([0], axis=0)
dfpRaw2 = dfpRaw2.reset_index(drop=True)  # inplace=True,

# Covert from Pandas to Numpy array
dfpRaw2 = dfpRaw2.values

# All data will be converted to float64
dfpRaw2 = np.asarray(dfpRaw2).astype('float64')

# It is needed to repetetive "1" after breaks be deleted
flagFirstOne = False
li = []
for i in range(dfpRaw2.shape[0]):
    if dfpRaw2[i, 0] == 1 and flagFirstOne == False:
        flagFirstOne = True
        continue
    if flagFirstOne == True and dfpRaw2[i, 0] == 1:
        li.append(i)
    if dfpRaw2[i, 0] == 0:
        flagFirstOne = False

dfpRaw2 = np.delete(dfpRaw2, li, axis=0)

# Applying minmax scaler to all columns except the label
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
x_scaled = min_max_scaler.fit_transform(dfpRaw2[:, 1:])
dfpRaw2Scaled = np.concatenate((dfpRaw2[:, 0].reshape(-1, 1), x_scaled), axis=1)



# Two rows after the break will be deleted as they are some noises
indexLblOnes = np.where(dfpRaw2Scaled[:, 0] == True)
l = []
for i in range(1, 3):
    for j in indexLblOnes:
        l.append(j + i)
l1 = np.array(l)
l2 = l1.reshape(l1.shape[0] * l1.shape[1], ).tolist()
dfpRaw3 = np.delete(dfpRaw2Scaled, l2, axis=0)


# dropped breaks that distance with previous breaks is less than Five
il = [0, 0]
ones = [[0]]
ll = []

il = [0, 0]
ones = [[0]]
ll = []

for i in range(dfpRaw3.shape[0]):
    if dfpRaw3[i, 0] == 1:
        il.append(i)
        if i - il[-2] <= 5:

            mm = np.max(np.where(dfpRaw3[:i, 0] == 1))
            ones.append(np.arange(i, mm, -1).tolist())

for i in range(1, len(ones)):
    for j in range(len(ones[i])):
        ll.append(ones[i][j])

ll.sort()
dfpRaw4 = np.delete(dfpRaw3, ll, axis=0)

datestr = time.strftime("%y%m%d_%H%M%S")


# start curve_shift, Shift data corresponds to lookback length
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

#Call Shifting function then shift data corresponds to lookback length
shiftedNumber = 2
dfpRaw5 = pd.DataFrame(dfpRaw4)
dfpRaw5.rename(columns={0: 'label'}, inplace=True)
dfpShifted5NoLess5 = curve_shift(dfpRaw5, shift_by=-1 * shiftedNumber)
dfpShifted5NoLess5 = dfpShifted5NoLess5.astype({"label": int})
dfpShifted5NoLess5.rename(columns={'label': 0}, inplace=True)
dfpShifted5NoLess5 = dfpShifted5NoLess5.reset_index(drop=True)

dfpShifted5NoLess5 = dfpShifted5NoLess5.values
datestr = time.strftime("%y%m%d_%H%M%S")
"""
Now we have Data without breaks but  labeled the 5 rows before each break to 1
"""



###Select first ones in order to later set their label to be label 2
firstOnes=[]
i=0
while i in range(dfpShifted5NoLess5.shape[0]):
    if i==dfpShifted5NoLess5.shape[0]-1:
        break
    if dfpShifted5NoLess5[i,0] == 0:
        i+=1
    if dfpShifted5NoLess5[i,0] == 1:
        firstOnes.append(i)
        #i += 5
        i += 2


#### GenerateRol data
def generateRol(X, y, lookback):
    output_X = []
    output_y = []

    for i in range(-2, len(X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            t.append(X[i + j + 1, :])

        lookback2 = 1
        i += 2

        if (i + lookback2) in firstOnes:  # label2 done!!!
            output_X.append(t)
            output_y.append(2)  # label2
            # print(f"i as label 2: {i}")

        if (i + lookback2) not in firstOnes and y[i + lookback2] == 1:
            output_X.append(t)
            output_y.append(y[i + lookback2])

        if (i + lookback2) not in firstOnes and y[i + lookback2] == 0:
            if y[i]!=1:
                output_X.append(t)
                output_y.append(y[i + lookback2])


    y1 = np.repeat(np.array(output_y), 2)
    y2 = y1.reshape(y1.shape[0], 1)
    X = np.array(output_X).reshape(len(output_X) * 2, 59)
    yX = np.concatenate((y2, X), axis=1)
    return yX


input_X = dfpShifted5NoLess5[:, 1:]
input_y = dfpShifted5NoLess5[:, 0]
lookback = 2
#### Call generateRol data
yX1 = generateRol(X=input_X, y=input_y, lookback=lookback)


5  ##Transform 5 rows to 1 row
###RanSel Part
y1 = yX1[:, 0]
X1 = yX1[:, 1:]

y11 = y1.reshape(int(y1.shape[0] / 2), 2)
y111 = y11[:, 0].reshape(y11[:, 0].shape[0], 1)
X11 = X1.reshape(int(X1.shape[0] / 2), X1.shape[1] * 2)
yX2 = np.concatenate((y111, X11), axis=1)
yXfinal = yX2


firstTwos = []
i = 0
for i in range(yXfinal.shape[0]):
    if yXfinal[i, 0] == 2:
        firstTwos.append(i)



#### Randomly select amount of Twos after rolling and after 5To1 and Augmentation step
#firstTwosRandom = random.sample(firstTwos, 20)


#noneSeenData= yXfinal[firstTwosRandom, :]
data = yXfinal#[np.delete(np.arange(yXfinal.shape[0]), firstTwosRandom), :]



6  ###******************************Final model test
from mainClassFile_NoRanSel_RanSel import mainClass
import preprocessRolLbl2_CV as aug

mc = mainClass()

baseFileName = os.path.basename(__file__).replace(".py", '')




############# Start Running codes
dataSplitValTestPCT = .25
train_test_split_Shuffle = True
flagFitShuffle = True



###***Generate synthetic data with NN
augedTwos = aug.GenerateAug_NN_Rolling(data, jitterNum4Lbl2=2)


data_Auged = np.concatenate((data, augedTwos), axis=0)



yXtrain, yXvalid = train_test_split(data_Auged, shuffle=train_test_split_Shuffle,
                                      test_size=dataSplitValTestPCT, #random_state=42,
                                      stratify=data_Auged[:, 0])



yXtest =data#noneSeenData#np.concatenate((noneSeenData,data),axis=0)



ytrain = to_categorical(yXtrain[:, 0])
yvalid = to_categorical(yXvalid[:, 0])
ytest = to_categorical(yXtest[:, 0])

xtrain = yXtrain[:, 1:]
xvalid = yXvalid[:, 1:]
xtest = yXtest[:, 1:]

pd.DataFrame(xtrain).to_csv('xtrain.csv')
pd.DataFrame(xvalid).to_csv('xvalid.csv')
pd.DataFrame(xtest).to_csv('xtest.csv')

###****Hyperparameters of final model
epochs = 30#5800#800#00
batch = 64
lr = 0.0001


#flagR1 = True
flagR1=False
r1 = .1
r2 = .005
d1 = .2
from tensorflow.keras.regularizers import l1, l2, l1_l2

model1 = Sequential()
model1.add(Dense(240, activation='tanh', input_dim=xtrain.shape[1]))
model1.add(Dense(120, activation='tanh',
                  kernel_regularizer=l1(r1) if flagR1 else l2(r2)
                 ))
model1.add(Dense(60, activation='tanh',
                 kernel_regularizer=l1(r1) if flagR1 else l2(r2)
                 ))
model1.add(Dense(3, activation='softmax'))

adam = optimizers.Adam(lr)


model1.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

print("\n Final test model.summary(): \n")
print(model1.summary())
print(f"\n model.get_config: {str(model1.get_config())} \n")

# fit model
history1 = model1.fit(xtrain, ytrain, batch_size=batch, epochs=epochs
                     , validation_data=(xvalid, yvalid)
                     , verbose=1, use_multiprocessing=True,
                     shuffle=flagFitShuffle).history



mc.pltLossVal(history1['loss'], history1['val_loss'], epochs, ylim=.5)

#pd.DataFrame(noneSeenData).to_csv('noneSeenData.csv')

# yXtestAug=aug.GenerateAug_NN_Rolling(noneSeenData, jitterNum4Lbl2=2)
# pd.DataFrame(yXtestAug).to_csv('yXtestAug.csv')
# XtestAug=yXtestAug[:,1:]
# ytestAug=yXtestAug[:,0]
#
# ytestAug = to_categorical(yXtestAug[:, 0])

yPred = model1.predict(xtest, verbose=1)
#yPredAug=model1.predict(XtestAug)



yt=ytest.argmax(axis=1)
yp=yPred.argmax(axis=1)
yy=np.ones(ytest.shape[0])
ss=[]
for i in range(ytest.shape[0]):
    if yp[i]==2:
        ss.append(i)

print(f"ss: {ss}")



# print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
#
# yt=ytest.argmax(axis=1)
# yp=yPredAug.argmax(axis=1)
# yy=np.ones(ytest.shape[0])
# for i in range(ytestAug.shape[0]):
#     if yp[i]==2:
#         ss.append(i)
#
# print(f"ss: {ss}")



mlbClasses = [0, 1, 2]

print(f"classification_report:\n {classification_report(ytest.argmax(axis=1), yPred.argmax(axis=1))} \n")

print(f"confusion_matrix:\n {confusion_matrix(ytest.argmax(axis=1), yPred.argmax(axis=1))} \n")

mlbConfusion = multilabel_confusion_matrix(ytest.argmax(axis=1), yPred.argmax(axis=1),labels=mlbClasses)
print(f"multilabel_confusion_matrix:\n {mlbConfusion} \n")



# print('AAAAAAAAAAAAAAAAAAAAAAA')
# print(f"classification_report:\n {classification_report(ytestAug.argmax(axis=1), yPredAug.argmax(axis=1))} \n")
#
# print(f"confusion_matrix:\n {confusion_matrix(ytestAug.argmax(axis=1), yPredAug.argmax(axis=1))} \n")
#
# mlbConfusion = multilabel_confusion_matrix(ytestAug.argmax(axis=1), yPredAug.argmax(axis=1))
# print(f"multilabel_confusion_matrix:\n {mlbConfusion} \n")



#LABELS = ['Normal 0', 'Anomalous 1', 'Anomalous 5 step ahead']
mlbClasses = [0, 1, 2]

mc.pltConfMatrixMlbls(mlbConfusion,mlbClasses,figsizeValues=(14, 8))




datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Main End running time: {datestr}, ******************************.")
