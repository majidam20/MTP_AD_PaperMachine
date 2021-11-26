import os
import sys
import gc
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, KFold, cross_validate, \
    GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils import class_weight
from sklearn.feature_selection import chi2,SelectPercentile,f_classif,SelectKBest,mutual_info_classif
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import permutation_importance
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

from pathlib import Path

pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')

pathDataAllPre_FinalTest = pathMainCodes + "/data/paperMachine/forAllPre_FinalTest/"

pathSavingPlotsAllPre_FinalTest = pathMainCodes + "/reports/forAllPre_FinalTest/"



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



# Two rows after the break will be deleted as they are some noises
indexLblOnes = np.where(dfpRaw2[:, 0] == True)
l = []
for i in range(1, 3):
    for j in indexLblOnes:
        l.append(j + i)
l1 = np.array(l)
l2 = l1.reshape(l1.shape[0] * l1.shape[1], ).tolist()
dfpRaw3 = np.delete(dfpRaw2, l2, axis=0)


# Applying minmax scaler to all columns except the label
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
x_scaled = min_max_scaler.fit_transform(dfpRaw3[:, 1:])
dfpRaw3Scaled = np.concatenate((dfpRaw3[:, 0].reshape(-1, 1), x_scaled), axis=1)



il = [0, 0]
ones = [[0]]
ll = []
for i in range(dfpRaw3Scaled.shape[0]):
    if dfpRaw3Scaled[i, 0] == 1:
        il.append(i)
        if i - il[-2] <= 5:

            mm = np.max(np.where(dfpRaw3Scaled[:i, 0] == 1))
            ones.append(np.arange(i, mm, -1).tolist())

for i in range(1, len(ones)):
    for j in range(len(ones[i])):
        ll.append(ones[i][j])

ll.sort()
dfpRaw3NoLess = np.delete(dfpRaw3Scaled, ll, axis=0)


8###Proof for removing less than windows rows between breaks
# l1=[]
# i=0
# while i in range(dfpRaw4.shape[0]):
#
#     if i==dfpRaw4.shape[0]-6:
#         break
#     if dfpRaw4[i,0] == 1:
#         if np.argmax(dfpRaw4[i+1:i+6,0])!=0:
#             b=np.arange(i+1,i+6)[::-1]
#             c=len(b) - np.argmax(b) - 1
#             l1.append(np.arange(i+1,c))
#             i+=1
#     i += 1
#np.argmax([0,0,0,1])!=0
#len(b) - np.argmax(b) - 1
#b=[::-1]




###***Shift labels one to up, so here just upper shifted row get label one, and remove Actual breaks





firstOnes=[]
shiftedOnes=[]
shiftedNumber=3
for i in range(dfpRaw3NoLess.shape[0]):
    if i==dfpRaw3NoLess.shape[0]-1:
        break
    if dfpRaw3NoLess[i,0] == 1:
        firstOnes.append(i)
        shiftedOnes.append(i-shiftedNumber)

dfpRaw3NoLess[shiftedOnes,0]=1
dfpRaw3Shifted = np.delete(dfpRaw3NoLess,firstOnes, axis=0)


"""
Now we have Data without breaks but  labeled the 5 rows before each break to 1
"""

datestr = time.strftime("%y%m%d_%H%M%S")


#### GenerateRol data
def generateRol(X, y, lookback):
    output_X = []
    output_y = []

    for i in range(-2, len(X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            t.append(X[i + j + 1, :])

        lookback2 = lookback-1
        i += 2

        output_X.append(t)
        output_y.append(y[i + lookback2])


    y1 = np.repeat(np.array(output_y), lookback)
    y2 = y1.reshape(y1.shape[0], 1)
    X = np.array(output_X).reshape(len(output_X) * lookback, 59)
    yX = np.concatenate((y2, X), axis=1)

    return yX



input_X = dfpRaw3Shifted[:, 1:]
input_y = dfpRaw3Shifted[:, 0]
winLen = 5
#### Call generateRol data
yXRolled = generateRol(X=input_X, y=input_y, lookback=winLen)





#### GenerateRol data
def generateMean(X, y, lookback):
    yXMean = np.empty([0, yXRolled.shape[1] - 1])
    output_X = []
    output_y = []
    i=-2
    while i < len(X) - lookback - 1:
        t = []
        for j in range(1, lookback + 1):
            t.append(X[i + j + 1, :])

        lookback2 = lookback-1
        x = i + 2

        output_X.append(t)
        tempMean=np.concatenate(output_X, axis=0).mean(axis=0)

        yXMean = np.append(yXMean,tempMean.reshape(1,tempMean.shape[0]) , axis=0)
        output_y.append(y[x + lookback2])

        output_X = []

        i+=5
    #y1 = np.repeat(np.array(output_y), lookback)
    y2=np.array(output_y).reshape(len(output_y), 1)
    #y2 = output_y.reshape(output_y.shape[0], 1)
    #X = yXMean.reshape(len(output_X) * lookback, 59)
    yX = np.concatenate((y2, yXMean), axis=1)

    return yX



input_X = yXRolled[:, 1:]
input_y = yXRolled[:, 0]
winLen = 5
#### Call generateMean data
yXMeanFinal = generateMean(X=input_X, y=input_y, lookback=winLen)

yXMeanFinal=pd.DataFrame(yXMeanFinal)
df=yXMeanFinal

l=[]
temp=[]
selFeat=0
for i in range(2):

    forest = RandomForestClassifier(n_estimators=800,criterion="entropy",verbose=1 ,n_jobs=-1,random_state=42)#800
    forest.fit(df.iloc[:,1:], df.iloc[:,0])

    permRF = permutation_importance(forest, df.iloc[:,1:], df.iloc[:,0],n_jobs=-1,n_repeats=20,random_state=42)#20

    sorted_idx = permRF.importances_mean.argsort()[::-1]

    print(f"permRF.importances_mean_{i}:\n {pd.DataFrame(np.round(permRF.importances_mean,6))}\n")

    if i==0:
        threshold = .0001
        ###because in first iter selected df columns number is passed
        selFeat = [x+1 for x in np.where(permRF.importances_mean>=threshold)[0].tolist()]
        print(f"NumberOf_SelFeatIter_{i}: {len(selFeat)}")
        print(f"selFeatIter_{i}: {selFeat}")

        l.append(selFeat)
        df = df.loc[:, [0] + selFeat]

    if i==1:
        threshold = .001
        ###because in second iter index selected df columns number is passed
        l.append(np.where(permRF.importances_mean>=threshold)[0].tolist())
        selFeat=[l[0][index] for index in l[1]]

        print(f"NumberOf_FinalSelFeatIter_{i}: {len(selFeat)}")
        print(f"FinalSelFeatIter_{i}: {selFeat}")
        #df = df.loc[:, [0] + selFeat]

    del forest
    del permRF

sys.exit()

X=df[:,1:]
y=df[:,0]



5#Peyman mean
# yX1new=np.empty((dfpRaw3Shifted.shape[0],60))
#
# i=0
# k=0
# temp=np.empty((5,59))
# while i <yX1new.shape[0]-5:
#     c=i
#     for j in range(5):
#         temp[j,:]=yX1[c,1:]
#         c+=1
#     yX1new[k, 0] = yX1[i, 0]
#     i+=5
#     bb=temp.mean(axis=0)
#     yX1new[k,1:]=bb
#     k+=1











# y = y.reshape(y.shape[0], 1)
# yX1=np.concatenate((y,X),axis=1)

5##Transform 5 rows to 1 row
y1 = yXMeanFinal[:, 0]
X1 = yXMeanFinal[:, 1:]
X11 = X1.reshape(int(X1.shape[0] / winLen), X1.shape[1] * winLen)

y11 = y1.reshape(int(y1.shape[0] / winLen), winLen)
y111 = y11[:, 0].reshape(y11[:, 0].shape[0], 1)

yXTransformed = np.concatenate((y111, X11), axis=1)







X=df[:,fs.index.tolist()]
y=df[:,0]




ones=[]
for i in range(yXTransformed.shape[0]):
    if i==yXTransformed.shape[0]-1:
        break
    if yXTransformed[i,0] == 1:
        ones.append(i)


#### Randomly select amount of Twos after rolling and after 5To1 and Augmentation step
onesRandom = random.sample(ones, 20)


noneSeenData= yXTransformed[onesRandom, :]
data = yXTransformed[np.delete(np.arange(yXTransformed.shape[0]), onesRandom), :]



6  ###******************************Final model test
from mainClassFile_NoRanSel_RanSel import mainClass
import generateAugmentedData_FeatureSel as aug

mc = mainClass()

baseFileName = os.path.basename(__file__).replace(".py", '')




############# Start Running codes
dataSplitValTestPCT = .25
train_test_split_Shuffle = True
flagFitShuffle = True

#def generateAugAE(yX,makeNoiseByJitter,foldNum,model=0,jitterNum4Lbl2=20):

makeNoiseByJitter=False
###***Generate synthetic data with NN
augedTwos = aug.generateAugAE(data,makeNoiseByJitter, jitterNum4Lbl2=4)


data_Auged = np.concatenate((data, augedTwos), axis=0)



yXtrain, yXvalid = train_test_split(data_Auged, shuffle=train_test_split_Shuffle,
                                      test_size=dataSplitValTestPCT, #random_state=42,
                                      stratify=data_Auged[:, 0])



yXtest =noneSeenData#noneSeenData#np.concatenate((noneSeenData,data),axis=0)



# ytrain = to_categorical(yXtrain[:, 0])
# yvalid = to_categorical(yXvalid[:, 0])
# ytest = to_categorical(yXtest[:, 0])


ytrain = yXtrain[:, 0]
yvalid = yXvalid[:, 0]
ytest  = yXtest[:, 0]

xtrain = yXtrain[:, 1:]
xvalid = yXvalid[:, 1:]
xtest = yXtest[:, 1:]


###****Hyperparameters of final model
epochs = 50#5800#800#00
batch = 32
lr = 0.001


#flagR1 = True
flagR1=False
r1 = .1
r2 = .005
d1 = .2
from tensorflow.keras.regularizers import l1, l2, l1_l2

model1 = Sequential()
model1.add(Dense(590, activation='tanh', input_dim=xtrain.shape[1]))
model1.add(Dense(500, activation='tanh',
                  kernel_regularizer=l1(r1) if flagR1 else l2(r2)
                 ))
model1.add(Dense(400, activation='tanh',
                 kernel_regularizer=l1(r1) if flagR1 else l2(r2)
                 ))
model1.add(Dense(1, activation='sigmoid'))

adam = optimizers.Adam(lr)


model1.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

print("\n Final test model.summary(): \n")
print(model1.summary())
print(f"\n model.get_config: {str(model1.get_config())} \n")

# fit model
history1 = model1.fit(xtrain, ytrain, batch_size=batch, epochs=epochs
                     , validation_data=(xvalid, yvalid)
                     , verbose=1, use_multiprocessing=True,
                     shuffle=flagFitShuffle).history



#mc.pltLossVal(history1['loss'], history1['val_loss'], epochs, ylim=.5)


yPred = model1.predict(xtest, verbose=1)


print(f"yPred{yPred}")

l = []
for i in yPred:
    if i < .5:
        l.append(0)
    else:
        l.append(1)

yPred = l


labelsValues = [0, 1]

print(f"classification_report:\n {classification_report(ytest, yPred)} \n")

print(f"confusion_matrix:\n {confusion_matrix(ytest, yPred)} \n")

tn, fp, fn, tp = confusion_matrix(ytest, yPred, labels=labelsValues).ravel()

print("True Negatives: ", tn)
print("False Positives: ", fp)
print("False Negatives: ", fn)
print("True Positives: ", tp, "\n")


datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Main End running time: {datestr}, ******************************.")
