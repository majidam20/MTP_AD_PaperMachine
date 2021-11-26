###*** Master Thesis Project Anomaly detection and forecasting in time series sensor data for Paper Machine

###*** Author Majid Aminian, Department of Data Science in Beuth Hochschule Berlin


### Import Libraries
import os
import sys
import gc
import random
import time
import pandas as pd
import numpy as np
from pathlib import Path
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils import class_weight
from sklearn.feature_selection import chi2,SelectPercentile,f_classif,SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier
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
from tensorflow.keras.callbacks import ModelCheckpoint
###**** End tensorflow.keras
import warnings
warnings.filterwarnings("ignore")
####End of importing libraries


###Some different possibilities are achieved by setting below variables
"""
search
selectModelType=
saveModelFinalTestEvaluation
saveModelAug
addAug=
numberOfAug
epochs=
flagPos_weight
pos_weight
thresholdNum
doScaleAfterSplit
flagR1
makeNoiseByJitter
doGenerateAugDTW
"""



### mainClassFile_Evaluation consists of all preprocessing and plots
from mainClassFile_Evaluation import mainClass_Evaluation



### generateAugmentedData_CV consists of codes to generate synthetic data(augmented data) by the Autoencoder model.
import generateAugmentedData_Evaluation as aug




### Define main path addresses of project
pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')


pathDataAllPre_FinalTest = pathMainCodes + "/data/paperMachine/forAllPre_FinalTest/"
pathSavingPlotsAllPre_FinalTest_Eval = pathMainCodes + "/reports/forAllPre_FinalTest_Evaluation/"

pathSavingModelFinalTest=pathCurrrent + "/savedModelFinalTest"
pathSavingModelAE=pathCurrrent + "/savedModelAE"

baseFileName=os.path.basename(__file__).replace(".py", '')


############################****** Start Running codes ******##################################

datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Main Start running time: {datestr},******************************:")
print(f"\nName of corresponding python code file: {baseFileName} \n")



### Define variable mc that is an object of mainClass_Evaluation
mc = mainClass_Evaluation()


### Read Actual raw dataset(DS)
dfActual = pd.read_csv(pathDataAllPre_FinalTest + "actualRaw_data-augmentation.csv", header=None)### when documentation finished uncomment here



doScaleAfterSplit=True


###Under the first scenario, we fit and transform the scaler estimator purposely on a whole cleaned dataset
if doScaleAfterSplit == False:

    ###Create dataset with having just one ActualRaw row as break label 1, drop a few unnecessary columns, And Delete RepetitiveOnes(breaks)
    dfpRawNoReptOnes = mc.delRepetitiveOnes(dfActual)


    ###Secario first for scaling dataset
    dfpRawNoReptOnes_scaled = mc.scaleData(dfpRawNoReptOnes, doScaleAfterSplit=False)



    ### Split dataset dfpRawNoReptOnes_scaled, that scaled and deleted RepetitiveOnes(breaks)
    ### npTrain is used for Train part and npFinalTest that will be used here for final test evaluation
    finalTestPercentage=0.2
    npTrain, npFinalTest = train_test_split(dfpRawNoReptOnes_scaled, test_size=finalTestPercentage, random_state=42,shuffle=False)

    print(f"\n doScaleAfterSplit= {doScaleAfterSplit} approach. \n")






##Second scenario, we split the actual cleaned data set to train set and test set and, then, we fit and transform the scaler on the train set and apply the scaler transform function on the test set
if doScaleAfterSplit == True:

    dfpRawNoReptOnes = mc.delRepetitiveOnes(dfActual)


    finalTestPercentage = 0.2
    npTrain, finalTest = train_test_split(dfpRawNoReptOnes, test_size=finalTestPercentage, random_state=42, shuffle=False)


    npTrain, npFinalTest = mc.scaleData(npTrain, finalTest, doScaleAfterSplit=True)

    print(f"\n doScaleAfterSplit= {doScaleAfterSplit} approach. \n")





###Define user-predefined shift Number and window Length
shiftNumber=2
winLen = 2
saveModelFinalTestEvaluation=False



###makeNoiseByJitter will be used if you want to generated synthetic data by making noise on selected label ones
makeNoiseByJitter=False



###Set doGenerateAugDTW=True, If we want to generate Synthetic data with DTW Approach
doGenerateAugDTW= False



###Define variable pos_weight that determines a decimal multiplication weight values on positive labels(Anomalies as labelled 1)
pos_weight=0
flagPos_weight=False



###Define variable thresholdNum that determines a decimal Threshold value for classifying probabilities that will be predicted by the classifier
thresholdNum=0



print(f"\nshiftNumber: {shiftNumber}, windowLength: {winLen}")
print(f"makeNoiseByJitter: {makeNoiseByJitter}")




### Calculate proportion of Anomalies in the whole dataset
num_neg, num_pos = np.bincount(npTrain[:,0].astype(int))
mc.get_class_weights(num_pos, num_neg,False)



###Define flagSeed to create reproducible results for each run, shuffle boolian variables,  FitShuffle will be used in Model fit function
flagSeed=True
FitShuffle =True

ModelFinalTestEvaluation=0




print(f"\nbaseFileName of output Folders and Plots name: {baseFileName} \n")


print(f"\n   Shape of Actual npTrain dataset: {npTrain.shape}")
print(f"   Shape of Actual npFinalTest dataset: {npFinalTest.shape}\n")

print(f"   Number of Actual npTrain labelled 0: {len(npTrain[np.where(npTrain[:, 0]==0)])}")
print(f"   Number of Actual npTrain labelled 1: {len(npTrain[np.where(npTrain[:, 0]==1)])} \n")

print(f"   Number of Actual npFinalTest labelled 0: {len(npFinalTest[np.where(npFinalTest[:, 0]==0)])}")
print(f"   Number of Actual npFinalTest labelled 1: {len(npFinalTest[np.where(npFinalTest[:, 0]==1)])} \n")




###Select type of classifier, Deep Neural Network(DNN) OR Long short-term memory(LSTM)
selectModelType="DNN" ### "DNN" , "LSTM"




########################################Start Train part(above for loop in CV)


yXtrain, yXfinalTest= npTrain, npFinalTest


datestr = time.strftime("%y%m%d_%H%M%S")
print(f"\n Start running time : {datestr} ,--------------------------: \n")



###Shift label of Actual Anomalies to user-predefined shiftNumber then delete Actual Anomalies to prevent the classifier from learning Actual anomalies; hence classifier learns shifted Anomalies
yXtrainShifted = mc.shiftLblOnes(yXtrain, shiftNumber)
yXfinalTestShifted = mc.shiftLblOnes(yXfinalTest, shiftNumber)



#### Call generateRoll data to makes rolled data by passing window Length value that already assigned by the user
yXtrainRolled = mc.generateRoll(X=yXtrainShifted[:, 1:], y=yXtrainShifted[:, 0], winLength=winLen)
yXfinalTestRolled = mc.generateRoll(X=yXfinalTestShifted[:, 1:], y=yXfinalTestShifted[:, 0], winLength=winLen)




print(f"\n   Shape of Actual yXtrainRolled After Rolling: {yXtrainRolled.shape}")

print(f"\n   Shape of Actual yXfinalTestShifted dataset: {yXfinalTestShifted.shape}")
print(f"   Shape of Actual yXfinalTestRolled After Rolling: {yXfinalTestRolled.shape}")




###***Transform ToOneRow, e.g. 2To1, In order to keep order of window blocks
yXtrainRolled_Transformed = mc.transformShapeData(yXtrainRolled, winLen, toOneRow=True)
yXfinalTestRolled_Transformed = mc.transformShapeData(yXfinalTestRolled, winLen, toOneRow=True)
yXfinalTest = yXfinalTestRolled_Transformed




datestrFolder = time.strftime("%y%m%d_%H%M%S")
print(f"\n Start Creating new folder finalTest run: {datestrFolder}\n")



###Create new folder for finalTest run in order to save results of plots
pathSavingPlotsPerRunning = pathSavingPlotsAllPre_FinalTest_Eval + datestrFolder + "_" + baseFileName
if not os.path.exists(pathSavingPlotsPerRunning):
    os.makedirs(pathSavingPlotsPerRunning)




###Saving model for FinalTestEvaluation
if saveModelFinalTestEvaluation==True:

    pathSavingModelFinalTest = f"{pathSavingModelFinalTest}/{selectModelType}_{datestrFolder}/"
    if not os.path.exists(pathSavingModelFinalTest):
        os.makedirs(pathSavingModelFinalTest)

    # pathSavingModelFinalTest = os.path.join(pathSavingModelFinalTest, "model.hdf5")###when you want to save with your desired file name instead of saving with tensorflow predefined folders and file names







###In the approach that does not add synthetic data, we should assign addAug= False
###Default is True that means use adding augmentation approach
addAug=True



###***Generate synthetic data with DTW Approach, Real numberOfAug is 14 time more that actual, but here is 15 because inside generateAugDTW function loop statement start from [1 to 15)
if doGenerateAugDTW == True:
    generatedAug = aug.generateAugDTW(yXtrainRolled_Transformed, addAug=addAug, numberOfAug=15)






###***Generate synthetic data with AE_NN
if doGenerateAugDTW == False:

    ###***Generate synthetic data with AE_NN, Real numberOfAug is 14 time more that actual, but here is 15 because inside generateAugAE function loop statement start from [1 to 15)
    generatedAug = aug.generateAugAE(yXtrainRolled_Transformed, makeNoiseByJitter, f"{pathSavingModelAE}/AE_{selectModelType}_{datestrFolder}/" ,addAug=addAug, numberOfAug=15,saveModelAug=False)







if addAug ==False:
    generatedAug=yXtrainRolled_Transformed
    print("\n------------*** Train ModelFinalTestEvaluation without adding Augmented data ***------------\n ")###Use this line just for approach without adding Augmented data





Actual_generatedAug = np.concatenate((yXtrainRolled_Transformed, generatedAug), axis=0)###Use this line for approach without adding Augmented data





###Use this line just for approach without adding Augmented data
if addAug == False:
    Actual_generatedAug=yXtrainRolled_Transformed





###Shuffle Actual_generatedAug that consists of  yXtrainRolled_Transformed data and added synthetic data
yXtrain1, yXtrain2 = train_test_split(Actual_generatedAug, shuffle=True,
                                    test_size=.3, random_state=42,stratify=Actual_generatedAug[:, 0])




yXtrain = np.concatenate((yXtrain1, yXtrain2), axis=0)




### numberOfActual_yXtrainLbl1 keeps number of Anomalies before adding augmented data
numberOfActual_yXtrainLbl1 = len(yXtrainRolled_Transformed[np.where(yXtrainRolled_Transformed[:, 0] == 1)])



baseFileName = os.path.basename(__file__).replace(".py", '')



print(f"\n    Shape of data after Transform ToOneRow to give finalTest Model: ")
print(f"\n    xtrain: {np.shape(yXtrain[:,1:])}, ytrain: {np.shape(yXtrain[:,0])}")
print(f"    xfinalTest: {np.shape(yXfinalTest[:,1:])}, yfinalTest: {np.shape(yXfinalTest[:,0])}")



print(f"\n   Number of final label 0 in yXtrain: {len(yXtrain[np.where(yXtrain[:, 0] == 0)])}")

print(f"\n   Number of final label 1 in yXtrain_Before_AddingAugedData: "
      f"{ numberOfActual_yXtrainLbl1}")

print(f"   Number of final label 1 in yXtrain_After_AddingAugedData: "
      f"{len(Actual_generatedAug[np.where(Actual_generatedAug[:, 0] == 1)]) } \n")



ytrain = yXtrain[:, 0]
yfinalTest = yXfinalTest[:, 0]


xtrain = yXtrain[:, 1:]
xfinalTest = yXfinalTest[:, 1:]



###****Define Hyperparameters of ModelFinalTestEvaluation
epochs= 500
batch = 32
lr = 0.0001

#flagR1 = True
flagR1=False
r1 = .1
r2 = .015#.015### comment in print hypers when do not use ridge possibility





###****Hyperparameters of ModelFinalTestEvaluation

print("\n Hyperparameters of finalTest Model:")
print(f" epochs: {epochs}, batch: {batch}, lr: {lr}"
      f", flagSeed: {flagSeed}"
      f", FitShuffle: {FitShuffle}\n ")





ModelFinalTestEvaluation = Sequential()

###When selected classifier is Deep Neural Network(DNN)
if selectModelType=="DNN":

   ModelFinalTestEvaluation = mc.selectModelTypeFunc(xtrain , ModelFinalTestEvaluation, flagR1,r1,r2, selectModelType="DNN")





###When selected classifier is Long short-term memory(LSTM)
if selectModelType=="LSTM":
    xtrain, ytrain, xfinalTest , yfinalTest = mc.transformTo3Dim(xtrain , ytrain.reshape(-1,1) , xfinalTest , yfinalTest.reshape(-1,1))

    ModelFinalTestEvaluation = mc.selectModelTypeFunc(xtrain , ModelFinalTestEvaluation,flagR1,r1,r2,selectModelType="LSTM")





adam = optimizers.Adam(lr)


ModelFinalTestEvaluation.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])





print("\n finalTest Model.summary(): \n")
print(ModelFinalTestEvaluation.summary())

print(f"\n finalTest Model config:\n {str(ModelFinalTestEvaluation.get_config())} \n\n")





### Call get_class_weights To Make balance classes by assigning weights
num_neg, num_pos = np.bincount(ytrain.astype(int))

pos_weight=.9###WARNING:tensorflow:sample_weight
class_weight = mc.get_class_weights(num_pos, num_neg,pos_weight=pos_weight)###Higher pos_weight Number Means enhance proportion to the weight of Positive class






### Fit ModelFinalTestEvaluation, shuffle is by defult True
if saveModelFinalTestEvaluation == True:
    ### Fit modelFinalTest
    chkp = ModelCheckpoint(pathSavingModelFinalTest, save_best_only=True, monitor='accuracy', verbose=2)
    history1 = ModelFinalTestEvaluation.fit(xtrain, ytrain, batch_size=batch, epochs=epochs
                                  , verbose=1, use_multiprocessing=True,
                                  shuffle=FitShuffle , class_weight=class_weight if flagPos_weight== True else None , callbacks=[chkp]).history  # ,class_weight=class_weight





    savedModelFinalTest = tf.keras.models.load_model(pathSavingModelFinalTest)
    savedModelFinalTest.predict(xfinalTest, verbose=1)





if saveModelFinalTestEvaluation == False:
    ### Fit modelFinalTest
    history1 = ModelFinalTestEvaluation.fit(xtrain, ytrain, batch_size=batch, epochs=epochs
                                  , verbose=1, use_multiprocessing=True,
                                  shuffle=FitShuffle ,class_weight=class_weight if flagPos_weight== True else None ).history  # ,class_weight=class_weight





### Predict probabilities of xfinalTest data part
yPredProb = ModelFinalTestEvaluation.predict(xfinalTest, verbose=1)







print(f"\n\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$: \n")
print(f"---*** Average's Results of Model FinalTest Evaluation is such as below: \n")


###Assign variable thresholdNum that determines a decimal Threshold value for classifying probabilities that will be predicted by the classifier
thresholdNum=.5 ###.7,.56,.6,.8, .9
yPredCategorical = mc.lossFunction(yPredProb, thresholdNum=thresholdNum)





mc.pltLossVal(history1['loss'],shiftNumber,winLen, epochs, pathSavingPlotsPerRunning, baseFileName,
               batch,flagSeed, ylim=np.max(np.round(history1['loss'],4)), r2=r2, addAug= addAug, selectModelType= selectModelType)



if flagPos_weight==True:
    print(f"\nclass_weight: {class_weight} \n")
    print(f"\npos_weight: {pos_weight}")

else:
    print(f"\n***Approach class_weight did not use***\n")


print(f"thresholdNum: {thresholdNum}\n")



print(f"\nAverage of train loss: {np.round(np.mean(history1['loss']),4)}")
print(f"Average of yPredProbability: {np.round(np.float(yPredProb.mean()), 4)} \n")






LABELS = ['Normal 0', 'Anomaly 1']

print(f"\nclassification_report FinalTest:")
print(classification_report(yfinalTest.reshape(-1,1), np.array(yPredCategorical).reshape(-1,1), target_names=LABELS) , "\n")



mc.pltPrRe(yfinalTest.reshape(-1,1), yPredProb,shiftNumber,winLen, epochs, pathSavingPlotsPerRunning, baseFileName,batch,flagSeed,r2=r2, addAug= addAug, selectModelType= selectModelType)




mc.printConfMatrix(yfinalTest, yPredCategorical, labelsValues=[0, 1])




mc.pltConfMatrix(yfinalTest, yPredCategorical,shiftNumber,winLen, LABELS, epochs, pathSavingPlotsPerRunning, baseFileName,batch, flagSeed,
                 figsizeValues=(6, 6), labelsValues=[0, 1],r2=r2, addAug= addAug, selectModelType= selectModelType)




mc.printPrReThreshold(yfinalTest, yPredProb, thresholdDistance=4)






1###Comparison distplot of Actual data(left side) and generated synthetic(right side)
# if addAug ==True:
#     yXtrainRolled_TransformedOnes=yXtrainRolled_Transformed[np.where(yXtrainRolled_Transformed[:,0]==1)]
#      mc.pltDfCompSnsDist(yXtrainRolled_Transformed[:, 1:60], generatedAug[:, 1:60], shiftNumber, winLen, pathCurrrent,
#                    baseFileName, flagSeed, w=20, h=200, bins=20, selectModelType=selectModelType)





datestr = time.strftime("%y%m%d_%H%M%S")
print(f"End running time FinalTest Evaluation: {datestr}, ******************************.")











