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
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
from sklearn.model_selection import StratifiedKFold, KFold
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
doFeatureSelection
"""




### mainClassFile_CV consists of all preprocessing and plots
from mainClassFile_CV import mainClass_CV




### generateAugmentedData_CV consists of codes to generate synthetic data(augmented data) by the Autoencoder model.
import generateAugmentedData_CV as aug





### Define main path addresses of project
pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')



pathDataAllPre_FinalTest = pathMainCodes + "/data/paperMachine/forAllPre_FinalTest/"
pathSavingPlotsAllPre_FinalTest_CV = pathMainCodes + "/reports/forAllPre_FinalTest_CV/"


baseFileName=os.path.basename(__file__).replace(".py", '')





############################****** Start Running codes ******##################################

datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Main Start running time: {datestr},******************************:")
print(f"\nName of corresponding python code file: {baseFileName} \n")




###Define Variables that keep results that related to their names in each iteration K-fold cross validation
allFolds_LossEpoch_Train=np.empty([0,1])
allFolds_LossEpoch_plot_Train=[]
allFolds_ytest_yPredProb=np.empty([0,2])
allPrRe = np.empty([0, 3])




### Define variable mc that is an object of mainClass_CV
mc = mainClass_CV(allPrRe)



### Read Actual raw dataset(DS)
dfActual = pd.read_csv(pathDataAllPre_FinalTest + "actualRaw_data-augmentation.csv", header=None)



doScaleAfterSplit=True


###Under the first scenario, we fit and transform the scaler estimator purposely on a whole cleaned dataset
if doScaleAfterSplit == False:

    ###Create dataset with having just one ActualRaw row as break label 1, drop a few unnecessary columns, And Delete RepetitiveOnes(breaks)
    dfpRawNoReptOnes = mc.delRepetitiveOnes(dfActual)


    ###Secario first for scaling dataset
    dfpRawNoReptOnes_scaled = mc.scaleData(dfpRawNoReptOnes, doScaleAfterSplit=False)



    ###Split dataset dfpRawNoReptOnes_scaled, that scaled and deleted RepetitiveOnes(breaks)
    ### npTrain is used for cross validation approach and finalTest later will be used for final test evaluation of whole npTrain
    finalTestPercentage=0.2
    npTrain, finalTest = train_test_split(dfpRawNoReptOnes_scaled, test_size=finalTestPercentage, random_state=42,shuffle=False)

    print(f"\n doScaleAfterSplit= {doScaleAfterSplit} approach. \n")




##Second scenario, we split the actual cleaned data set to train set and test set and, then later, in each CV itaration indivisually we fit and transform the scaler on the train set and apply the scaler transform function on the validation set
if doScaleAfterSplit == True:

    dfpRawNoReptOnes = mc.delRepetitiveOnes(dfActual)


    finalTestPercentage = 0.2
    npTrain, finalTest = train_test_split(dfpRawNoReptOnes, test_size=finalTestPercentage, random_state=42, shuffle=False)

    print(f"\n doScaleAfterSplit= {doScaleAfterSplit} approach. \n")








###Define user-predefined shift Number and window Length
shiftNumber=2
winLen = 2



###makeNoiseByJitter will be used if you want to generated synthetic data by making noise on selected label ones
makeNoiseByJitter= False



###Set doGenerateAugDTW=True, If we want to generate Synthetic data with DTW Approach
doGenerateAugDTW= False



###Define variable pos_weight that determines a decimal multiplication weight's value on positive labels(Anomalies as labelled 1)
pos_weight=0
flagPos_weight=False




###Define variable thresholdNum that determines a decimal Threshold value for classifying probabilities that will be predicted by the classifier
thresholdNum=0




print(f"\nshiftNumber: {shiftNumber}, windowLength: {winLen}")
print(f"makeNoiseByJitter: {makeNoiseByJitter}")




### Calculate proportion of Anomalies in the whole dataset
num_neg, num_pos = np.bincount(npTrain[:,0].astype(int))
mc.get_class_weights(num_pos, num_neg,False)





###Define flagSeed to create reproducible results for each run, shuffle boolian variables, train_val_split_Shuffle will be used in KFold split function, FitShuffle will be used in Model fit function
flagSeed=True
train_val_split_Shuffle=False
FitShuffle =True

numberOfSplits=5
modelEvaluateVal=0




###Define Kfold cross validation variable with five folds splits
skf = KFold(n_splits=numberOfSplits,shuffle=train_val_split_Shuffle)





print(f"\nbaseFileName of output Folders and Plots name: {baseFileName} \n")


print(f"\n   Shape of Actual npTrain dataset: {npTrain.shape}")
print(f"   Number of Actual npTrain labelled 0: {len(npTrain[np.where(npTrain[:, 0]==0)])}")
print(f"   Number of Actual npTrain labelled 1: {len(npTrain[np.where(npTrain[:, 0]==1)])} \n")





###Select type of classifier, Deep Neural Network(DNN) OR Long short-term memory(LSTM)
selectModelType="DNN" ### "DNN" , "LSTM"




#### Call Features selection approach, with permutation_importance(RandomForestClassifier)
doFeatureSelection=False

if doFeatureSelection==True:


    yXtrainShifted = mc.shiftLblOnes(npTrain, shiftNumber)



    yXtrainRolled = mc.generateRoll(X=yXtrainShifted[:, 1:], y=yXtrainShifted[:, 0], winLength=winLen)



    print(f"\n   Shape of Actual yXtrainRolled After Rolling and Before Feature Selection: {yXtrainRolled.shape}")

    selectedFeatures=mc.selectFeatures(yXtrainRolled[:,1:], yXtrainRolled[:,0], winLen)
    npTrain= yXtrainRolled[:,selectedFeatures]

    print(f"\n   Shape of Actual yXtrainRolled_selectedFeatures After Feature Selection: {npTrain.shape}")







###Start validation variable loop, Here npTrain part will be used that already was separated from finalTest part
###After splitting train and validation then required preprocessing approaches will be performed on them separately

for foldNum, (trainIndex, valIndex) in enumerate(skf.split(npTrain[:, 1:],npTrain[:, 0]),start=1):

    yXtrain, yXvalid= npTrain[trainIndex], npTrain[valIndex]

    if doScaleAfterSplit == True:
        yXtrain, yXvalid = mc.scaleData(yXtrain, yXvalid, doScaleAfterSplit=True)


    datestrfoldNum = time.strftime("%y%m%d_%H%M%S")
    print(f"\n Start running time Fold_{foldNum}: {datestrfoldNum} ,--------------------------: \n")




    ###Shift label of Actual Anomalies to user-predefined shiftNumber then delete Actual Anomalies to prevent the classifier from learning Actual anomalies; hence classifier learns shifted Anomalies
    yXtrainShifted = mc.shiftLblOnes(yXtrain, shiftNumber)
    yXvalidShifted = mc.shiftLblOnes(yXvalid, shiftNumber)





    #### Call generateRoll data to makes rolled data by passing window Length value that already assigned by the user
    yXtrainRolled = mc.generateRoll(X=yXtrainShifted[:, 1:], y=yXtrainShifted[:, 0], winLength=winLen)
    yXvalidRolled = mc.generateRoll(X=yXvalidShifted[:, 1:], y=yXvalidShifted[:, 0], winLength=winLen)





    print(f"\n   Shape of Actual yXtrainRolled After Rolling Fold_{foldNum}: {yXtrainRolled.shape}")

    print(f"\n   Shape of Actual yXvalidShifted dataset: {yXvalidShifted.shape}")
    print(f"   Shape of Actual yXvalidRolled After Rolling Fold_{foldNum}: {yXvalidRolled.shape}")





    ###***Transform ToOneRow, e.g. 2To1, In order to keep order of window blocks
    yXtrainRolled_Transformed = mc.transformShapeData(yXtrainRolled, winLen, toOneRow=True)
    yXvalidRolled_Transformed = mc.transformShapeData(yXvalidRolled, winLen, toOneRow=True)
    yXvalid = yXvalidRolled_Transformed




    ###In the approach that does not add synthetic data, we should assign addAug=False
    ###Default is True that means use adding augmentation approach
    addAug=True



    ###***Generate synthetic data with DTW, Real numberOfAug is 14 time more that actual, but here is 15 because inside generateAugDTW function loop statement start from [1 to 15)
    if doGenerateAugDTW== True:
        generatedAug= aug.generateAugDTW(yXtrainRolled_Transformed,  foldNum, addAug=addAug , numberOfAug=15)




    ###***Generate synthetic data with AE_NN, Real numberOfAug is 14 time more that actual, but here is 15 because inside generateAugAE function loop statement start from [1 to 15)
    if doGenerateAugDTW == False:
        generatedAug = aug.generateAugAE(yXtrainRolled_Transformed, makeNoiseByJitter, foldNum,addAug=addAug, numberOfAug=15)




    if addAug==False:
        generatedAug=yXtrainRolled_Transformed
        print("\n------------*** Train modelEvaluateVal without adding Augmented data ***------------\n ")###Use this line just for approach without adding Augmented data




    ###Use this line for approach without adding Augmented data
    Actual_generatedAug = np.concatenate((yXtrainRolled_Transformed, generatedAug), axis=0)




    if addAug == False:
        Actual_generatedAug=yXtrainRolled_Transformed###Use this line just for approach without adding Augmented data





    ###Shuffle Actual_generatedAug that consists of  yXtrainRolled_Transformed data and added synthetic data
    yXtrain1, yXtrain2 = train_test_split(Actual_generatedAug, shuffle=True,
                                        test_size=.3, random_state=42,stratify=Actual_generatedAug[:, 0])



    yXtrain = np.concatenate((yXtrain1, yXtrain2), axis=0)




    ### numberOfActual_yXtrainLbl1 keeps number of Anomalies before adding augmented data
    numberOfActual_yXtrainLbl1 = len(yXtrainRolled_Transformed[np.where(yXtrainRolled_Transformed[:, 0] == 1)])



    baseFileName = os.path.basename(__file__).replace(".py", '')



    print(f"\n    Shape of data after Transform ToOneRow to give Cross Validation Model in fold_{foldNum}: ")
    print(f"\n    xtrain: {np.shape(yXtrain[:,1:])}, ytrain: {np.shape(yXtrain[:,0])}")
    print(f"    xvalid: {np.shape(yXvalid[:,1:])}, yvalid: {np.shape(yXvalid[:,0])}")




    print(f"\n   Number of final label 0 in yXtrain_Fold_{foldNum}: {len(yXtrain[np.where(yXtrain[:, 0] == 0)])}")

    print(f"\n   Number of final label 1 in yXtrain_Before_AddingAugedData_Fold_{foldNum}: "
          f"{ numberOfActual_yXtrainLbl1}")

    print(f"   Number of final label 1 in yXtrain_After_AddingAugedData_Fold_{foldNum}: "
          f"{len(Actual_generatedAug[np.where(Actual_generatedAug[:, 0] == 1)]) } \n")





    ytrain = yXtrain[:, 0]
    yvalid = yXvalid[:, 0]


    xtrain = yXtrain[:, 1:]
    xvalid = yXvalid[:, 1:]




    ###****Define Hyperparameters of modelEvaluateVal
    epochs= 500
    batch = 32
    lr = 0.0001

    #flagR1 = True
    flagR1=False
    r1 = .1
    r2 = .015#.015### comment in print hypers when do not use ridge possibility






    ###****Hyperparameters of modelEvaluateVal
    if foldNum==1:
        print("\n Hyperparameters of Cross Validation Model:")
        print(f" epochs: {epochs}, batch: {batch}, lr: {lr}"
              f", numberOfSplits:{numberOfSplits}, flagSeed: {flagSeed}"
              f", train_val_split_Shuffle: {train_val_split_Shuffle}, FitShuffle: {FitShuffle}\n ")



        datestrFolder = time.strftime("%y%m%d_%H%M%S")
        print(f"\n Start Creating new folder for each run: {datestrFolder}\n")




        ###Create new folder for each run in order to save results of plots
        pathSavingPlotsPerRunning = pathSavingPlotsAllPre_FinalTest_CV + datestrFolder + "_" + baseFileName
        if not os.path.exists(pathSavingPlotsPerRunning):
            os.makedirs(pathSavingPlotsPerRunning)





    modelEvaluateVal = Sequential()



    ###When selected classifier is Deep Neural Network(DNN)
    if selectModelType=="DNN":

       modelEvaluateVal = mc.selectModelTypeFunc(xtrain, modelEvaluateVal, flagR1, r1, r2, selectModelType="DNN")





    ###When selected classifier is Long short-term memory(LSTM)
    if selectModelType=="LSTM":


        xtrain, ytrain , xvalid, yvalid = mc.transformTo3Dim(xtrain, ytrain.reshape(-1, 1), xvalid, yvalid.reshape(-1, 1))


        modelEvaluateVal = mc.selectModelTypeFunc(xtrain, modelEvaluateVal, flagR1, r1, r2, selectModelType="LSTM")



    adam = optimizers.Adam(lr)


    modelEvaluateVal.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])




    if foldNum == 1:
        print("\n Cross Validation Model.summary(): \n")
        print(modelEvaluateVal.summary())

        print(f"\n Cross Validation Model config:\n {str(modelEvaluateVal.get_config())} \n\n")





    ### Call get_class_weights To Make balance classes by assigning weights
    num_neg, num_pos = np.bincount(ytrain.astype(int))

    pos_weight= .9###WARNING:tensorflow:sample_weight
    class_weight = mc.get_class_weights(num_pos, num_neg,pos_weight=pos_weight)###Higher pos_weight Number Means enhance proportion to the weight of Positive class





    ### Fit modelEvaluateVal, shuffle is by defult True
    history1 = modelEvaluateVal.fit(xtrain, ytrain, batch_size=batch, epochs=epochs
                         , verbose=1, use_multiprocessing=True,
                         shuffle=FitShuffle ,class_weight=class_weight if flagPos_weight== True else None).history #,class_weight=class_weight





    ### Predict probabilities of xvalid data part
    yPredProb = modelEvaluateVal.predict(xvalid, verbose=1)




    ###Assign variable thresholdNum that determines a decimal Threshold value for classifying probabilities that will be predicted by the classifier
    thresholdNum=.5###.7,.56,.6,.8, .9
    yPredCategorical = mc.lossFunction(yPredProb,thresholdNum=thresholdNum)





    ###allFolds_LossEpoch_Train [:, [loss] ]
    allFolds_LossEpoch_Train = np.append(allFolds_LossEpoch_Train,
                        np.array(history1['loss']).reshape(len(history1['loss']), 1)
                        , axis=0) #append





    ####for plot all subplots
    allFolds_LossEpoch_plot_Train.append(history1['loss'])





    mc.pltLossVal(history1['loss'],shiftNumber,winLen, epochs, pathSavingPlotsPerRunning, baseFileName,
                   foldNum,batch,flagSeed, ylim=np.max(np.round(history1['loss'],4)),r2=r2, addAug=addAug , selectModelType= selectModelType)






    print(f"\nclass_weight_Fold_{foldNum}: {class_weight} \n")
    print(f"\npos_weight: {pos_weight}")
    print(f"thresholdNum: {thresholdNum}\n")


    print(f"\nAverage of train loss Fold_{foldNum}: {np.round(np.mean(history1['loss']),4)}")
    print(f"Average of yPredProbability Fold_{foldNum}: {np.round(np.float(yPredProb.mean()), 4)} \n")






    ###allFolds_ytest_yPredProb [:, [ytest,yPredProb] ]
    allFolds_ytest_yPredProb = np.append(allFolds_ytest_yPredProb,
                                  np.concatenate((yvalid.reshape(yvalid.shape[0], 1),yPredProb), axis=1)#concat
                                  , axis=0)#append






    LABELS = ['Normal 0', 'Anomaly 1']


    print(f"\nclassification_report_Fold_{foldNum}:")
    print(classification_report(yvalid.reshape(-1,1), np.array(yPredCategorical).reshape(-1,1), target_names=LABELS))





    mc.pltPrRe(yvalid.reshape(-1,1), yPredProb,shiftNumber,winLen, epochs, pathSavingPlotsPerRunning, baseFileName,foldNum,batch,flagSeed,r2=r2,addAug=addAug, selectModelType= selectModelType)





    mc.printConfMatrix(yvalid, yPredCategorical, foldNum, labelsValues=[0, 1])





    mc.pltConfMatrix(yvalid, yPredCategorical,shiftNumber,winLen, LABELS, epochs, pathSavingPlotsPerRunning, baseFileName,foldNum,batch, flagSeed,
                     figsizeValues=(6, 6), labelsValues=[0, 1],r2=r2,addAug=addAug, selectModelType= selectModelType)






    ###Delete classifier modelEvaluateVal and all dependencies of it and release memory to prevent crashing memory and CPU
    del modelEvaluateVal
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()





    datestr = time.strftime("%y%m%d_%H%M%S")
    print(f"End running time Fold_{foldNum}: {datestr} ,--------------------------. \n")







print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$: \n")
print(f"Average's Results of All Folds is such as below: \n")



LABELS = ['Normal 0', 'Anomaly 1']



###allFolds_ytest_yPredProb [:, [ytest,yPredProb] ]
yPredCategoricalAllfolds = mc.lossFunction(allFolds_ytest_yPredProb[:,1], thresholdNum=thresholdNum)



if flagPos_weight == True:
    print(f"\nclass_weight: {class_weight} \n")
    print(f"\npos_weight: {pos_weight}")

else:
    print(f"\n***Approach class_weight did not use***\n")




print(f"thresholdNum: {thresholdNum}\n")


print(f"\nAverage's train loss of All Folds : {np.round(np.float(allFolds_LossEpoch_Train[:,0].mean()),4)}")
print(f"Average of yPredProbability fold_All: {np.round(np.float(allFolds_ytest_yPredProb[:, 1].mean()), 4)} \n")






print("\n classification_report_AllFolds:")
print(classification_report(allFolds_ytest_yPredProb[:,0], yPredCategoricalAllfolds, target_names=LABELS))






mc.printConfMatrix(allFolds_ytest_yPredProb[:,0], yPredCategoricalAllfolds, foldNum="", labelsValues=[0, 1],AllFold=True)






mc.printPrReThreshold(allFolds_ytest_yPredProb[:,0], allFolds_ytest_yPredProb[:,1],thresholdDistance=4)






# ###allFolds_LossEpoch_Train [:, [loss] ]
mc.pltLossVal(allFolds_LossEpoch_plot_Train, shiftNumber, winLen, epochs, pathSavingPlotsPerRunning,
              baseFileName, foldNum="",batch=batch,
              flagSeed=True, ylim=.5,AllFold=True,r2=r2, addAug=addAug, selectModelType= selectModelType)






mc.pltConfMatrix(allFolds_ytest_yPredProb[:,0], yPredCategoricalAllfolds, shiftNumber, winLen
                 , LABELS, epochs, pathSavingPlotsPerRunning, baseFileName, foldNum="",batch=batch, flagSeed=True,
                 figsizeValues=(6, 6), labelsValues=[0, 1],AllFold=True,r2=r2, addAug=addAug, selectModelType= selectModelType)







mc.pltPrRe(allFolds_ytest_yPredProb[:,0], allFolds_ytest_yPredProb[:,1], shiftNumber, winLen, epochs, pathSavingPlotsPerRunning, baseFileName,foldNum,batch=batch, flagSeed=True,AllFold=True,r2=r2, addAug=addAug, selectModelType= selectModelType)







datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Main End running time: {datestr}, ******************************.")



