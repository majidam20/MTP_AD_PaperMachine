###*** Master Thesis Project Anomaly detection and forecasting in time series sensor data for Paper Machine

###*** Author Majid Aminian, Department of Data Science in Beuth Hochschule Berlin


### Import Libraries
import os
import sys
import gc
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import math
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
from sklearn.metrics import precision_recall_curve
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
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
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import ModelCheckpoint
###**** End tensorflow.keras

import warnings
warnings.filterwarnings("ignore")





class mainClass_CV:
    def __init__(self,allPrRe):
        self.dfpRaw1=pd.DataFrame()
        self.allPrRe=allPrRe



    ###Create DS with having just one ActualRaw row as break label 1, Then drop a few unnecessary columns.
    def delRepetitiveOnes(self,dfpRaw1):

        # Delete Unnecessary columns
        dfpRaw2 = dfpRaw1.drop([0, 29, 62], axis=1)
        dfpRaw2.columns = [i for i in range(dfpRaw2.shape[1])]
        dfpRaw2 = dfpRaw2.drop([0], axis=0)
        dfpRaw2 = dfpRaw2.reset_index(drop=True)



        # Covert from Pandas to Numpy array
        dfpRaw2 = dfpRaw2.values



        # Convert data of DS to float64
        dfpRaw2 = np.asarray(dfpRaw2).astype('float64')



        # Delete repetitive breaks that labeled as 1 (when machine breaks, then generates labels one for a while) in ActualRaw dataset(DS)
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



        # Delete Two rows after each break, because they are noisy data that machine generates after repairing its breaks
        indexLblOnes = np.where(dfpRaw2[:, 0] == True)
        l = []
        for i in range(1, 3):
            for j in indexLblOnes:
                l.append(j + i)
        l1 = np.array(l)
        l2 = l1.reshape(l1.shape[0] * l1.shape[1], ).tolist()

        dfpRawNoReptOnes = np.delete(dfpRaw2, l2, axis=0)



        return dfpRawNoReptOnes



    # Apply minmax scaler to all columns except the label column
    def scaleData(self, yXtrain, yXtest=0, doScaleAfterSplit= True):


        if doScaleAfterSplit == False:

            min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
            x_scaled = min_max_scaler.fit_transform(yXtrain[:, 1:])
            yXAll = np.concatenate((yXtrain[:, 0].reshape(-1, 1), x_scaled), axis=1)

            return yXAll



        if doScaleAfterSplit == True:
            # Apply minmax scaler to all columns except the label column
            min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

            Xtrain_scaled = min_max_scaler.fit_transform(yXtrain[:, 1:])
            Xtest_scaled  = min_max_scaler.transform(yXtest[:, 1:])


            yXtrain_scaled = np.concatenate((yXtrain[:, 0].reshape(-1, 1), Xtrain_scaled), axis=1)
            yXtest_scaled = np.concatenate((yXtest[:, 0].reshape(-1, 1), Xtest_scaled), axis=1)

            return yXtrain_scaled, yXtest_scaled




    def shiftLblOnes(self,dfpRaw3Scaled_NoReptOnes,shiftNumber):

        # Drop breaks that distance with previous breaks are less than shiftNumber
        il = [0, 0]
        ones = [[0]]
        ll = []

        for i in range(dfpRaw3Scaled_NoReptOnes.shape[0]):
            if dfpRaw3Scaled_NoReptOnes[i, 0] == 1:
                il.append(i)
                if i - il[-2] <= shiftNumber:
                    if np.where(dfpRaw3Scaled_NoReptOnes[:i, 0] == 1)[0].size==0:
                        ones.append(np.arange(i, -1, -1).tolist())
                        continue

                    mm = np.max(np.where(dfpRaw3Scaled_NoReptOnes[:i, 0] == 1))
                    ones.append(np.arange(i, mm, -1).tolist())



        for i in range(1, len(ones)):
            for j in range(len(ones[i])):
                ll.append(ones[i][j])

        ll.sort()
        dfpRaw3NoLess = np.delete(dfpRaw3Scaled_NoReptOnes, ll, axis=0)



        # Assign label one to row with desired user-predefined shift number
        firstOnes = []
        shiftedOnes = []



        for i in range(dfpRaw3NoLess.shape[0]):
            if i == dfpRaw3NoLess.shape[0] - 1:
                break
            if dfpRaw3NoLess[i, 0] == 1:
                firstOnes.append(i)
                shiftedOnes.append(i - shiftNumber)



        ###Assign label one to row(determined by shift number) before than actual label one and delete actual label one
        dfpRaw3NoLess[shiftedOnes, 0] = 1
        dfpRaw3Shifted = np.delete(dfpRaw3NoLess, firstOnes, axis=0)

        """
        Now we have Data without Actual breaks, but just one row as label one before actual breaks followed by  user-predefined shift number 
        """


        return dfpRaw3Shifted





    #### generateRoll data corresponding to user-predefined windowLength
    def generateRoll(self,X, y, winLength):
        output_X = []
        output_y = []


        for i in range(-2, len(X) - winLength - 1):
            t = []
            for j in range(1, winLength + 1):
                t.append(X[i + j + 1, :])

            winLength2 = winLength - 1
            i += 2

            output_X.append(t)
            output_y.append(y[i + winLength2])


        y1 = np.repeat(np.array(output_y), winLength)
        y2 = y1.reshape(y1.shape[0], 1)
        X = np.array(output_X).reshape(len(output_X) * winLength, X.shape[1])
        yX = np.concatenate((y2, X), axis=1)


        return yX





    #### generateMean similar labeled rows in amount of user-predifined windowLength, purpose of calculating mean is for feature selection issues, e.g I calculate mean(row base mean axis=0) of each 2 rows(windowLength) data for feature selection purposes, instead of do feature selection after transforming 2 rows to 1 row data.
    def generateMean(self,X, y, winLength):

        yXMean = np.empty([0, X.shape[1]])
        output_X = []
        output_y = []
        i = -2

        while i < len(X) - winLength - 1:
            t = []
            for j in range(1, winLength + 1):
                t.append(X[i + j + 1, :])

            winLength2 = winLength - 1
            ii = i + 2

            output_X.append(t)
            tempMean = np.concatenate(output_X, axis=0).mean(axis=0)

            yXMean = np.append(yXMean, tempMean.reshape(1, tempMean.shape[0]), axis=0)
            output_y.append(y[ii + winLength2])

            output_X = []

            i += winLength


        y2 = np.array(output_y).reshape(len(output_y), 1)
        yX = np.concatenate((y2, yXMean), axis=1)


        return yX






    #### Implement  Features selection approach, with permutation_importance(RandomForestClassifier)
    def selectFeatures(self,X,y,winLength):

        yXMeanFinal = self.generateMean(X, y, winLength)

        yXMeanFinal = pd.DataFrame(yXMeanFinal)
        df = yXMeanFinal

        tempSelFeat = []
        selFeat = []



        ###Two times run RandomForestClassifier and permutation_importance To decrease unpredictive features
        for i in range(2):

            forest = RandomForestClassifier(n_estimators=300, criterion="entropy", verbose=0, n_jobs=-1,
                                        random_state=42)
            forest.fit(df.iloc[:, 1:], df.iloc[:, 0])



            permRF = permutation_importance(forest, df.iloc[:, 1:], df.iloc[:, 0], n_jobs=-1, n_repeats=5,
                                        random_state=42)

            print(f"permRF.importances_mean_{i}:\n {pd.DataFrame(np.round(permRF.importances_mean, 6))}\n")



            ###Select lower predictive features on target variable in first iteration with a lower threshold value
            if i == 0:
                threshold = .0001


                selFeat = [x + 1 for x in np.where(permRF.importances_mean >= threshold)[0].tolist()]
                print(f"NumberOf_SelFeatIter_{i}: {len(selFeat)}")
                print(f"selFeatIter_{i}: {selFeat}")

                tempSelFeat.append(selFeat)
                df = df.loc[:, [0] + selFeat]



            ###Select higher predictive features on target variable in second iteration with a higher threshold value
            if i == 1:
                threshold = .001

                tempSelFeat.append(np.where(permRF.importances_mean >= threshold)[0].tolist())
                selFeat = [tempSelFeat[0][index] for index in tempSelFeat[1]]

                print(f"NumberOf_FinalSelFeatIter_{i}: {len(selFeat)}")
                print(f"FinalSelFeatIter_{i}: {selFeat}")



            del forest
            del permRF


        #Value [0] is label column index
        return [0]+ selFeat






    ###Transform rows corresponding to user-predefined windowLength, Transformation is done for the purpose to prevent disordering rows
    def transformShapeData(self,yX,winLength,toOneRow=False,fromOneRow=False):

        y1 = yX[:, 0]
        X1 = yX[:, 1:]

        if toOneRow== True:
            X11 = X1.reshape(int(X1.shape[0] / winLength), X1.shape[1] * winLength)
            y11 = y1.reshape(int(y1.shape[0] / winLength), winLength)
            y111 = y11[:, 0].reshape(y11[:, 0].shape[0], 1)

            yXTransformed = np.concatenate((y111, X11), axis=1)


        if fromOneRow == True:
            X11=np.reshape(X1,(int(X1.shape[0]*winLength),X1.shape[1]))
            y11=np.reshape(y1,(int(y1.shape[0]*winLength),1))
            y111=y11.reshape(y11.shape[0],y11.shape[1])[:,0]

            yXTransformed = np.concatenate((y111, X11), axis=1)



        return yXTransformed





    ###Transformation data to be in 3 dimensional, for LSTM model
    def transformTo3Dim(self, xtrain , ytrain , xtest , ytest):


        xtrain = np.reshape(xtrain, (int(xtrain.shape[0]), 1, xtrain.shape[1]))
        ytrain = np.reshape(ytrain, (int(ytrain.shape[0]), 1, 1))
        ytrain = ytrain.reshape(ytrain.shape[0], ytrain.shape[1])[:, 0]


        xtest = np.reshape(xtest, (int(xtest.shape[0]), 1, xtest.shape[1]))
        ytest = np.reshape(ytest, (int(ytest.shape[0]), 1, 1))
        ytest = ytest.reshape(ytest.shape[0], ytest.shape[1])[:, 0]


        return xtrain, ytrain , xtest , ytest




    ###Define layers of each selected Model type
    def selectModelTypeFunc(self,xtrain , modelEvaluateVal, flagR1,r1,r2, selectModelType="DNN"):

        if selectModelType == "DNN":

            modelEvaluateVal.add(Dense(xtrain.shape[1] * 2, activation='tanh', input_dim=xtrain.shape[1]))  # relu, tanh
            modelEvaluateVal.add(Dense(xtrain.shape[1] * 1.6, activation='tanh',
                                       kernel_regularizer=l1(r1) if flagR1 else l2(r2)
                                       ))
            modelEvaluateVal.add(Dense(xtrain.shape[1] * .31, activation='tanh',
                                       kernel_regularizer=l1(r1) if flagR1 else l2(r2)
                                       ))
            modelEvaluateVal.add(Dense(1, activation='sigmoid'))





        if selectModelType=="LSTM":


            timesteps = xtrain.shape[1]  # equal to the lookback
            n_features = xtrain.shape[2]



            ###LSTM Model
            modelEvaluateVal.add(LSTM(xtrain.shape[2]*2, activation='tanh', input_shape=(timesteps, n_features)))### tanh, relu

            modelEvaluateVal.add(Dense(int(xtrain.shape[2]*1.6)
                                     , activation='tanh'
                                     , kernel_regularizer=l1(r1) if flagR1 else l2(r2)
                                     ))
            modelEvaluateVal.add(Dense(int(xtrain.shape[2] * .31), activation='tanh',
                                       kernel_regularizer=l1(r1) if flagR1 else l2(r2)
                                       ))


            modelEvaluateVal.add(Dense(1, activation='sigmoid'))



        return modelEvaluateVal








    ###Define loss function for classifying predicted probabilities that created by Deep Neural Network binary classifier
    def lossFunction(self,yPredProb,thresholdNum=.5):
        l = []
        for i in yPredProb:
            if i < thresholdNum:
                l.append(0)
            else:
                l.append(1)

        yPred = l

        return yPred






    ###Calculate the weight of each label to assign weight to minority class in order to prevent the model from overfitting on the majority class
    def get_class_weights(self,num_pos, num_neg,insideCrossVal=True, pos_weight=0.7):
        '''
        Computes weights for each class to be applied in the loss function during training.
        :param num_pos: # positive samples
        :param num_neg: # negative samples
        :return: A dictionary containing weights for each class
        '''

        if insideCrossVal==False:
            total =  num_pos + num_neg
            print('\nNumber of Rows Aucual Dataset After Cleaning step:\n    Total: {}\n    Anomalies(positives): {} ({:.2f}% of total)\n'.format(total, num_pos, 100 * num_pos / total))



        else:
            weight_neg = (1 - pos_weight) * (num_neg + num_pos) / (num_neg)
            weight_pos = pos_weight * (num_neg + num_pos) / (num_pos)
            class_weight = {0: weight_neg , 1: weight_pos }

            return class_weight







   ##############################****** Ploting Part ******##############################

    ###Plot Loss and epochs for Train part of DS
    def pltLossVal(self,historyLoss, shiftNumber,winLength, epochs, pathSavingPlotsPerRunning,baseFileName, foldNum,batch,flagSeed=True,ylim=.5,AllFold=False,r2="", addAug=True , selectModelType= "DNN"):
        if AllFold==True:
            foldNum="All"

            fig, axes = plt.subplots(5, 1, figsize=(30, 25))  # ,sharex =False
            for i, lossFolds in zip(range(len(historyLoss)) , historyLoss) :

                axes[i].plot(lossFolds, linewidth=2, label='Train', color="goldenrod")  # OR accuracy


                leg1 = axes[i].legend(loc='best',prop={'size': 20})
                leg1.get_frame().set_edgecolor('b')
                leg1.get_frame().set_linewidth(2)

                axes[i].set_title(f'Model Loss-Epoch Fold_{i+1}', pad=20, fontdict={'weight': 'bold' , 'size':15, 'color': 'blue'})
                axes[i].set_ylabel('Loss', labelpad=10, fontdict={'weight': 'bold', 'size':20})
                axes[i].set_ylim(0, np.max(np.round(lossFolds,4)))  # .2
                axes[i].set_xlabel('Epoch', labelpad=10, fontdict={'weight': 'bold' , 'size':20})



            plt.tight_layout(h_pad=3, w_pad=1)
            plt.savefig(pathSavingPlotsPerRunning + "/" + f"loss_Fold_{foldNum}_shiftNumber{shiftNumber}_winLength{winLength}_Model{selectModelType}_Epochs{epochs}_Batch{batch}_Ridge{r2}_flagSeed{flagSeed}_addAug{addAug}_{baseFileName}.png",dpi=300, format='png')

            return 0



        plt.figure()
        plt.plot(historyLoss, linewidth=2, label='Train', color="goldenrod")  # OR accuracy
        plt.legend(loc='upper right', bbox_to_anchor=(1.13, 1.13))
        plt.title(f'Loss-Epoch Model Fold_{foldNum}',pad=20,fontdict= {'weight' : 'bold'})
        plt.ylabel('Loss',labelpad=10,fontdict= {'weight' : 'bold'})
        plt.ylim(0, ylim)  # .2
        plt.xlabel('Epoch',labelpad=10,fontdict= {'weight' : 'bold'})
        plt.savefig(pathSavingPlotsPerRunning + "/" + f"loss_Fold_{foldNum}_shiftNumber{shiftNumber}_winLength{winLength}_Model{selectModelType}_Epochs{epochs}_Batch{batch}_Ridge{r2}_flagSeed{flagSeed}_addAug{addAug}_{baseFileName}.png",
                    dpi=300, format='png')
        #plt.show()








    ###Plot precision_rt, recall_rt Curve by precision_recall_curve(ytest, yPredProbability) function
    def pltPrRe(self,yvalid, yPred, shiftNumber, winLength,epochs,pathSavingPlotsPerRunning,baseFileName,foldNum,batch,flagSeed=True,AllFold=False,r2="",addAug=True, selectModelType= "DNN"):

        if AllFold == True:
            foldNum = "All"
        precision_rt, recall_rt, threshold_rt = precision_recall_curve(yvalid, yPred)
        plt.figure()
        plt.plot(precision_rt[1:], recall_rt[1:], label="Precision-Recall (Anomaly 1)", linewidth=2, color="green")
        plt.title(f'Precision-Recall in Fold_{foldNum}',pad=20,fontdict= {'weight' : 'bold','size'   : 8})
        plt.xlabel('Recall',labelpad=10,fontdict= {'weight' : 'bold'})
        plt.ylabel('Precision',labelpad=10,fontdict= {'weight' : 'bold'})
        plt.legend(loc="best")
        plt.savefig(pathSavingPlotsPerRunning + "/" + f"Precision&Recall_Fold_{foldNum}_shiftNumber{shiftNumber}_winLength{winLength}_Model{selectModelType}_Epochs{epochs}_Batch{batch}_Ridge{r2}_flagSeed{flagSeed}_addAug{addAug}_{baseFileName}.png",
                    dpi=300, format='png')
        #plt.show()





    ### To print Average rates such as threshold, precision, recall AllFolds at the end of running after finishing all CV itereations
    def printPrReThreshold(self,yvalid, yPred, thresholdDistance = 1):

        precision_rt, recall_rt, threshold_rt = precision_recall_curve(yvalid, np.round(yPred, thresholdDistance))

        self.allPrRe = np.append(self.allPrRe,np.concatenate([threshold_rt.reshape(-1,1), precision_rt[1:].reshape(-1,1), recall_rt[1:].reshape(-1,1)], axis=1),axis=0)

        threshold_rt2, precision_rt2, recall_rt2=self.allPrRe[:,0].mean() , self.allPrRe[:,1].mean() , self.allPrRe[:,2].mean()
        print("Average threshold_rt AllFolds: ", np.round(threshold_rt2,2))
        print("Average precision_rt AllFolds: ", np.round(precision_rt2,2))
        print("Average recall_rt AllFolds: ",    np.round(recall_rt2,2), "\n")







    ###Print specific number of tn, fp, fn, tp by confusion_matrix(ytest, yPredCategorical) function
    def printConfMatrix(self,yvalid,yPredCategorical, foldNum, labelsValues=[0, 1],AllFold=False,numberOfSplits=5):
        if AllFold==True:
            print(f"confusion_matrix_AllFolds: ")
        else:
            print(f"confusion_matrix_Fold_{foldNum}: ")

        tn, fp, fn, tp = confusion_matrix(yvalid, yPredCategorical, labels=labelsValues).ravel()

        print("True Negatives: ", tn)
        print("False Positives: ", fp)
        print("False Negatives: ", fn)
        print("True Positives: ", tp, "\n")






    ###Plot heatmap corresponding number of tn, fp, fn, tp by confusion_matrix(ytest, yPredCategorical) function
    def pltConfMatrix(self,yvalid, yPredCategorical,shiftNumber,winLength,LABELS,epochs,pathSavingPlotsPerRunning,baseFileName,foldNum,batch,flagSeed=True,figsizeValues=(6, 6), labelsValues=[0, 1],AllFold=False,r2="" ,addAug=True, selectModelType= "DNN"):

        if AllFold==True:
            foldNum="All"

        conf_matrix = confusion_matrix(yvalid, yPredCategorical, labels=labelsValues)

        plt.figure(figsize=figsizeValues)
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d",cmap='YlGnBu');
        plt.title(f"Confusion matrix in Fold_{foldNum}",pad=20,fontdict= {'weight' : 'bold'})
        plt.ylabel('True class',labelpad=20,fontdict= {'weight' : 'bold'})
        plt.xlabel('Predicted class',labelpad=20,fontdict= {'weight' : 'bold'})

        plt.savefig(pathSavingPlotsPerRunning + "/" + f"ConfusionMatrix_Fold_{foldNum}_shiftNumber{shiftNumber}_winLength{winLength}_Model{selectModelType}_Epochs{epochs}_Batch{batch}_Ridge{r2}_flagSeed{flagSeed}_addAug{addAug}_{baseFileName}.png",dpi=300, format='png')
        #plt.show()





    ###Comparison distplot of Actual data and generated synthetic
    def pltDfCompSnsDist(self,Xtrain, Xauged,shiftNumber,winLen, pathSavingPlotsPerRunning, baseFileName, flagSeed, w=20, h=200, bins=20, epochs=1000):

        colNumSubPlot=8
        rowsNumSubPlot = math.ceil(Xtrain.shape[1] / colNumSubPlot)
        colsList = [i for i in range(0, colNumSubPlot)] * rowsNumSubPlot

        rowsList=np.repeat(np.arange(rowsNumSubPlot), colNumSubPlot)

        fig, axes = plt.subplots(rowsNumSubPlot, colNumSubPlot,figsize=(w, h),sharex =False)


        for (norm, abnorm,
             row,col,
             colorNorm, colorAbnorm) in itertools.zip_longest \
            (np.arange(0, Xtrain.shape[1]), np.arange(0, Xauged.shape[1]), \
            rowsList, colsList,
            ['g'] * int(Xtrain.shape[1]), ['r'] * int(Xauged.shape[1])):

            if norm == None:
                break

            sns.distplot(Xtrain[:, norm], ax=axes[row, col], color=colorNorm, label=norm, bins=bins)
            sns.distplot(Xauged[:, abnorm], ax=axes[row, col], color=colorAbnorm, label=abnorm, bins=bins)

            axes[row, col].set_ylabel(f"Col_{norm}", labelpad=5, fontweight="bold", fontsize=7)


            axes[row, col].tick_params(axis='y', which='major', labelsize=7)
            axes[row, col].tick_params(axis='y', which='major', labelsize=7)

            plt.setp(axes, xticks=[])
            plt.tight_layout(h_pad=.2,w_pad=1)
            plt.legend(loc='upper left', bbox_to_anchor=(0, 0))


            ###Adjust Legend properties
            if norm==0:
                leg1 = axes[row, col].legend(["Actual","Synthetic"],loc="upper left", prop={'size': 6,"weight":"bold"})
                leg1.get_frame().set_edgecolor('b')
                leg1.get_frame().set_linewidth(1)


            print(f"ColNum_{norm+1}, created")



        plt.savefig(pathSavingPlotsPerRunning + "/" + f"CompSnsDist_shiftNumber{shiftNumber}_winLength{winLen}_Epochs_{epochs}_flagSeed{flagSeed}_{baseFileName}.png",dpi=300, format='png')
        #plt.show()


