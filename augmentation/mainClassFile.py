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

class mainClass:
    def __init__(self):
        self.dfconfMatrix=pd.DataFrame()


    #dfconfMatrix=pd.DataFrame()

    def generateRol(self,X, y, lookback=5):
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

    def pltLossVal(self,historyLoss, historyValLoss, foldNum, epochs, pathSavingPlotsPerRunning,baseFileName,flagSeed=True,ylim=.5,AllFold=False):
        if AllFold==True:
            foldNum="All"

        plt.figure()
        plt.plot(historyLoss, linewidth=2, label='Train', color="goldenrod")  # OR accuracy
        plt.plot(historyValLoss, linewidth=2, label='Validation', color="brown")  # OR val_accuracy
        plt.legend(loc='upper right', bbox_to_anchor=(1.13, 1.13))
        plt.title(f'Model Loss&Epoch Fold_{foldNum}',pad=20,fontdict= {'weight' : 'bold'})
        plt.ylabel('Loss',labelpad=20,fontdict= {'weight' : 'bold'})
        plt.ylim(0, ylim)  # .2
        plt.xlabel('Epoch',labelpad=10,fontdict= {'weight' : 'bold'})
        plt.savefig(pathSavingPlotsPerRunning + "/" + f"loss&valLoss_Fold_{foldNum}_Epochs{epochs}_flagSeed{flagSeed}_{baseFileName}.png",
                    dpi=300, format='png')
        #plt.show()



    def pltPrRe(self,ytest, yPred,foldNum,epochs,pathSavingPlotsPerRunning,baseFileName,flagSeed=True,AllFold=False):
        if AllFold==True:
            foldNum="All"

        yPred=np.round(yPred[np.where(yPred > 0)], 2)#[1:]
        precision_rt, recall_rt, threshold_rt = precision_recall_curve(ytest, yPred)
        plt.figure()
        plt.plot(threshold_rt,precision_rt[1:], label="Precision", linewidth=2, color="blue")
        plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=2, color="green")
        plt.title(f'Precision and recall for different threshold values in Fold_{foldNum}',pad=20,fontdict= {'weight' : 'bold','size'   : 8})
        plt.xlabel('Threshold',labelpad=10,fontdict= {'weight' : 'bold'})
        plt.ylabel('Precision/Recall',labelpad=20,fontdict= {'weight' : 'bold'})
        plt.legend()
        #plt.tight_layout()
        plt.savefig(
            pathSavingPlotsPerRunning + "/" + f"Precision&Recall_Threshold_Fold_{foldNum}_Epochs{epochs}_flagSeed{flagSeed}_{baseFileName}.png",
            dpi=300, format='png')
        #plt.show()


    def printConfMatrix(self,ytest,yPred, foldNum, labelsValues=[0, 1],AllFold=False,numberOfSplits=5):
        if AllFold==True:
            foldNum="All"

        print(f"confusion_matrix_Fold_{foldNum}: ")
        tn, fp, fn, tp = confusion_matrix(ytest, yPred, labels=labelsValues).ravel()

        print("True Negatives: ", tn)
        print("False Positives: ", fp)
        print("False Negatives: ", fn)
        print("True Positives: ", tp, "\n")

        print(f"NumberOfTrueClassified_AllLabels_Fold_{foldNum}: {tn+tp} \n")
        print(f"NumberOfFalseClassified_AllLabels_Fold_{foldNum}: {fn + fp} \n")


    def pltConfMatrix(self,ytest, yPred,LABELS,foldNum,epochs,pathSavingPlotsPerRunning,baseFileName,flagSeed=True,figsizeValues=(6, 6), labelsValues=[0, 1],AllFold=False):

        if AllFold==True:
            foldNum="All"

        conf_matrix = confusion_matrix(ytest, yPred, labels=labelsValues)

        plt.figure(figsize=figsizeValues)
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d",cmap='YlGnBu');
        plt.title(f"Confusion matrix in Fold_{foldNum}",pad=20,fontdict= {'weight' : 'bold'})
        plt.ylabel('True class',labelpad=20,fontdict= {'weight' : 'bold'})
        plt.xlabel('Predicted class',labelpad=20,fontdict= {'weight' : 'bold'})
        plt.savefig(pathSavingPlotsPerRunning + "/" + f"ConfusionMatrix_Fold_{foldNum}_Epochs{epochs}_flagSeed{flagSeed}_{baseFileName}.png",dpi=300, format='png')
        #plt.show()