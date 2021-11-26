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

    def pltLossVal(self,historyLoss, historyValLoss, epochs,pathSavingPlotsPerRunning="",baseFileName="",ylim=.5):

        plt.figure()
        plt.plot(historyLoss, linewidth=2, label='Train', color="goldenrod")  # OR accuracy
        plt.plot(historyValLoss, linewidth=2, label='Validation', color="brown")  # OR val_accuracy
        plt.legend(loc='upper right', bbox_to_anchor=(1.13, 1.13))
        plt.title(f'Model Loss&Epoch ',pad=20,fontdict= {'weight' : 'bold'})
        plt.ylabel('Loss',labelpad=20,fontdict= {'weight' : 'bold'})
        plt.ylim(0, ylim)  # .2
        plt.xlabel('Epoch',labelpad=10,fontdict= {'weight' : 'bold'})
        # plt.savefig(pathSavingPlotsPerRunning + "/" + f"loss&valLoss_Fold_{foldNum}_Epochs{epochs}_flagSeed{flagSeed}_{baseFileName}.png",
        #             dpi=300, format='png')
        plt.show()



    def pltPrRe(self,ytest, yPred,foldNum,epochs,pathSavingPlotsPerRunning,baseFileName):

        yPred=np.round(yPred, 2)#[1:]
        precision_rt, recall_rt, threshold_rt = precision_recall_curve(ytest, yPred)
        plt.figure()
        plt.plot(threshold_rt,precision_rt[1:], label="Precision", linewidth=2, color="blue")
        plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=2, color="green")
        plt.title(f'Precision and recall for different threshold values in Fold_{foldNum}',pad=20,fontdict= {'weight' : 'bold','size'   : 8})
        plt.xlabel('Threshold',labelpad=10,fontdict= {'weight' : 'bold'})
        plt.ylabel('Precision/Recall',labelpad=20,fontdict= {'weight' : 'bold'})
        plt.legend()
        # plt.savefig(
        #     pathSavingPlotsPerRunning + "/" + f"Precision&Recall_Threshold_Epochs{epochs}_{baseFileName}.png",
        #     dpi=300, format='png')
        #plt.show()



    def pltConfMatrixMlbls(self,mlbConfusion, mlbClasses, figsizeValues=(6, 6)):

        plt.figure(figsize=figsizeValues)
        for j, (label, matrix) in enumerate(zip(mlbClasses, mlbConfusion.astype(int))):
            plt.subplot(f'23{j + 1}')
            labels = [f'Not_{label}', label]
            sns.heatmap(matrix, annot=True, square=True, fmt='d', cbar=False, cmap='YlGnBu',
                        cbar_kws={'label': 'My Colorbar'},  # , fmt = 'd'
                        xticklabels=labels, yticklabels=labels, linecolor='black', linewidth=1)

            plt.ylabel('True class', labelpad=20, fontdict={'weight': 'bold'})
            plt.xlabel('Predicted class', labelpad=20, fontdict={'weight': 'bold'})
            plt.title(labels[0], pad=20, fontdict={'weight': 'bold'})

        plt.tight_layout()
        # plt.savefig(pathSavingPlotsPerRunning + "/" + f"pltConfMatrixMlbls_Fold_{foldNum}_Epochs{epochs}_flagSeed{flagSeed}_{baseFileName}.png",dpi=300, format='png')
        plt.show()


