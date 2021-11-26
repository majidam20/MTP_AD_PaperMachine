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

import tsfresh as t
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction.settings import TimeBasedFCParameters
from tsfresh.feature_extraction import extract_features
from tsfresh import extract_features
from tsfresh.examples import robot_execution_failures
import warnings

###***
import seglearn
from seglearn.preprocessing import check_ts_data_with_ts_target
from seglearn.transform import FeatureRep, Segment
from seglearn.pipe import Pype
from seglearn.feature_functions import mean, var, std, skew,base_features
from seglearn.datasets import load_watch
###***

from pathlib import Path
pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')
pathData = pathMainCodes + "/data/paperMachine/"
pathDataAuged = pathMainCodes + "/data/paperMachine/auged/"
pathDataAuged_jitter_NN = pathMainCodes + "/data/paperMachine/auged/jitterFor_NN/"
pathData_NewFromWeb = pathMainCodes + "/data/paperMachine/paperMachine_NewFromWeb/"
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)

datestr = time.strftime("%y%m%d_%H%M%S")
print(f"Start running time: {datestr}")

print(seglearn.feature_functions.all_features())

import numpy as np
from seglearn.transform import InterpLongToWide

# sample stacked input with values from 2 variables each with 2 channels
t = np.array([1.1, 1.2, 2.1, 3.3, 3.4, 3.5])
s = np.array([0, 1, 0, 0, 1, 1])
v1 = np.array([3, 4, 5, 7, 15, 25])
v2 = np.array([5, 7, 6, 9, 22, 35])
X = [np.column_stack([t, s, v1, v2])]
y = [np.array([1, 2, 2, 2, 3, 3])]

stacked_interp = InterpLongToWide(0.5)
stacked_interp.fit(X, y)
Xc, yc, _ = stacked_interp.transform(X, y)

print('Xc',Xc)
print('yc',yc)

#df = pd.read_csv(pathData + "dfpShifted5_ForAug_withAuged_1To5_201209_233059.csv",header=None)
# df["id"]=1
# df=df[:10]
# y=df.loc[:, 0]

#
# robot_execution_failures.download_robot_execution_failures()
# df, y = robot_execution_failures.load_robot_execution_failures()
# print(df)
# extracted_features = extract_features(df,column_id="id", column_sort="time")
# print(extracted_features.shape)
# print(extracted_features.head(5))

# from tsfresh import select_features
# from tsfresh.utilities.dataframe_functions import impute
#
# impute(extracted_features)
# features_filtered = select_features(extracted_features, y)