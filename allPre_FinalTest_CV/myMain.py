import numpy as np
import os
import utils.augmentation as aug
import random

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)


from pathlib import Path
import pandas as pd
### Define main path addresses of project
pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')



yXtrain = pd.read_csv(pathCurrrent + "/npScaledNoRepOne.csv", header=None)


yXtrain=yXtrain.values


xPred = np.empty([0, yXtrain.shape[1] - 1])



yXtrain=yXtrain[np.where(yXtrain[:,0]==1)]


Xtrain=yXtrain[:,1:]




Xtrain = Xtrain.reshape((-1, Xtrain.shape[0], Xtrain.shape[1]))


numberOfAug=4

for i in range(1, numberOfAug):

    npTimeWarping_X= aug.time_warp(Xtrain, sigma= 0.2 + (0.0000001 + i / 10000000))[0]

    xPred = np.append(xPred, npTimeWarping_X, axis=0)




ytrain = np.repeat(np.array([[1]]), xPred.shape[0],axis=0)
yX_TimeWarping= np.concatenate((ytrain,xPred),axis=1)




###pd.DataFrame(npTimeWarping).to_csv('npScaledNoRepOne_DTW.csv',header=None, index=None)

print('done')


1## Time Warping Description

# Random smooth time warping.
#
# ```python
# aug.time_warp(x, sigma=0.2, knot=4)
# ```
#
# Based on:
# T. T. Um et al, "Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks," in ACM ICMI, pp. 216-220, 2017.
#
# ##### Arguments
# **x** : *3D numpy array*
#
# &nbsp;&nbsp;&nbsp;&nbsp; Numpy array of time series in format `(batch, time_steps, channel)`.
#
# **sigma** : *float*
#
# &nbsp;&nbsp;&nbsp;&nbsp; Standard deviation of the random magnitudes of the warping path.
#
# **knot** : *int*
#
# &nbsp;&nbsp;&nbsp;&nbsp; Number of hills/valleys on the warping path.
#
# ##### Returns
# *3D numpy array*
#
# &nbsp;&nbsp;&nbsp;&nbsp; Numpy array of generated data of equal size of the input `x`.