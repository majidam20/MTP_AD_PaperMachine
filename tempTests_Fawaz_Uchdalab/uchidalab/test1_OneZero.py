import numpy as np
import pandas as pd
import os
from utils.input_data import read_data_sets
import utils.datasets as ds
import utils.augmentation as aug
import utils.helper as hlp
import matplotlib.pyplot as plt
import time
import random

#dfp.drop(['time','index'], axis=1, inplace=True)
#
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)

dataset = "CBF"

nb_class = ds.nb_classes(dataset)
nb_dims = ds.nb_dims(dataset)

# Load Data
# train_data_file = os.path.join("data", dataset, "%s_TRAIN.tsv"%dataset)
# test_data_file = os.path.join("data", dataset, "%s_TEST.tsv"%dataset)
# x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter="\t")

train_data_file = os.path.join("data", "forAugDS", "dfpShifted5_ForAugOneZero201219_201615.csv")#dfpShifted5_ForAug_AllTrainTest_201204_161806
test_data_file = os.path.join("data","forAugDS", "dfpShifted5_ForAugOneZero201219_201615.csv")#dfpShifted5_ForAug_AllTrainTest_201204_161806
x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter=",")

y_train = ds.class_offset(y_train, dataset)
y_test= ds.class_offset(y_test, dataset)
nb_timesteps = int(x_train.shape[1] / nb_dims)
input_shape = (nb_timesteps , nb_dims)

# x_train_max = np.max(x_train)
# x_train_min = np.min(x_train)
# x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
# # Test is secret
# x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.

# wDBA
# x_train=x_train[:8]
# y_train=y_train[:8]

x_test = x_test.reshape((-1, input_shape[0], input_shape[1]))
x_train = x_train.reshape((-1, input_shape[0], input_shape[1]))

#print(hlp.plot1d(x_train[0], aug.jitter(x_train)[0]))
datestr = time.strftime("%y%m%d_%H%M%S")

# CBF=pd.DataFrame(np.array(np.squeeze(aug.jitter(x_train),axis=2)))
# CBF.to_csv(f"data/augedOtherDS/CBF{datestr}.csv")


#Jittering, Rotation, Scaling, Magnitude Warping, Permutation, Slicing, Time Warping, Window Warping ,SPAWNER, wDBA, RGW, DGW
########## best--> Jittering, Scaling, wDBA, Magnitude **************
df_jitter2=pd.DataFrame()
for i in range(4):
    df_jitter     =pd.DataFrame(np.array(np.squeeze(aug.jitter(x_train,sigma=0.03+i/100),axis=2)))
    df_jitter     =pd.concat([pd.DataFrame(np.where(y_train==0,1,0).astype(int)),df_jitter],axis=1)
    df_jitter2=df_jitter2.append(df_jitter)

df_jitter2     .to_csv(f"data/augedPaperMachine/df_jitter2_OneZero_{datestr}.csv",index=False,header=None)
#
# # df_Rotation   =pd.DataFrame(np.array(np.squeeze(aug.rotation(x_train),axis=2)))
# #df_Scaling    =pd.DataFrame(np.array(np.squeeze(aug.scaling(x_train),axis=2)))
# #df_Scaling    =pd.concat([pd.DataFrame(np.where(y_train==0,1,0).astype(int)),df_Scaling],axis=1)
# df_Magnitude  =pd.DataFrame(np.array(np.squeeze(aug.magnitude_warp(x_train),axis=2)))
# df_Magnitude  =pd.concat([pd.DataFrame(np.where(y_train==0,1,0).astype(int)),df_Magnitude],axis=1)
# df_Permutation=pd.DataFrame(np.array(np.squeeze(aug.permutation(x_train),axis=2)))
# df_Slicing    =pd.DataFrame(np.array(np.squeeze(aug.window_slice(x_train),axis=2)))
#
# df_TimeWarping     =pd.DataFrame(np.array(np.squeeze(aug.time_warp(x_train),axis=2)))
# df_TimeWarping     =pd.concat([pd.DataFrame(np.where(y_train==0,1,0).astype(int)),df_TimeWarping],axis=1)
# df_WindowWarping    =pd.DataFrame(np.array(np.squeeze(aug.window_warp(x_train),axis=2)))
# df_WindowWarping     =pd.concat([pd.DataFrame(np.where(y_train==0,1,0).astype(int)),df_WindowWarping],axis=1)
#
# #df_jitter     .to_csv(f"data/augedPaperMachine/df_jitter_{datestr}.csv",index=False,header=None)
# #df_Rotation   .to_csv(f"data/augedPaperMachine/df_Rotation_{datestr}.csv",index=False,header=None)
# #df_Scaling    .to_csv(f"data/augedPaperMachine/df_Scaling_{datestr}.csv",index=False,header=None)
# df_Magnitude  .to_csv(f"data/augedPaperMachine/df_Magnitude_{datestr}.csv",index=False,header=None)
# df_Permutation.to_csv(f"data/augedPaperMachine/df_Permutation_{datestr}.csv",index=False,header=None)
# df_Slicing    .to_csv(f"data/augedPaperMachine/df_Slicing_{datestr}.csv",index=False,header=None)
# df_TimeWarping.to_csv(f"data/augedPaperMachine/df_TimeWarping_{datestr}.csv",index=False,header=None)
# df_WindowWarping.to_csv(f"data/augedPaperMachine/df_WindowWarping_{datestr}.csv",index=False,header=None)
#
# # wDBA, Random Guided Warping, Discriminative Guided Warping
# df_wDBA     =pd.DataFrame(np.array(np.squeeze(aug.wdba(x_train,y_train),axis=2)))
# df_wDBA     .to_csv(f"data/augedPaperMachine/df_wDBA_{datestr}.csv",index=False,header=None)

# df_discriminative_dtw_warp     =pd.DataFrame(np.array(np.squeeze(aug.discriminative_dtw_warp(x_train,y_train),axis=2)))
# df_discriminative_dtw_warp     .to_csv(f"data/augedPaperMachine/df_discriminative_guided_warp_{datestr}.csv",index=False,header=None)
# #
# df_discriminative_shape_dtw_warp     =pd.DataFrame(np.array(np.squeeze(aug.discriminative_shape_dtw_warp(x_train,y_train),axis=2)))
# df_discriminative_shape_dtw_warp     =pd.concat([pd.DataFrame(np.where(y_train==0,1,0).astype(int)),df_discriminative_shape_dtw_warp],axis=1)
# df_discriminative_shape_dtw_warp     .to_csv(f"data/augedPaperMachine/df_discriminative_shape_dtw_warp_{datestr}.csv",index=False,header=None)
#
#
# df_SPAWNER     =pd.DataFrame(np.array(np.squeeze(aug.spawner(x_train,y_train),axis=2)))
# df_SPAWNER     .to_csv(f"data/augedPaperMachine/df_SPAWNER_{datestr}.csv",index=False,header=None)