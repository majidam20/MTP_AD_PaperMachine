from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import MAX_PROTOTYPES_PER_CLASS
from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
from utils.constants import DTW_PARAMS

from utils.utils import read_all_datasets
from utils.utils import calculate_metrics
from utils.utils import transform_labels
from utils.utils import create_directory
from utils.utils import plot_pairwise
import os
from augment import augment_train_set
import pathlib
import sys
from pathlib import Path
import pandas as pd
import os
#sys.path.append("..")
import numpy as np
import time

import random

# os.environ['PYTHONHASHSEED'] = '0'
# np.random.seed(42)
# random.seed(12345)

def augment_function(augment_algorithm_name, x_train, y_train, classes, N, limit_N=True):
    if augment_algorithm_name == 'as_dtw_dba_augment':
        return augment_train_set(x_train, y_train, classes, N,limit_N = limit_N,
                                 weights_method_name='as', distance_algorithm='dtw'), 'dtw'

def read_data_from_dataset(use_init_clusters=True):
    dataset_out_dir = root_dir_output + archive_name + '/' + dataset_name + '/'

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    # make the min to zero of labels
    y_train, y_test = transform_labels(y_train, y_test)

    classes, classes_counts = np.unique(y_train, return_counts=True)

    if len(x_train.shape) == 2:  # if univariate 
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # maximum number of prototypes which is the minimum count of a class
    max_prototypes = min(classes_counts.max() + 1,
                         MAX_PROTOTYPES_PER_CLASS + 1)
    init_clusters = None

    return x_train, y_train, x_test, y_test, nb_classes, classes, max_prototypes, init_clusters

# for mesocentre
##### you should change these for your directories
#'/b/home/uha/hfawaz-datas/dba-python/'
p1=Path.cwd()
p1=str(p1).replace(os.sep, '/')
root_dir = str(p1)+"/"
root_dir_output = str(root_dir) + 'results/'
root_deep_learning_dir = str(root_dir) +'dl-tsc/'
root_dir_dataset_archive = str(root_dir) +'dl-tsc/archives/'

# make sure before doing data augmentation to have the models trained without data augmentation
# in order to use the same weights init method

do_data_augmentation = True
#do_data_augmentation = False
#do_ensemble = True
do_ensemble = False

if do_ensemble:
    root_dir_output = root_deep_learning_dir + 'results/ensemble/'
else:
    if do_data_augmentation:
        root_dir_output = root_deep_learning_dir + 'results/resnet_augment/'
    else:
        root_dir_output = root_deep_learning_dir + 'results/resnet/'


from resnet import Classifier_RESNET

# loop the archive names
for archive_name in ARCHIVE_NAMES:
    # read all the datasets
    datasets_dict = read_all_datasets(root_dir_dataset_archive, archive_name)
    # loop through all the dataset names
    for dataset_name in DATASET_NAMES:
        print('dataset_name: ', dataset_name)
        # read dataset
        x_train, y_train, x_test, y_test, nb_classes, classes, max_prototypes, \
        init_clusters = read_data_from_dataset(use_init_clusters=False)

        # specify the output directory for this experiment
        output_dir = root_dir_output + archive_name + '/' + dataset_name + '/'

        x_train = np.concatenate((x_train, x_test), axis=0)
        y_train = np.concatenate((y_train, y_test), axis=0)

        _, classes_counts = np.unique(y_train, return_counts=True)
        # this means that all classes will have a number of time series equal to
        # nb_prototypes
        nb_prototypes = classes_counts.max()
        #nb_prototypes=4
        temp = output_dir
        # create the directory if not exists
        output_dir = create_directory(output_dir)
        # check if directory already exists
        # if output_dir is None:
        #     print('Already_done')
            #print(temp)
            #continue

        # if do_ensemble==False:
        #     # create the resnet classifier
        #     classifier = Classifier_RESNET(output_dir, x_train.shape[1:],
        #                                    nb_classes, nb_prototypes, classes,
        #                                    verbose=True, load_init_weights=do_data_augmentation)
        if do_data_augmentation:
                    # augment the dataset
                        datestr = time.strftime("%H%M%S")
                        print(f"Start Time: {datestr}")
                        syn_train_set, distance_algorithm = augment_function('as_dtw_dba_augment',
                                                                             x_train, y_train, classes,
                                                                             nb_prototypes,limit_N=False)
                        # get the synthetic train and labels
                        syn_x_train, syn_y_train = syn_train_set

                        # concat the synthetic with the reduced random train and labels
                        #aug_x_train = np.array(x_train.tolist() + syn_x_train.tolist())
                        #aug_y_train = np.array(y_train.tolist() + syn_y_train.tolist())

                        # aug_x_train =  syn_x_train
                        # aug_y_train = syn_y_train
##############################################
                        #a=np.array(syn_y_train.tolist()+syn_x_train.tolist())
                        x = np.squeeze(np.array(syn_x_train.tolist()), axis=2)
                        y=np.squeeze(np.array(np.reshape(syn_y_train,(syn_y_train.shape[0],1,1)).tolist(),dtype=int) , axis=2)
                        #b=np.squeeze(a, axis=2)
                        yx=np.concatenate((y, x), axis=1)
                        dfauged=pd.DataFrame(yx)
                        datestr = time.strftime("%y%m%d_%H%M%S")
                        dfauged.to_csv((f"dfAuged_{datestr}_DTW_PARAMS_{DTW_PARAMS['w']}.csv"), index=False,header=None)
                        datestr = time.strftime("%H%M%S")
                        print(f"End Time: {datestr}")
                        # print("np.unique(y_train): ",np.unique(y_train,return_counts=True))
                        # print("np.unique(aug_y_train): ",np.unique(aug_y_train,return_counts=True))

                        #y_pred = classifier.fit(aug_x_train, aug_y_train, x_test, y_test)
            # else:
            #     # no data augmentation
            #     y_pred = classifier.fit(x_train, y_train, x_test,
            #                         y_test)

            #do_ensemble==False
            # df_metrics = calculate_metrics(y_test, y_pred, 0.0)
            # df_metrics.to_csv(output_dir + 'df_metrics.csv', index=False)
            # print('DONE')
            # create_directory(output_dir+'DONE')

        #do_ensemble == True
        # else:
        #     # for ensemble you will have to compute both models in order to ensemble them
        #     from ensemble import Classifier_ENSEMBLE
        #     classifier_ensemble = Classifier_ENSEMBLE(output_dir, x_train.shape[1:], nb_classes, False)
        #     classifier_ensemble.fit(x_test, y_test)


# plot pairwise once all results are computed for resnet and resnet_augment and ensemble
#plot_pairwise(root_deep_learning_dir,root_dir_dataset_archive, 'resnet', 'resnet_augment')