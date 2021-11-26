# MTP_AD_PaperMachine
This Git repository demonstrate master thesis project on the topic: **Anomaly Detection and Forecasting for Time Series Sensor Data of a Paper Machine**

## Abstract
The use of sensors in our lives has been increasing in recent years, especially in industries.
Also, data mining of sensor data has become an important issue for humans. Hence, operators
install sensors on machines to record their data for monitoring and controlling devices.
Accordingly, predictive maintenance and anomaly detection of sensor data significantly reduce manufacturing crash rate and maintenance costs. This thesis
aims to detect and forecast anomalies for preventing breaks in the paper manufacturing
machine. To achieve this objective, we complete multiple time series pre-processing strategies
and develop some oversampling approaches for decreasing the adverse effects of the
corresponding imbalanced labeled dataset. Moreover, the Deep Neural Network (DNN)
and Long Short-Term Memory (LSTM) models are employed to perform our classification.
Our investigations reveal that the pre-processing and oversampling approaches increased
classification metric recall and imbalanced impact. However, other classification metrics such as precision and F1-score could not be improved due to faulty data gathering and
preparation of the paper machine.

## Problem Statement
A real-world dataset was produced by paper machine manufacturing. The connected multivariate
time-dependent sensors made the time series dataset. The dataset contains paper
breaks (anomalies) that commonly occur in the industry. Moreover, the dataset comprises
sensor readings at regular intervals (every two minutes), and the occurrence label
(Anomaly). During producing papers, sometimes the machine breaks. Therefore, if a
break happens, the entire process is stopped, the paper reel is taken out, then any found
problem(s) is fixed, and the production continues. The resumption can be taken more than
an hour. Hence, the cost of this lost production time is notable for the corresponding factory.
Even a 5% reduction in the break occurrences will provide a high-cost saving for the
factory (https://arxiv.org/abs/1809.10717) [1]. Furthermore, these uncommon events (breaks) create imbalanced time series
data that negatively affect the performances of the supervised DNN models. Therefore,
forming the dataset more balanced and forecasting anomalies in some timesteps in advance
are recognized as the most significant challenges.

## Conclusion
We aimed in this thesis to detect anomalous observations and forecast actual anomalies
for reducing the paper machine breaks. We worked on a multivariate time series sensor
dataset that consists of many normal and rare abnormal (anomalies) observations. The
strange occurrences (breaks) provided imbalanced data that negatively impacted supervised
model (DNN and LSTM) performances. Hence, making the dataset more balanced, forecasting
actual anomalies a few timesteps in advance, and improving the model prediction
achievements are the most important purposes.
Therefore, we presented a few beneficial time series data analyses to understand the dataset
better. Accordingly, we realized the difficulties of this project at the beginning stages when
the high correlations among the features and fewer relationships between the features and
corresponding labels were observed. We developed numerous time series pre-processing approaches
such as shifting, rolling, transforming, and shuffling to prepare the time-dependent
dataset for use in the DNN and LSTM models. Furthermore, our implementations could
provide different possibilities such as shift numbers, window lengths, new data generations,
model selections, and model empowerments.
We performed several plans to minimize the negative effects of imbalanced labeled data
on our classifiers via generating synthetic data by varying approaches such as the AE and
DTW. As a result, we generated sufficient augmented data with fewer differences compared
to the actual given data.
