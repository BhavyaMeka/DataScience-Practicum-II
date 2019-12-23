### Data Science Practicum II

# Implementing Deep neural network models for Human Activity Detection

# Project Details
The purpose of this project is to use deep neural networks for human activity detection. Six human activities were detected and logged manually using smartphone’s accelerometer and gyro meter. Usually with domain knowledge the main important features are sorted out for machine learning algorithms and classifying time series data. This DNN model extracts features and learns from raw data with more accuracy. This is a time series multi-class classification problem.

# Data
Human Activity Recognition Using Smartphones Data Set is available in UCI Machine Learning Repository which has details of the 30 subjects between age of 19-48 performing daily human actions. These movements were captured with smartphone’s accelerometer and gyro meter. This dataset contains activity label, time-frequency domain variables, triaxial acceleration, triaxial angular velocity and identifier of the subject. This separated into 70% of trained data and 30% of test data. Recordings of 30 study participants performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors. Six human activities were detected and logged manually using smartphone’s accelerometer and gyro meter.

# Project Execution
This is a multivariate, time series dataset without any missing values. I intend to perform exploratory data analysis (EDA) on dataset to get any interesting initial finds from data. Since it’s a deep neural network task I will use Convolution neural network (CNN) by adding several convolution layers, pooling layers and activations and also create a LSTM model. Model creation and python code were executed on Google colab with UCI datasets on google drive.

A 1D CNN is very effective for deriving features from a fixed-length segment of the overall dataset, where it is not so important where the feature is located in the segment.

# 1D CNN Model

![1D CNN model](/images/cnn.jpg)

LSTMs are quite popular in dealing with text based data, and has been quite successful in sentiment analysis, language translation and text generation. Since this problem also involves a sequence of similar sorts, an LSTM is a great candidate to be tried.

# LSTM Model

![LSTM model](/images/lstm.jpg)

# Summary
From the project it can be concluded that 1D CNN had better accuracy performance(%) than LSTM model (%). In the confusion matrix it can be observed that almost all the predictions were true. Since the predictions were made from small amount of data the accuracy can be further improved with large datasets. Improvement can done by tuning model hyperparameters such as the number of units, training epochs, batch size, and more. In the future hybrids of CNNs and LSTMS such as the CNN-LSTM and the ConvLSTM can be explored further in improving the accuracy.

# Confusion Matrix

![Confusion Matrix](/images/CM.jpg)



Data used for exploratory purpose
uci_raw_data: https://drive.google.com/open?id=18QdeKf_BgEEFQHDbTjQ8_YJJgQYeW9oA
