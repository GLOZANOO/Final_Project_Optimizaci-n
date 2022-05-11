#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 22:14:27 2022

@author: karenmanguart
"""

# %% Import libraries
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import math
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler


# %% Load data_frame

df=pd.read_csv("Desktop/MCD/Optimizacion/Proyecto/corn-prices")
df=df.dropna()
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df2=df.set_index(df['date'], drop=False, append=False, inplace=False, verify_integrity=False).drop('date', 1)
    #Allocate train and test start periods
train_start_dt = '1959-07-01 00:00:00'
test_start_dt = '2003-05-30 00:00:00'
    #Visualize differences
df2[(df2.index < test_start_dt) & (df2.index >= train_start_dt)][['value']].rename(columns={'value':'train'}) \
    .join(df2[test_start_dt:][['value']].rename(columns={'value':'test'}), how='outer') \
    .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('value', fontsize=12)
plt.show()


#%% Prepare data fro training

train = df2.copy()[(df2.index >= train_start_dt) & (df2.index < test_start_dt)][['value']]
test = df2.copy()[df2.index >= test_start_dt][['value']]
    #Scale the training data to be in the range (0, 1): 
scaler = MinMaxScaler()
train['value'] = scaler.fit_transform(train)    
test['value'] = scaler.transform(test)
    #Create data with time-steps (transform the input data to be of the form [batch, timesteps])
    #Converting to numpy arrays
train_data = train.values
test_data = test.values
    #take timesteps = 5
timesteps=5
        #Converting training and test data to 2D tensor using nested list comprehension:
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
    #Selecting inputs and outputs from training and testing data:
x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]


#%% Implement SVR

model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
    #Fit the model on training data
model.fit(x_train, y_train[:,0])
    #Make model prediction
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)


#%% Evaluate model
    # Scaling the predictions
y_test_pred = scaler.inverse_transform(y_test_pred)
    # Scaling the original values
y_test = scaler.inverse_transform(y_test)
    #Check model performance on training and testing data
train_timestamps = df2[(df2.index < test_start_dt) & (df2.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = df2[test_start_dt:].index[timesteps-1:]
    #Plot the predictions for testing data
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()

#%% Metrics

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

print('MAPE for testing data: ', mape(y_test, y_test_pred), '%')
    

