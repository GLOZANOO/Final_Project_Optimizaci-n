#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 20:02:35 2022

@author: karenmanguart
"""

# %% Import libraries

import numpy as np
import pandas as pd
import datetime as dt
import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor

# %%Load data_frame

df=pd.read_csv("Desktop/MCD/Optimizacion/Proyecto/corn-prices")
df=df.dropna()
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df=df.set_index(df['date'], drop=False, append=False, inplace=False, verify_integrity=False).drop('date', 1)


# %% Time Series Data Preparation

    #Scale data frame to be in the range (0, 1)
scaler = MinMaxScaler()
df['value'] = scaler.fit_transform(df)

    #Convert to numpy array
df_data= df.values    
    #Create data with time-steps (transform the input data to be of the form [batch, timesteps])
        #take timesteps = 5
timesteps=5
    #Converting training and test data to 2D tensor using nested list comprehension:
df_data_timesteps=np.array([[j for j in df_data[i:i+timesteps]] for i in range(0,len(df_data)-timesteps+1)])[:,:,0]
    #Split x and y
x, y= df_data_timesteps[:,:timesteps-1],df_data_timesteps[:,[timesteps-1]]
x_raw=x.copy()

    
# %% Symbolic Transformer

    #Scale x to be in the range (0, 1)
x= scaler.fit_transform(x)
x_raw= scaler.inverse_transform(x)
NUM = 15796 #Row number of df
est = Ridge()
est.fit(x[:NUM, :], y[:NUM])
function_set = ['add', 'sub', 'mul', 'div',
				'sqrt', 'log', 'abs', 'neg', 'inv',
				'max', 'min', 'sin', 'cos', 'tan']
gp = SymbolicTransformer(generations=20, population_size=2000,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=0, n_jobs=3,
                         metric='spearman')
gp.fit(x[:NUM, :], y[:NUM])
gp_features = gp.transform(x)
x = np.hstack((x, gp_features))

# %% Split x_train, y_train, x_test, y_test 70:30

x_train= x[0:11057,:]
y_train=y[0:11057,:] 
x_test= x[11057:15796,:]
y_test=y[11057:15796,:] 

# %% Train Linear Regression Model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

 
print(regressor.intercept_)
print(regressor.coef_)
print(f'r_sqr value: {regressor.score(x_train, y_train)}')

#%% Predictions
y_pred= regressor.predict(x_test)

#%% Metrics
def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('MAPE for testing data: ', mape(y_test, y_pred), '%')



