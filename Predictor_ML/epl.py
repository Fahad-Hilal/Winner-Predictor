# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 23:13:58 2018

@author: Fahad Hilal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas  as pd

#Importing the dataset

dataset=pd.read_csv('epldata_final.csv')
dataset.iat[188, 10] = 4 
X=dataset.iloc[:,[2,4,6,7,9,10,14,15,16]].values
Y=dataset.iloc[:,[5]].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1,5,6,7,8])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dumy trap

X=X[:,1:]

#Spliting the dataset into training and test sets

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=0)

#Fitting multiple linear regression to the training set

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)




#Building the optimal mody_predict=regressor.predict(X_test)el using backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)

X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

