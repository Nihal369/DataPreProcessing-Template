# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:51:37 2017

@author: nihal369
"""
#Importing the librarires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset=pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Adding missing values
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(x[:, 1:3])
x[:,1:3]=imputer.transform(x[:, 1:3])

#Catogerising data(Convert all text to int)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_x = LabelEncoder()
x[:,0]=labelEncoder_x.fit_transform(x[:,0]);
oneHotEncoder=OneHotEncoder(categorical_features=[0])
x=oneHotEncoder.fit_transform(x).toarray()
labelEncoder_y = LabelEncoder()
y=labelEncoder_y.fit_transform(y)

#Making of Training and Test sets
from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
xtrain=scaler.fit_transform(xtrain)
xtest=scaler.transform(xtest)


