# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:51:37 2017

@author: nihal369
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(x[:, 1:3])
x[:,1:3]=imputer.transform(x[:, 1:3])

