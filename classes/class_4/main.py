# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 07:14:11 2022

@author: usuario
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


data = pd.read_csv("../../data/home_data.csv")

y = np.asanyarray(data[['price']])
x = np.asanyarray(data.drop(['id', 'date', 'price'], axis=1))
names = np.asanyarray(data.drop(['id', 'date', 'price'], axis=1).columns)

std_scaler = StandardScaler()
lin_reg = LinearRegression()

#model = Pipeline([('scaler', std_scaler),('lin_reg', lin_reg)])
#beforeTrain = model['lin_reg']
#model.fit(x, y)
#afterTrain = model.named_steps['lin_reg'].coef_

std_scaler.fit(x)
newTest = std_scaler.transform(x)
lin_reg.fit(newTest, y)
print(r2_score(y, lin_reg.predict(newTest)))
coefs = abs(lin_reg.coef_)[0].reshape(18, 1)
orderByMagintud = np.sort(coefs[0])

results = pd.DataFrame()
results = results.assign(varNames=names, coef=coefs)
results = results.sort_values(by='coef', ascending=False)

print(results)

