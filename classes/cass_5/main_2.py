
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("../../data/home_data.csv")
y = np.asanyarray(data[['price']])
x = np.asanyarray(data.drop(['id', 'date', 'price'], axis=1))
names = np.asanyarray(data.drop(['id', 'date', 'price'], axis=1).columns)
std_scaler = StandardScaler()
lin_reg = LinearRegression()
std_scaler.fit(x)
newTest = std_scaler.transform(x)
lin_reg.fit(newTest, y)
print(r2_score(y, lin_reg.predict(newTest)))
coefs = abs(lin_reg.coef_)[0].reshape(18, 1)
results = pd.DataFrame()
results = results.assign(varNames=names, coef=coefs)
results = results.sort_values(by='coef', ascending=False)
print(results)

model = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),('ridge', Ridge(alpha=1e-02))])

x_train, x_test, y_train, y_test = train_test_split(newTest,y)

model.fit(x_train, y_train)
print(f'Train R2: {model.score(x_train, y_train)}')
print(f'Train R2: {model.score(x_test, y_test)}')


x2 = np.asanyarray(data.drop(['id', 'date', 'price','floors', 'sqft_lot', 'yr_renovated'], axis=1))
std_scaler.fit(x2)
newTest2= std_scaler.transform(x2)
model2 = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),('ridge', ElasticNet(alpha=1e-02, max_iter=2000))])
x_train2, x_test2, y_train2, y_test2 = train_test_split(newTest2,y)

model2.fit(x_train2, y_train2)
print(f'Train R2: {model2.score(x_train2, y_train2)}')
print(f'Train R2: {model2.score(x_test2, y_test2)}')