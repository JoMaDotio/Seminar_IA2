# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 07:36:29 2022

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

np.random.seed(42)

m = 100
x =  6 * np.random.rand(m, 1) - 3
y = 0.5 * x ** 2 + x + 2 + np.random.randn(m, 1)

poly = PolynomialFeatures(degree=300, include_bias=(False))

lin_reg = LinearRegression()

std_scaler = StandardScaler()

model = Pipeline([('poly_features', poly),('scaler', std_scaler),('lin_reg', lin_reg)])

model.fit(x, y)
y_pred = model.predict(x)

#lin_reg.fit(x_poly, y)

x_new = np.linspace(-3, 3, 100).reshape(100, 1)
#x_new_poly = poly.fit_transform(x_new)
y_new = model.predict(x_new)

plt.figure()
plt.plot(x, y, 'b.')
plt.ylabel('$x_1$', fontsize=18)
plt.xlabel('$y$', fontsize=18, rotation=0)
plt.axis([-3, 3, 0 ,10])
plt.plot(x_new, y_new, 'r-', linewidth=2, label='predictions')
plt.legend(loc='upper left', fontsize=18)
plt.show()



print(r2_score(y, y_pred))