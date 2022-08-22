# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 07:36:29 2022

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

m = 100
x =  6 * np.random.rand(m, 1) - 3
y = 0.5 * x ** 2 + x + 2 + np.random.randn(m, 1)

plt.figure()
plt.plot(x, y, 'b.')
plt.ylabel('$x_1$', fontsize=18)
plt.xlabel('$y$', fontsize=18, rotation=0)
plt.axis([-3, 3, 0 ,10])
#plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=100, include_bias=(False))

x_poly = poly.fit_transform(x)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

x_new = np.linspace(-3, 3, 100).reshape(100, 1)
x_new_poly = poly.fit_transform(x_new)
y_new = lin_reg.predict(x_new_poly)

plt.plot(x_new, y_new, 'r-', linewidth=2, label='predictions')
plt.legend(loc='upper left', fontsize=18)
plt.show()

y_pred = lin_reg.predict(x_poly)

from sklearn.metrics import r2_score
print(r2_score(y, y_pred))