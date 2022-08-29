from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)
m = 200
x = 3 * np.random.rand(m,1)
y = 1 + 0.5 * x + np.random.randn(m, 1) / 1.5

x_new = np.linspace(0,3,100).reshape(100,1)

x_train, x_test, y_train, y_test = train_test_split(x,y)

model = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),
                  ('scaler', StandardScaler()),
                  ('reg_lin', Ridge(alpha=0))])

model.fit(x_train, y_train)
print(f'Train R2: {model.score(x_train, y_train)}')
print(f'Train R2: {model.score(x_test, y_test)}')


plt.figure()
#lt.plot(x,y,'ob')
plt.plot(x_test, y_test, 'r.')
plt.plot(x_train, y_train, 'b.')
plt.plot(x_new, model.predict(x_new), '-k')