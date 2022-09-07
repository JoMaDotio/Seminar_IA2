import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR

data = pd.read_csv('../../data/daily-min-temperatures.csv')

#x = np.asanyarray(data[['Temp']])

#plt.plot(x)

#p = 365
#plt.scatter(x[p:], x[:-p])
#plt.show()
#print(np.corrcoef(x[p:].T, x[:-p].T))

#from pandas.plotting import autocorrelation_plot
#autocorrelation_plot(data.Temp)

data2 = pd.DataFrame(data.Temp)

p = 5 
for i in range(1, p+1):
    data2 = pd.concat([data2, data.Temp.shift(-i)], axis=1)
    
data2 = data2[:-p]

x = np.asanyarray(data2.iloc[:,:-1])
y = np.asanyarray(data2.iloc[:,-1])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model = Pipeline([('scaler', StandardScaler()), ('Mamalona',MLPRegressor(hidden_layer_sizes=(100, 25, 1)))])
#model = Pipeline([('scaler', StandardScaler()), ('SVR',SVR(gamma='scale', C=1, epsilon=0.1, kernel='rbf'))])

model.fit(xtrain, ytrain)

print(f'Score: {model.score(xtrain, ytrain)}')
print(f'Score: {model.score(xtest, ytest)}')