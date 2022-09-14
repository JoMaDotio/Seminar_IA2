# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# data = pd.read_csv('../../data/diabetes.csv')
# #pd.plotting.scatter_matrix(data)
# #corr = data.corr()
# #sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

# x = np.asanyarray(data.drop(columns=['Outcome']))
# y = np.asanyarray(data[['Outcome']]).ravel()

# xtrain, xtest, ytrain, ytest = train_test_split(x, y)

# model = Pipeline([
#     ('scaler', StandardScaler()),
#     ('logist', LogisticRegression())])

# model.fit(xtrain, ytrain)

# print(f'Train {model.score(xtrain, ytrain)}')
# print(f'Train {model.score(xtest, ytest)}')

# coeff = list(np.abs(model.named_steps['logist'].coef_[0]))
# coeff = coeff / np.sum(coeff)
# labels = list(data.drop(columns=(['Outcome'])).columns)

# features = pd.DataFrame()
# features = features.assign(Features=labels, Coef=coeff)
# features.sort_values(by=['Coef'], ascending=True, inplace=True)
# features.set_index('Features', inplace=True)
# features.Coef.plot(kind='barh')
# plt.xlabel('Coeff')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

iris = datasets.load_iris()

x = iris['data'][:, (2, 3)]
y = iris['target']

plt.plot(x[y==0,0], x[y==0,1], 'g^', label='Iris-Setosa')
plt.plot(x[y==1,0], x[y==1,1], 'bs', label='bs-Versicolor')
plt.plot(x[y==2,0], x[y==2,1], 'yo', label='Iris-Virginica')

xtrain, xtest, ytrain, ytest = train_test_split(x, y)

soft_max = LogisticRegression(multi_class='multinomial')
soft_max.fit(xtrain, ytrain)

print(f'Train {soft_max.score(xtrain, ytrain)}')
print(f'Train {soft_max.score(xtest, ytest)}')

ypred = soft_max.predict(xtest)
print(f'Confusion matrix:\n {confusion_matrix(ytest, ypred)}')
print(f'Classification report: {classification_report(ytest, ypred)}')