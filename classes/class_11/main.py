import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

data = pd.read_csv('../../data/mnist_784.csv')

n_samples = 7000


x = np.asanyarray(data.drop(columns=['class']))[:n_samples,:]
y = np.asanyarray(data[['class']])[:n_samples].ravel()

"""

"""

xtrain, xtest, ytrain, ytest = train_test_split(x, y)

model = Pipeline([('Scaler', StandardScaler()), ('PCA', PCA(n_components=(50))), ('SVM', svm.SVC(gamma=0.0001))])

model.fit(xtrain, ytrain)

print(f'Train {model.score(xtrain, ytrain)}')
print(f'Test {model.score(xtest, ytest)}')

y_pred = model.predict(xtest)
print(f"Classification report\n{metrics.classification_report(ytest, y_pred)}")
print(f"Matriz de confusion\n{metrics.confusion_matrix(ytest, y_pred)}")

sample = np.random.randint(xtest.shape[0])
plt.imshow(x[sample].reshape(28,28), cmap=plt.cm.gray)
plt.title(f"Target {y[sample]}")
plt.show()

import pickle
pickle.dump(model, open("digit_class.sav", 'wb'))