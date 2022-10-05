import numpy as np
import pandas as pnd
import time
import warnings
from sklearn import metrics, preprocessing, tree
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split 
warnings.filterwarnings('ignore')

data = pnd.read_csv('../../data/loan_prediction.csv')

x = np.asanyarray(data.iloc[:,0:-1])
y = np.asanyarray(data.iloc[:,-1])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

model = tree.DecisionTreeClassifier()

#scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring='f1_macro')

# grid seach is brute force algorithm 
# random serach, is also a brute force algorithm but with a number off tries

hp = {'max_depth': [1,5,7,10,12],
      'min_samples_leaf':[2,5,10,15],
      'min_samples_split':[2,3,6,8,10],
      'criterion':['gini', 'entropy']}

search_obj = GridSearchCV(model, hp, cv=5, scoring='f1_macro')
fit_obj = search_obj.fit(xtrain, ytrain)
print(f"{fit_obj.cv_results_['mean_test_score']}")
best_model = fit_obj.best_estimator_

best_model.fit(xtrain, ytrain)
print(f"train: {best_model.score(xtrain, ytrain)}")
print(f"test: {best_model.score(xtest, ytest)}")