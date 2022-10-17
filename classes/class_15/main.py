import numpy as np
import pandas as pnd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler


data = pnd.read_csv('../../data/diabetes.csv')

x = np.asanyarray(data.iloc[:,:-1])
y = np.asanyarray(data.iloc[:,-1])

x = StandardScaler().fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y)

dt = DecisionTreeClassifier().fit(xtrain, ytrain)

print(f"Just the tree alone")
print(f"Train: {dt.score(xtrain, ytrain)}")
print(f"Test: {dt.score(xtest, ytest)}")

rf = RandomForestClassifier(n_estimators=100)
rf.fit(xtrain, ytrain)
print(f"Random forest")
print(f"Train: {rf.score(xtrain, ytrain)}")
print(f"Test: {rf.score(xtest, ytest)}")

bg = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=1.0, n_estimators=100).fit(xtrain, ytrain)
print(f"Bagging")
print(f"Train: {bg.score(xtrain, ytrain)}")
print(f"Test: {bg.score(xtest, ytest)}")

aboost = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=10, learning_rate=1).fit(xtrain, ytrain)
print(f"Ada boosting")
print(f"Train: {aboost.score(xtrain, ytrain)}")
print(f"Test: {aboost.score(xtest, ytest)}")

dt = DecisionTreeClassifier()
lr = LogisticRegression(solver = 'lbfgs', max_iter = 500)
svm = SVC(kernel='rbf', gamma='scale')
voting = VotingClassifier([('lr', lr), ('dt',dt), ('svm', svm)])

voting.fit(xtrain, ytrain)
print(f"Voting")
print(f"Train: {voting.score(xtrain, ytrain)}")
print(f"Test: {voting.score(xtest, ytest)}")