import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from matplotlib.colors import ListedColormap

#import of all the models Models
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# instance of all the models
classifier = {
    'KNN': KNeighborsClassifier(3),
    'SVM': SVC(gamma=2, C=1),
    'GP' : GaussianProcessClassifier(10*RBF(1.0)),
    'DT' : DecisionTreeClassifier(max_depth=5),
    'MLP' : MLPClassifier(alpha=0.1, max_iter=1000),
    'Bayes' : GaussianNB()}


#Create the dataset
x, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)

rng = np.random.RandomState(2)
x += 1 * rng.uniform(size=x.shape)
linearly_separable = (x, y)


datasets = [make_moons(noise=0.1), make_circles(noise=0.1, factor=0.5), linearly_separable]

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

#creating example figure
model_name = 'MLP'

figure = plt.figure(figsize=(9,3))
h = 0.02 #step
i = 1

for ds_cnt, ds in enumerate(datasets):
    x, y = ds
    x = StandardScaler().fit_transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y)
    
    #boundaries and granularidad? ralated to the graphic
    x_min, x_max = x[:,0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:,0].min() - 0.5, x[:, 0].max() + 0.5
    xx , yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    #Calling the model and train it
    model = classifier[model_name]
    model.fit(xtrain, ytrain)
    score_train = model.score(xtrain, ytrain)
    score_test   = model.score(xtest, ytest)
    
    #create the graphic
    ax = plt.subplot(1,3, i)
    if hasattr(model, 'decision_function'):
        zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        zz = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
    zz = zz.reshape(xx.shape)
    ax.contourf(xx, yy,zz, cmap=cm, alpha=0.8)
    
    ax.scatter(xtrain[:,0], xtrain[:,1], c = ytrain, cmap = cm_bright, edgecolors='k')
    ax.scatter(xtest[:,0], xtest[:,1], c = ytest, cmap = cm_bright, edgecolors='k', alpha=0.6)
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    
    ax.text(xx.max() - 0.3, yy.min() + 0.7, '%.2f' % score_train, size=15, horizontalalignment = 'right')
    ax.text(xx.max() - 0.3, yy.min() + 0.3, '%.2f' % score_test, size=15, horizontalalignment = 'right')
    i += 1
    
plt.tight_layout()
plt.show()