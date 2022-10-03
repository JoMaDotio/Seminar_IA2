import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


#projector.tensorflow.org

data = pd.read_csv('../../data/mnist_784.csv')
n_samples = 3000

x = np.asanyarray(data.drop(columns=['class']))[:n_samples]
y = np.asanyarray(data[['class']])[:n_samples]

model = TSNE(n_components=2, n_iter=2000)

x_2d = model.fit_transform(x)

plt.scatter(x_2d[:, 0], x_2d[:,1], c=y, cmap=plt.cm.tab10)