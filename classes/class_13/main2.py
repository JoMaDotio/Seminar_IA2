import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture, metrics
from sklearn.preprocessing import StandardScaler


#creating datasets to compare
np.random.seed(0)
n_samples = 1500
X = 6 * [None]

#concentric circles
xtemp, _ = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
X[0] = StandardScaler().fit_transform(xtemp)

#moons
xtemp, _ = datasets.make_moons(n_samples=n_samples, noise=0.05)
X[1] = StandardScaler().fit_transform(xtemp)

#blobs
xtemp, _ = datasets.make_blobs(n_samples=n_samples, random_state=(8))
X[2] = StandardScaler().fit_transform(xtemp)

#non-grouping plane
xtemp = np.random.rand(n_samples, 2)
X[3] = StandardScaler().fit_transform(xtemp)

#concentric circles
xtemp, _ = datasets.make_blobs(n_samples=n_samples, random_state=170)
xtemp = np.dot(xtemp, [[0.6, -0.6],[-0.4, 0.8]])
X[4] = StandardScaler().fit_transform(xtemp)

#concentric circles
xtemp, _ = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5])
X[5] = StandardScaler().fit_transform(xtemp)


clusters = [2,2,3,3,3,3]

eps = [0.3,0.3,0.3,0.3, 0.15, 0.18]
y = []
for c, x, e in zip(clusters, X,eps):
    #model = cluster.Kmeans(n_clusters=c)
    #model = cluster.Birch(n_clusters=c)
    #model = cluster.SpectralClustering(n_clusters=c, affinity='nearest_neighbors')
    #model = mixture.GaussianMixture(n_components=c, covariance_type='full')
    #model = cluster.OPTICS(min_samples=20, xi=0.05, min_cluster_size=0.1)
    model = cluster.DBSCAN(eps=e)
    model.fit(x)
    if (hasattr(model, 'labels_')):
        y.append(model.labels_.astype(np.int))
    else:
        y.append(model.predict(x))




plt.figure(figsize=(27, 9))
plt.suptitle('DBSCAN', fontsize=18)
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    ax.scatter(X[i][:,0], X[i][:,1], c=y[i])