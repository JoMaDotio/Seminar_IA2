import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

img =  mpimg.imread('../../data/gato.jpg')
img = np.array(img, dtype = np.float64) / 255

w, h, d = img.shape
n_clusters = 10
img_array = np.reshape(img, (w*h, d))

image_sample = shuffle(img_array)[:1000]

kmeans = KMeans(n_clusters=n_clusters).fit(image_sample)

labels = kmeans.predict(img_array)


img_labels = np.reshape(labels, (w, h))
print(img_labels.shape)
img_out = np.zeros((w,h,d))
for i in range (w):
    for j in range(h):
        img_out[i][j][:] = kmeans.cluster_centers_[img_labels[i][j]][:]
        
        
plt.figure()
plt.title("Original")
plt.imshow(img)
plt.figure()
plt.title('Cuantizada')
plt.imshow(img_out)