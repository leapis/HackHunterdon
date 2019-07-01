from numpy import array
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
# Create 50 datapoints in two clusters a and b
pts = 500
a = np.random.multivariate_normal([0, 0], [[4, 1], [1, 4]], size=pts)
b = np.random.multivariate_normal([30, 10],
                                           [[10, 2], [2, 1]],
                                           size=pts)
c = np.random.multivariate_normal([0,0],[[3,1],[1,3]],size=pts)
features = np.concatenate((a, b, c))
# Whiten data
whitened = whiten(features)
# Find 2 clusters in the data
codebook, distortion = kmeans(whitened, 3)
cb2, distortion2 = kmeans(whitened,2)
cb3, distortion3 = kmeans(whitened, 4)
print(distortion, distortion2, distortion3)
# Plot whitened data and cluster centers in red
plt.scatter(whitened[:, 0], whitened[:, 1])
plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
plt.show()
