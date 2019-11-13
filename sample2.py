from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
plt.scatter(X[:,0],X[:,1])
plt.show()
print(indices)
print(distances)
