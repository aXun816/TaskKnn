import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier

#生成数据
centers=[[-2,2],[2,2],[0,4]]
X, y = make_blobs(n_samples=60, centers=centers, random_state=0, cluster_std=0.60)

array=([1, 0, 0, 1, 0, 1, 1, 0, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1,
       2, 0, 0, 1, 2, 0, 0, 2, 2, 2, 1, 2, 1, 2, 2, 1, 0, 1, 1, 0, 1, 2,
       1, 1, 0, 1, 0, 2, 2, 1, 1, 2, 2, 0, 1, 2, 0, 1])

# 画出数据
#plt.figure(figsize=(16,10), dpi=144)
c=np.array(centers)
print(c)
#plt.scatter(X[:, 0], X[:, 1], c=y,s=100, cmap='cool')         # 画出样本
#X[:, 0]所有点的x轴坐标， X[:, 1]所有点y轴坐标。s点的大小，c是一个数组类别0，1，2按类别作色
#plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='orange')   # 画出中心点

clf=KNeighborsClassifier(n_neighbors=10)
clf.fit(X,y)

# 进行预测
X_sample = np.array([[0, 2]])
y_sample = clf.predict(X_sample)
# y_sample
neighbors=clf.kneighbors(X_sample, return_distance=False)
print(neighbors)
print(neighbors[0])

# 画出示意图
plt.figure(figsize=(16,10), dpi=144)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')    # 样本
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='k')   # 中心点
plt.scatter(X_sample[0][0], X_sample[0][1], marker="x",
            c='r', s=200, cmap='cool')    # 待预测的点

for i in neighbors[0]:
    plt.plot([X[i][0], X_sample[0][0]], [X[i][1], X_sample[0][1]],
             '-.', linewidth=0.6)    # 预测点与距离最近的 10 个样本的连线
#[X[i][0], X_sample[0][0]] , x坐标
# [X[i][1], X_sample[0][1]]，y坐标


plt.show()