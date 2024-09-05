import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import *

isNeedDebug = False

def kmeans(X, k, max_iters = 100000):
    centroids = X[torch.randperm(X.shape[0])[ : k]]
    for _ in range(max_iters):
        distances = torch.cdist(X, centroids)
        labels = torch.argmin(distances, dim = 1)
        new_centroids = torch.stack([X[labels == i].mean(dim = 0) for i in range(k)])
        if torch.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    PrintTorch(labels, 'labels', True)
    PrintTorch(centroids, 'centroids', True)
    
    return labels, centroids

def PrintTorch(data, des = '', isNeedPrintData = False):
    if isNeedDebug:
        if (des == ''):
            if isNeedPrintData:
                print(f"data = {data}, shape = {data.shape}")
            else:
                print(f"shape = {data.shape}")
        else:
            if isNeedPrintData:
                print(f"des = {des}, data = {data}, shape = {data.shape}")
            else:
                print(f"des = {des}, shape = {data.shape}")

n_points = 1000
X = torch.randn(n_points, 3)

k = 2

startTime = datetime.now()
labels, centroids = kmeans(X, k)
endTime = datetime.now()
print(f"time = {endTime - startTime}")

fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot(111, projection = '3d')

colors = ['r', 'g', 'b']
for i in range(k):
    cluster_points = X[labels == i]
    ax.scatter(cluster_points[ : , 0], cluster_points[ : , 1], cluster_points[ : , 2], c = colors[i], label = f'Cluster {i+1}')

ax.scatter(centroids[ : , 0], centroids[ : , 1], centroids[ : , 2], c = 'black', s = 200, marker = '*', label = 'Centroids')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('3D K-means Clustering')
plt.show()

