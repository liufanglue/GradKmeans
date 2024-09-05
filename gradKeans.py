import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import *

isNeedDebug = True

def kmeans_plus_plus_init(X, k):
    n_points, n_dims = X.shape
    centroids = torch.zeros(k, n_dims, device = X.device)
    centroids[0] = X[torch.randint(0, n_points, (1, ))]
    for i in range(1, k):
        dist = torch.cdist(X, centroids[:i]).min(dim = 1)[0]
        probs = dist / dist.sum()
        cumprobs = torch.cumsum(probs, dim = 0)
        r = torch.rand(1)
        centroids[i] = X[torch.searchsorted(cumprobs, r)]
    
    return centroids

def kmeans_optimized(X, k, max_iters = 10, lr = 1e-3, tol = 1e-4):
    n_points, n_dims = X.shape

    initial_centroids = kmeans_plus_plus_init(X, k)
    C = torch.zeros(n_points, k, device = X.device)
    C.data = F.softmax(-(torch.cdist(X, initial_centroids) ** 2), dim = 1)
    C.requires_grad = True

    optimizer = torch.optim.Adam([C], lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.9)
    
    prev_loss = float('inf')
    for iteration in range(max_iters):
        optimizer.zero_grad()
        CX = C.unsqueeze(-1) * X.unsqueeze(1)
        centroids = torch.sum(CX, dim = 0) / torch.sum(C, dim = 0).unsqueeze(1)
        diff = X.unsqueeze(1) - centroids.unsqueeze(0)
        mask = C.unsqueeze(-1)
        masked_diff = diff * mask
        distances_squared = torch.sum(masked_diff ** 2, dim = (1, 2))
        

        reg_term = 1e-3 * torch.sum(torch.sqrt(torch.sum(C ** 2, dim = 0)))
        loss = torch.sum(distances_squared) + reg_term
        #loss = torch.sum(distances_squared)

        if abs(prev_loss - loss.item()) < tol:
            print(f"Converged at iteration {iteration}")
            break
        
        prev_loss = loss.item()
        #print(f"Iteration {iteration}, Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        #temperature = max(0.1, 1 - iteration / max_iters)
        #with torch.no_grad():
        #    C.data = F.softmax(C.data / temperature, dim=1)
    labels = torch.argmax(C, dim = 1)
    final_centroids = torch.sum(C.unsqueeze(-1) * X.unsqueeze(1), dim = 0) / torch.sum(C, dim = 0).unsqueeze(1)
    
    return labels, final_centroids

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
labels, centroids = kmeans_optimized(X, k)
endTime = datetime.now()
print(f"time = {endTime - startTime}")

labels = labels.detach().cpu().numpy()
centroids = centroids.detach().cpu().numpy()
X = X.detach().cpu().numpy()

fig = plt.figure(figsize=(10, 8))
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
plt.title('3D K-means Clustering (Optimized)')
plt.show()

