# GradKmeans

1 Summarize:

   This code provides an implementation of the keans clustering algorithm based on gradient descent. Although due to the time 
consumption of memory allocation, the performance of the gradient descent algorithm is reduced compared with the original algorithm. 
However, it can be seen from the code implementation that gradient learning can be applied to more unsupervised learning scenarios. 
Therefore, supervised learning and unsupervised learning have the same nature and learning methods

2 Code file description

(1) In the file folder:
    The oriKeans.py is the original implementation of the keans clustering algorithm;
    The gradKeans.py is a gradient method implementation of the keans clustering algorithm.
(2) Because the code needs to continuously allocate the batch memory when implementing the clustering method with the gradient, 
    the speed advantage of using the gradient is offset by the time spent on the memory allocation. Therefore, the gradient algorithm 
    performance is not as good as the original algorithm