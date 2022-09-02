## K Means Clustering Algorithm

Machine learning algorithms are categorized into three main categories 

 1. Supervised learning
 2. Unsupervised learning
 3. Reinforcement learning

Up till now, we have looked into supervised learning, where in the given data set we have a class label or target variable. In unsupervised learning, all you have is a set of features we don't know about our target variable or a class label. Using this data set, we try to identify the underlying structure in that data Or we try to find the clusters in that data so that we can make useful predictions out of it.

## Introduction to K-Means Algorithm

The K-means clustering algorithm computes centroids and repeats until the optimal centroid is found. It is presumptively known how many clusters there are. It is also known as the flat clustering algorithm. The number of clusters found from data by the method is denoted by the letter ‘K’ in K-means. 

In this method, data points are assigned to clusters in such a way that the sum of the squared distances between the data points and the centroid is as small as possible. It is essential to note that reduced diversity within clusters leads to more identical data points within the same cluster.

## Implementation of K Means Clustering Graphical Form

 1. STEP 1: Let us pick k clusters, i.e., K=2, to separate the dataset and assign it to its appropriate clusters. We will select two random places to function as the cluster’s centroid.
 
 2. STEP 2: Now, each data point will be assigned to a scatter plot depending on its distance from the nearest K-point or centroid. This will be accomplished by establishing a median between both centroids. Consider the following illustration:

 3. STEP 3: The points on the line’s left side are close to the red centroid, while the points on the line’s right side are close to the green centroid. The left Form cluster has a red centroid, whereas the right Form cluster has a green centroid.
 ![enter image description here](https://raw.githubusercontent.com/jibintom/Machine-Learning-Codebasics-/main/a13.%20K%20Means%20Clustering%20Algorithm/Images/STEP-3.png)
 4.  STEP 4: Repeat the procedure, this time selecting a different centroid. To choose the new centroids, we will determine their new center of gravity, which is represented below: 
 
 5. STEP 5: After that, we’ll re-assign each data point to its new centroid. We shall repeat the procedure outlined before (using a median line). The red cluster will contain the green data point on the red side of the median line
 ![enter image description here](https://raw.githubusercontent.com/jibintom/Machine-Learning-Codebasics-/main/a13.%20K%20Means%20Clustering%20Algorithm/Images/STEP-5.png)

6. STEP 6: Now that reassignment has occurred, we will repeat the previous step of locating new centroids.

7. STEP 7: Recompute the clusters and repeat this till data points stop changing clusters

![enter image description here](https://raw.githubusercontent.com/jibintom/Machine-Learning-Codebasics-/main/a13.%20K%20Means%20Clustering%20Algorithm/Images/STEP-7.png)

8. STEP-8 So our final Cluster is as follows:

![enter image description here](https://raw.githubusercontent.com/jibintom/Machine-Learning-Codebasics-/main/a13.%20K%20Means%20Clustering%20Algorithm/Images/final.png)
 

## Elbow Algorithm

In reality, we will have so many features, and it is hard to visualize it data on a scatter plot. So how do we find the number of k? The Elbow method is used to solve this problem

In the Elbow method, we are actually varying the number of clusters (K) from 1 – 10. For each value of K, we are calculating WCSS ( Within-Cluster Sum of Square ). WCSS is the sum of the squared distance between each point and the centroid in a cluster. 

![enter image description here](https://raw.githubusercontent.com/jibintom/Machine-Learning-Codebasics-/main/a13.%20K%20Means%20Clustering%20Algorithm/Images/elbow%20method.png)

When we plot the WCSS with the K value, the plot looks like an Elbow. As the number of clusters increases, the WCSS value will start to decrease. WCSS value is largest when K = 1. When we analyze the graph we can see that the graph will rapidly change at a point and thus creating an elbow shape. From this point, the graph starts to move almost parallel to the X-axis. The K value corresponding to this point is the optimal K value or an optimal number of clusters.

![K-Means Clustering cluster numbers](https://editor.analyticsvidhya.com/uploads/43191elbow_img%20(1).png)

