---
title: Clustering
parent: Models/Methods
nav_order: 2
---

# Clustering
{: .no_toc }


1. TOC
{:toc}

## Overview
Clustering is an unsupervised machine learning technique used to group similar data points together. The goal of clustering is to identify patterns and structures in the data that are not explicitly labeled. Clustering algorithms partition the data into groups based on the similarity of data points, with the objective of maximizing the intra-cluster similarity and minimizing the inter-cluster similarity.

Some key concepts in clustering include:
- **Centroid**: The center of a cluster, which is calculated as the mean of all data points in the cluster.
- **Distance Metric**: A measure of similarity between data points, which is used to determine which points belong to the same cluster.

In this project, clustering will be used to identify materials of wind turbine blades with similar properties. The clustering algorithm will group together items with similar material composition, and performance characteristics. This information can be used to identify important features of materials to consider and will help drop the less important features.

There are several clustering algorithms available, each with its own strengths and weaknesses. Some common clustering algorithms used in this project include:
- **K-Means**: A partitioning algorithm that divides the data into K clusters by iteratively assigning data points to the nearest cluster centroid and updating the centroids based on the mean of the data points in each cluster.
![K-Means Clustering](/assets/imgs/clustering/k-means.png)
- **Hierarchical Clustering**: A hierarchical algorithm that builds a tree of clusters by iteratively merging or splitting clusters based on the similarity of data points.
![Hierarchical Clustering](/assets/imgs/clustering/hierarchical.png)
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: A density-based algorithm that groups together data points that are closely packed and separates outliers as noise.
![DBSCAN Clustering](/assets/imgs/clustering/dbscan.jpeg)



References:\
[1]: https://keytodatascience.com/k-means-clustering-algorithm/ \
[2]: https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68 \
[3]: https://www.geeksforgeeks.org/dbscan-clustering-in-ml-density-based-clustering/

## Data

The data is prepped in a similar fashion to the PCA model. Please refer to the PCA page for more information. 

The n_components = 3 is used to reduce the dimensionality of the dataset to 3 principal components. 

```plaintext
PCA Data n=3 (first 10 rows):
       pc1    pc2    pc3   Resin Type
0   -2.189 -0.355  0.009        Epoxy
1   -2.254 -0.525  0.004        Epoxy
2   -2.139 -0.201  0.014        Epoxy
3   -2.123 -0.092  0.018        Epoxy
4   -2.119 -0.072  0.013        Epoxy
5   -2.194 -0.174  0.014        Epoxy
6   -2.079  0.144  0.022        Epoxy
7   -2.238 -0.220  0.014        Epoxy
8   -2.233 -0.495  0.004        Epoxy
9   -2.238 -0.443 -0.000        Epoxy

Cumulative Explained Variance Ratio: 
[0.42561841 0.66286118 0.82963867]
```
So the first 3 principal components explain 82.96% of the variance in the data.

This data will be used to perform all clustering algorithms.

## Code

The code is provided in the GitHub repository using the link below:

[Clustering Code](/assets/code/clustering.py)

## Results

### K-Means Clustering

The K-Means algorithm was applied to the dataset with n_clusters=[2,3,4]. Here is the plot of the clusters:

![K-Means Clustering](/assets/imgs/clustering/k-means_result.png)

The plots show the original data points colored by their true labels and the centroids of the clusters for each value of k. By comparing these plots, you can see how the clustering changes with different values of k.\
k=2: This clustering seems to separate the data into two distinct groups, but it might not capture the full complexity of the data.\
k=3: This clustering appears to identify three clusters that  might better represent the underlying structure of the data.\
k=4: This clustering results in four clusters, but it might be overfitting the data, as the clusters seem to fit some outliers.

In this case, k=3 seems to provide a good balance between capturing the structure of the data and avoiding overfitting.

### Hierarchical Clustering

The Hierarchical Clustering algorithm was applied to the dataset. Here is the dendrogram of the clusters:

![Hierarchical Clustering](/assets/imgs/clustering/hierarchical_results.png)

The dendrogram shows the hierarchical structure of the clusters, with the data points at the bottom and the clusters at the top. We can see that in this case, the small cluster (orange) is not considered an outlier, but rather a separate cluster. This might be due to the hierarchical nature of the algorithm, which allows for more flexibility in defining clusters. Otherwise, the results match the K-Means clustering with k=3 without the need to specify the number of clusters.

### DBSCAN Clustering

The DBSCAN algorithm was applied to the dataset with eps=0.5 and min_samples=10. Here is the plot of the clusters:

![DBSCAN Clustering](/assets/imgs/clustering/dbscan_eps05.png)

The plot shows the original data points colored by their true labels and the clusters identified by DBSCAN. The algorithm seems to identify three main clusters and some outliers. The clusters are well-separated and capture the structure of the data well. The outliers are labeled as noise in black, which is a useful feature of this method.

DBSCAN was also applied with eps=0.8 and min_samples=10. Here is the plot of the clusters:

![DBSCAN Clustering](/assets/imgs/clustering/dbscan_eps08.png)

We still notice three main clusters, but the outliers are now included in the clusters. This might be due to the larger value of eps, which allows for more points to be considered part of the same cluster. 

For this case, DBSCAN with eps=0.5 seems to provide a good balance between capturing the structure of the data and avoiding overfitting.

To compare the results of the clustering algorithms, we can use metrics such as the silhouette score, which measures the quality of the clusters. The silhouette score ranges from -1 to 1, with higher values indicating better clustering. The silhouette score for each algorithm is as follows:

```plaintext
K-means (k=3) Silhouette Score:             0.567
Hierarchical Clustering Silhouette Score:   0.667
DBSCAN Silhouette Score (eps=0.5):          0.532
```
So, in this case, Hierarchical Clustering seems to provide the best clustering results based on the silhouette score.