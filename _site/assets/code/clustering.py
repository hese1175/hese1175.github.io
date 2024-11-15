#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:48:42 2024

@author: hemantsethi
"""

########### 
## CLUSTERING
###########

# Data prep similar to PCA 
#

# Label encode Resin Type 
label_encoder = LabelEncoder()
longitudinal_static_data['Resin Type Encoded'] = label_encoder.fit_transform(longitudinal_static_data['Resin Type'])

# Cleaning data to remove all categorical data
pca_df = longitudinal_static_data.select_dtypes(include=['int64', 'float64'])

# Temporarily add Resin Type column
pca_df = pd.concat([pca_df, longitudinal_static_data[['Resin Type']]], axis = 1)


le_name_mapping = dict(zip(label_encoder.classes_, 
                           label_encoder.transform(label_encoder.classes_)))

# Dropping compression columns
pca_df.drop(pca_df.columns[[2,6]],axis=1,inplace=True)
pca_df.info()

# Dropping columns with na values
pca_df = pca_df.dropna(axis=0, how='any')
pca_df.info()

# Extract target
target_pca = pd.DataFrame(data = pca_df['Resin Type'])
target_pca = target_pca.reset_index(drop=True)

# Remove target from df
pca_df.drop(pca_df.columns[[7]],axis=1,inplace=True)

# Scale data / Standardize
scaler = StandardScaler()
pca_df = scaler.fit_transform(pca_df)

### PCA using Sklearn for 3D dataset 

## n = 3
myPCA_3 = PCA(n_components=3)
pca_results_3 = myPCA_3.fit_transform(pca_df)

pca_results_3 = pd.DataFrame(data = pca_results_3
             , columns = ['pc1', 'pc2', 'pc3'])

pca_results_3 = pd.concat([pca_results_3, target_pca], axis = 1)



### K-Means

clustering_df = pca_results_3.copy()

features = clustering_df[['pc1', 'pc2', 'pc3']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Prepare three values of k to test
distances = []
silhouette_scores = []

k_values = [2, 3, 4]  # Three values of k based on the Silhouette Method
results = {}

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    labels = kmeans.labels_
    if k == 3:
        kmeans_labels = labels
    silhouette_avg = silhouette_score(scaled_data, labels)
    silhouette_scores.append(silhouette_avg)
    distances.append(kmeans.inertia_)
    results[k] = {
        'labels': labels,
        'centroids': kmeans.cluster_centers_
    }


if plot:
    # Prepare 3D plot for each k
    fig = plt.figure(figsize=(15, 5))
    
    for i, k in enumerate(k_values):
        ax = fig.add_subplot(1, len(k_values), i + 1, projection='3d')
        ax.set_title(f'KMeans Clustering with k={k}')
    
        # Scatter plot for the original data colored by true labels
        ax.scatter(scaled_data[:, 0], scaled_data[:, 1], scaled_data[:, 2], 
                   c=pca_results_3['Resin Type'].astype('category').cat.codes, 
                   label='Original Labels', marker='o', alpha=0.6)
    
        # Scatter plot for the cluster centroids
        centroids = results[k]['centroids']
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                   c='black', s=200, marker='X', label='Centroids')
        ax.legend()
        ax.view_init(elev=20, azim=-45)

    plt.tight_layout()
    plt.show()


########
# Hierarchial Clustering
########

from scipy.cluster.hierarchy import dendrogram, linkage

z = linkage(scaled_data, method='ward')

if plot:
    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(z)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()



########
# DBSCAN
########

from sklearn.cluster import DBSCAN

# Set parameters for DBSCAN

# eps = 0.5
dbscan = DBSCAN(eps=0.5, min_samples=10)

# Perform DBSCAN clustering
dbscan_labels = dbscan.fit_predict(scaled_data)

# Output the clustering results
print(dbscan_labels)

n_clusters_ = len(set(dbscan_labels)) - (1 if -1 in labels else 0)
n_noise_ = list(dbscan_labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


plot = True
if plot:
    # Plotting the DBSCAN clustering results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a color map for clusters
    unique_labels = set(dbscan_labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = [0, 0, 0]  # Black for noise points
        mask = (dbscan_labels == label)
        ax.scatter(scaled_data[mask][:, 0], 
                   scaled_data[mask][:, 1], 
                   scaled_data[mask][:, 2], 
                   c=[color], s=60)
    
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_zlabel('Principal Component 3', fontsize=12)
    ax.set_title('DBSCAN Clustering Results, EPS = 0.5', fontsize=15)
    plt.show()


# eps = 0.8
dbscan = DBSCAN(eps=0.8, min_samples=10)

# Perform DBSCAN clustering
dbscan_labels = dbscan.fit_predict(scaled_data)

# Output the clustering results
print(dbscan_labels)

n_clusters_ = len(set(dbscan_labels)) - (1 if -1 in labels else 0)
n_noise_ = list(dbscan_labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


plot = True
if plot:
    # Plotting the DBSCAN clustering results using 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a color map for clusters
    unique_labels = set(dbscan_labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = [0, 0, 0]  # Black for noise points
        mask = (dbscan_labels == label)
        ax.scatter(scaled_data[mask][:, 0], 
                   scaled_data[mask][:, 1], 
                   scaled_data[mask][:, 2], 
                   c=[color], s=60)
    
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_zlabel('Principal Component 3', fontsize=12)
    ax.set_title('DBSCAN Clustering Results, EPS = 0.8', fontsize=15)
    plt.show()

    
# Comparing results 

# Compare counts of clusters detected by each method
results_comparison = {
    'DBSCAN': np.unique(dbscan_labels, return_counts=True),
    'KMeans': np.unique(kmeans_labels, return_counts=True),
}

print(results_comparison)





