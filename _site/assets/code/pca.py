#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:12:24 2024

@author: hemantsethi
"""

###### PCA 

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder


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

### PCA using Sklearn

## n = 2
myPCA_2 = PCA(n_components=2)
pca_results_2 = myPCA_2.fit_transform(pca_df)

pca_results_2 = pd.DataFrame(data = pca_results_2
             , columns = ['pc1', 'pc2'])

pca_results_2 = pd.concat([pca_results_2, target_pca], axis = 1)

# Print the explained variance ratio for each component
print(myPCA_2.explained_variance_ratio_)

# Calculate the cumulative sum
cumulative_sum_2 = myPCA_2.explained_variance_ratio_.cumsum()
print(cumulative_sum_2)


if plot:
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    
    
    targets = list(le_name_mapping.keys())
    colors = ['r', 'g', 'b', 'y']
    for target, color in zip(targets,colors):
        mask = pca_results_2['Resin Type'] == target
        ax.scatter(pca_results_2.loc[mask, 'pc1']
                   , pca_results_2.loc[mask, 'pc2']
                   , c = color
                   , s = 50
                   )    
    ax.legend(targets)
    ax.grid()



## n = 3
myPCA_3 = PCA(n_components=3)
pca_results_3 = myPCA_3.fit_transform(pca_df)

pca_results_3 = pd.DataFrame(data = pca_results_3
             , columns = ['pc1', 'pc2', 'pc3'])

pca_results_3 = pd.concat([pca_results_3, target_pca], axis = 1)


# Print the explained variance ratio for each component
print(myPCA_3.explained_variance_ratio_)

# Calculate the cumulative sum
cumulative_sum_3 = myPCA_3.explained_variance_ratio_.cumsum()
print(cumulative_sum_3)


if plot:
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    ax.set_title('3 component PCA', fontsize=20)

    targets = list(le_name_mapping.keys())
    colors = ['r', 'g', 'b', 'y']
    for target, color in zip(targets, colors):
        mask = pca_results_3['Resin Type'] == target
        ax.scatter(pca_results_3.loc[mask, 'pc1'],
                   pca_results_3.loc[mask, 'pc2'],
                   pca_results_3.loc[mask, 'pc3'],
                   c=color,
                   s=50)

    ax.legend(targets)
    ax.grid(True)
    ax.view_init(elev=45, azim=-45)
    plt.show()
    



## n = 4
myPCA_4 = PCA(n_components=4)
pca_results_4 = myPCA_4.fit_transform(pca_df)

pca_results_4 = pd.DataFrame(data = pca_results_4
             , columns = ['pc1', 'pc2', 'pc3', 'pc4'])

pca_results_4 = pd.concat([pca_results_4, target_pca], axis = 1)


# Print the explained variance ratio for each component
print(myPCA_4.explained_variance_ratio_)

# Calculate the cumulative sum
cumulative_sum_4 = myPCA_4.explained_variance_ratio_.cumsum()
print(cumulative_sum_4)


## n = 5
myPCA_5 = PCA(n_components=5)
pca_results_5 = myPCA_5.fit_transform(pca_df)

pca_results_5 = pd.DataFrame(data = pca_results_5
             , columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5'])

pca_results_5 = pd.concat([pca_results_5, target_pca], axis = 1)


# Print the explained variance ratio for each component
print(myPCA_5.explained_variance_ratio_)

# Calculate the cumulative sum
cumulative_sum_5 = myPCA_5.explained_variance_ratio_.cumsum()
print(cumulative_sum_5)

# Top 3 eigenvalues for n=5
eig_5 = myPCA_5.explained_variance_
print('Top 3 eigenvalues for n_components = 5: ')
print(eig_5[0:3])