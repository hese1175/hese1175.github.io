---
title: Principal Component Analysis <b>(PCA)</b>
parent: Models/Methods
nav_order: 1
---

# Principal Component Analysis (PCA)
{: .no_toc }


1. TOC
{:toc}

## Overview

PCA is a dimensionality reduction technique that is widely used in machine learning and data analysis. It is used to reduce the number of features in a dataset while retaining as much information as possible. PCA works by transforming the original features into a new set of orthogonal features called principal components. These components are ordered by the amount of variance they explain in the data. The first principal component explains the most variance, the second principal component explains the second most variance, and so on. By selecting only the top principal components, you can reduce the dimensionality of the dataset while retaining most of the information.

![PCA OG](/assets/imgs/pca/PCA_original.png)
*Scatter plot showing Principal components V1 and V2*

![PCA OG](/assets/imgs/pca/PCA_transformed2.png)
*Scatter plot showing a transformed axis using the orthogonal principal component vectors V1 and V2*

In this project, PCA is used to reduce the dimensionality of the dataset and identify the most important features that explain the variance in the data. There are several columns in the dataset that are highly correlated, and PCA can help identify these relationships and reduce the dimensionality of the dataset. By reducing the number of features, the model will be less prone to overfitting and will be more interpretable. 

For example, the Materials and Resins columns are highly correlated. Similarly, the Resin Type and Resin Name columns are highly correlated. PCA will help identify these relationships and reduce the dimensionality of the dataset.

References:\
[1]: https://statisticsbyjim.com/basics/principal-component-analysis/ \
[2]: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html 

## Data

The data has already been preprocessed and cleaned using the processes shown in the EDA tab. This dataset contains 180 rows and 8 columns. 

Here is the first 10 rows of the dataset before PCA:

```plaintext

    Vf, %  Max. Stress, MPa  Freq., Hz  E, GPa  Max. % Strain  Cycles  Resin Type Encoded Resin Type
7  50.000           943.000      0.025 126.000          0.720   1.000                   0      Epoxy
8  49.000           910.000      0.025 125.000          0.710   1.000                   0      Epoxy
9  51.000           966.000      0.025 128.000          0.730   1.000                   0      Epoxy
10 52.000           967.000      0.025 131.000          0.710   1.000                   0      Epoxy
11 50.000          1066.000      0.025 125.000          0.810   1.000                   0      Epoxy
12 51.000           972.000      0.025 130.000          0.690   1.000                   0      Epoxy
13 52.000          1067.000      0.025 131.000          0.780   1.000                   0      Epoxy
14 51.000           939.000      0.025 134.000          0.710   1.000                   0      Epoxy
15 49.000           926.000      0.025 124.000          0.720   1.000                   0      Epoxy
16 47.000          1035.000      0.025 119.000          0.830   1.000                   0      Epoxy

<class 'pandas.core.frame.DataFrame'>
Index: 180 entries, 7 to 1059
Data columns (total 8 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   Vf, %               180 non-null    float64
 1   Max. Stress, MPa    180 non-null    float64
 2   Freq., Hz           180 non-null    float64
 3   E, GPa              180 non-null    float64
 4   Max. % Strain       180 non-null    float64
 5   Cycles              180 non-null    float64
 6   Resin Type Encoded  180 non-null    int64  
 7   Resin Type          180 non-null    object 
dtypes: float64(6), int64(1), object(1)
memory usage: 12.7+ KB
```
First the data was standardized using StandardScaler:

![PCA OG](/assets/imgs/pca/PCA_1.png)
*Table showing the first 10 rows of the dataset after standardization*


Then, PCA was applied to the dataset. For n_components=2, the dataset was transformed into 2 principal components. Here is the first 10 rows of the dataset after PCA:

Resin Type is the target variable here.

```plaintext
     pc1    pc2 Resin Type
0 -2.189 -0.355      Epoxy
1 -2.254 -0.525      Epoxy
2 -2.139 -0.201      Epoxy
3 -2.123 -0.092      Epoxy
4 -2.119 -0.072      Epoxy
5 -2.194 -0.174      Epoxy
6 -2.079  0.144      Epoxy
7 -2.238 -0.220      Epoxy
8 -2.233 -0.495      Epoxy
9 -2.238 -0.443      Epoxy
```

## Code

The code can be accesses on the GitHub repository using the link below:

[PCA Code](/assets/code/pca.py)

## Results

### n_components = 2
Here is the plot when PCA is applied to the dataset with n_components=2:
![PCA 2 componenet](/assets/imgs/pca/PCA_2-component.png)

The explained variance ratio for the first two principal components is:

```plaintext
Explained Variance Ratio: 
[0.42561841 0.23724278]
```
That means, in the 2D dataset, the cumulative explained variance ratio is 0.663. This means that the first two principal components explain 66.3% of the variance in the data.
```plaintext
Cumulative Explained Variance Ratio: 
[0.42561841 0.66286118]
```

### n_components = 3
Here is the plot when PCA is applied to the dataset with n_components=3:
![PCA 3 componenet](/assets/imgs/pca/PCA_3-component.png)


The explained variance ratio for the first two principal components is:

```plaintext
Explained Variance Ratio: 
[0.42561841 0.23724278 0.16677749]
```
That means, in the 3D dataset, the cumulative explained variance ratio is 0.829. This means that the first three principal components explain 82.9% of the variance in the data.

```plaintext
Cumulative Explained Variance Ratio: 
[0.42561841 0.66286118 0.82963867]
```

To get to 95% explained variance, more calculations were done. 

### n_components = 4

```plaintext
Explained Variance Ratio: 
[0.42561841 0.23724278 0.16677749 0.11197639]
Cumulative Explained Variance Ratio: 
[0.42561841 0.66286118 0.82963867 0.94161506]
```

### n_components = 5

```plaintext
Explained Variance Ratio: 
[0.42561841 0.23724278 0.16677749 0.11197639 0.0528304 ]
Cumulative Explained Variance Ratio:
[0.42561841 0.66286118 0.82963867 0.94161506 0.99444546]
```

So while n_components=4 comes close to 95% explained variance, n_components=5 is first to cross the 95% threshold.

The top three eigenvalues are:

```plaintext
Top 3 eigenvalues for n_components = 5: 
[2.56797697 1.43140894 1.00625523]
```

The plot showing the explained variance ratio for the first 5 principal components is also shown below:

![PCA 5 componenet](/assets/imgs/pca/PCA_5-component.png)