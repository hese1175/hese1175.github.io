---
title: Naive Bayes <b>(NB)</b>
parent: Models/Methods
nav_order: 5
---

# Naive Bayes (NB)
{: .no_toc }

1. TOC
{:toc}

## Overview

Naive Bayes is a classification algorithm that is based on Bayes' theorem. It is a simple and fast algorithm that is widely used in machine learning and data analysis. Naive Bayes is a probabilistic classifier that is based on the assumption that the features are conditionally independent given the class label. This assumption simplifies the computation of the posterior probability and makes the algorithm computationally efficient. Naive Bayes is often used in text classification, spam filtering, and other applications where the features are independent.

Naive Bayes comes in several different flavors, including Multinomial Naive Bayes, Gaussian Naive Bayes, and Categorical Naive Bayes. The choice of Naive Bayes model depends on the type of data you are working with. Multinomial Naive Bayes is used for discrete features, Gaussian Naive Bayes is used for continuous features, and Categorical Naive Bayes is used for categorical features. In the example below, we will use all three types of Naive Bayes models to classify the dataset. 

References:\
[1]: https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c \
[2]: https://scikit-learn.org/stable/modules/naive_bayes


## Data

The data has already been preprocessed and cleaned using the processes shown in the EDA tab. This dataset contains 180 rows and 10 columns.

Then the columns are divided into two: numerical and categorical. The numerical columns are scaled using the MinMaxScaler so that there are no negative values. The categorical columns are encoded using the LabelEncoder in order to run the model. The data is then split into training and testing sets using the train_test_split function from the sklearn library in a 75/25 split. 

The training data has 135 rows and 10 columns, while the testing data has 45 rows and 10 columns as shown below.

![NB Training Data](/assets/imgs/nb/X-train_df.png)
```plaintext
X_train
-------
X_train.shape: (135, 10)
<class 'pandas.core.frame.DataFrame'>
Index: 135 entries, 132 to 102
Data columns (total 10 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   Vf, %             135 non-null    float64
 1   Max. Stress, MPa  135 non-null    float64
 2   E, GPa            135 non-null    float64
 3   Max. % Strain     135 non-null    float64
 4   Material          135 non-null    float64
 5   Lay-up            135 non-null    float64
 6   Resin Name        135 non-null    float64
 7   0 Deg fabric      135 non-null    float64
 8   Cure / Post Cure  135 non-null    float64
 9   Process           135 non-null    float64
dtypes: float64(10)
memory usage: 11.6 KB
```
![NB Test Data](/assets/imgs/nb/X-test_df.png)

```plaintext
X_test
-------
X_test.shape: (45, 10)
<class 'pandas.core.frame.DataFrame'>
Index: 45 entries, 19 to 161
Data columns (total 10 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   Vf, %             45 non-null     float64
 1   Max. Stress, MPa  45 non-null     float64
 2   E, GPa            45 non-null     float64
 3   Max. % Strain     45 non-null     float64
 4   Material          45 non-null     float64
 5   Lay-up            45 non-null     float64
 6   Resin Name        45 non-null     float64
 7   0 Deg fabric      45 non-null     float64
 8   Cure / Post Cure  45 non-null     float64
 9   Process           45 non-null     float64
dtypes: float64(10)
memory usage: 3.9 KB
```



## Code

The code can be accesses on the GitHub repository using the link below:

[Naive Bayes Code](/assets/code/nb.py)

## Results

Three different Naive Bayes models were run on the dataset:  Multinomial Naive Bayes, Gaussian Naive Bayes, and Categorical Naive Bayes. The results are shown below. 

The correlation matrix for the dataset is shown below: 

![NB Correlation Matrix](/assets/imgs/nb/correlation_matrix.png)

It is important to plot the correlation matrix to understand the relationship between the features and the target variable. This will help in understanding the results of the Naive Bayes models.
### Multinomial Naive Bayes

The confusion matrix for the Multinomial Naive Bayes model is shown below:
![NB Multinomial Confusion Matrix](/assets/imgs/nb/cm-mnb.png)

The classification report for the Multinomial Naive Bayes model is shown below:
```plaintext
Multinomial Naive Bayes Results:
Accuracy: 0.8

Classification Report:
              precision    recall  f1-score   support

           0       0.79      1.00      0.88        33
           1       0.00      0.00      0.00         3
           2       0.00      0.00      0.00         1
           3       1.00      0.38      0.55         8

    accuracy                           0.80        45
   macro avg       0.45      0.34      0.36        45
weighted avg       0.75      0.80      0.74        45
```

### Gaussian Naive Bayes

Gaussian Naive Bayes is used when the features are continuous. The data prep was not altered from that of the Multinomial Naive Bayes model.

The confusion matrix for the Gaussian Naive Bayes model is shown below:
![NB Gaussian Confusion Matrix](/assets/imgs/nb/cm-gnb.png)

The classification report for the Gaussian Naive Bayes model is shown below:
```plaintext
Gaussian Naive Bayes Results:
Accuracy: 0.9333333333333333

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        33
           1       0.00      0.00      0.00         3
           2       1.00      1.00      1.00         1
           3       0.73      1.00      0.84         8

    accuracy                           0.93        45
   macro avg       0.68      0.75      0.71        45
weighted avg       0.88      0.93      0.91        45
```
### Categorical Naive Bayes

Finally, the Categorical Naive Bayes model was run on the dataset. The data prep method was slightly altered in order to run the model. While the categorical columns were encoded using the LabelEncoder, the numerical columns were not scaled. Instead, the numerical columns were binned into categories using the KBinsDiscretizer. The data was then split into training and testing sets using the train_test_split function from the sklearn library in a 75/25 split. 
The confusion matrix and classification report are shown below.

![NB Categorical Confusion Matrix](/assets/imgs/nb/cm-cnb.png)

```plaintext
Categorical Naive Bayes Results:
Accuracy: 0.9555555555555556

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        33
           1       1.00      0.33      0.50         3
           2       1.00      1.00      1.00         1
           3       0.80      1.00      0.89         8

    accuracy                           0.96        45
   macro avg       0.95      0.83      0.85        45
weighted avg       0.96      0.96      0.95        45
```

## Conclusion

In conclusion, the Categorical Naive Bayes model performed the best on the dataset with an accuracy of 95.6%. The Gaussian Naive Bayes model also performed well with an accuracy of 93.3%. The Multinomial Naive Bayes model performed the worst with an accuracy of 80%. The results show that the Categorical Naive Bayes model is the best choice for this dataset. This is likely due to the fact that the defining features of the dataset are categorical in nature, and the Categorical Naive Bayes model is able to handle this type of data effectively. 

The model was able to effectively classify the dataset into not two, but four target classes with high accuracy. This is a good result considering the small size of the dataset and the complexity of the features.