---
title: Decision Trees <b>(DT)</b>
parent: Models/Methods
nav_order: 4
---

# Decision Trees (DT)
{: .no_toc }

1. TOC
{:toc}

## Overview

Decision Trees are a popular machine learning algorithm that is used for classification and regression tasks. A Decision Tree is a tree-like structure that is used to make decisions based on the features of the data. The tree is built by recursively splitting the data into subsets based on the features that best separate the classes. The goal of the Decision Tree algorithm is to create a tree that is able to make accurate predictions on new data. 

The main criteria for splitting the data are the Gini impurity, the Information Gain and Entropy. The Gini impurity is a measure of how often a randomly chosen element would be incorrectly classified. The Information Gain is a measure of how much information a feature provides about the class label. Lastly, Entropy is a measure of the randomness in the data.

A small example to show how GINI impurity is calculated is shown below:

In equation form, the Gini impurity is calculated as follows:
$
Gini(D) = 1 - \sum_{i=1}^{n} p_i^2
$   
For instance, if we have a dataset with 3 classes and the class distribution is as follows:

$
Class\ 1: 10
$

$
Class\ 2: 20
$

$
Class\ 3: 30
$

The Gini impurity is calculated as follows:

$
Gini(D) = 1 - (10/60)^2 - (20/60)^2 - (30/60)^2
$

$
Gini(D) = 1 - (1/36) - (4/36) - (9/36)
$

$
Gini(D) = 1 - 14/36
$

$
Gini(D) = 22/36
$

$
Gini(D) = 0.6111
$

The Gini impurity ranges from 0 to 1, where 0 indicates that the data is perfectly classified and 1 indicates that the data is completely random. The goal of the Decision Tree algorithm is to minimize the Gini impurity at each node of the tree.

For decision trees, it is possible to create infinite trees that perfectly fit the training data. However, this can lead to overfitting, where the model performs well on the training data but poorly on new data. To prevent overfitting, the Decision Tree algorithm uses pruning techniques to limit the size of the tree. Pruning involves removing nodes from the tree that do not improve the performance of the model on the validation data.

References:\
[1]: https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052 \
[2]: https://scikit-learn.org/stable/modules/tree.html

## Data

Data is processed in a similar manner to that of the Naive Bayes model. The categorical data is encoded using the `LabelEncoder` class from the `sklearn.preprocessing` module. The data is then split into training and testing sets using the `train_test_split` function from the `sklearn.model_selection` module. The Decision Tree model is then trained on the training data and evaluated on the testing data.

The training and testing data is split into 75% training and 25% testing. This must be disjoint and random which the `train_test_split` function does by default. Refer to the data section of Naive Bayes model [here](nb.md#data) to see more details. Additionally, the data is not scaled as Decision Trees are not sensitive to the scale of the features.

## Code
The code can be accesses on the GitHub repository using the link below:

[DT Code](/assets/code/dt.py)

## Results

Three different Decision Tree models were trained on the dataset with different hyperparameters. The criterion is varied between Gini, Entropy and Log loss. The other hyperparameters like the maximum depth of the tree and the minimum number of samples required to split a node, are kept on the default setting. Additionally, splitter was set to random. While a random splitter is not particularly useful for small datasets, it can be useful and efficient for larger datasets.

Overall, the hyperparameters were varied to see how they affect the performance of the model. The results of the three models are shown below. 

#### Model 1: Gini, max_depth=None, min_samples_split=2

The first model was trained with the Gini criterion, a maximum depth of None, and a minimum number of samples required to split a node of 2. The model was evaluated on the testing data and the results are shown below:

```plaintext
Decision Tree Classifier Results:
Accuracy: 0.8888888888888888

Classification Report:
              precision    recall  f1-score   support

           0       0.94      0.97      0.96        33
           1       0.00      0.00      0.00         3
           2       1.00      1.00      1.00         1
           3       0.70      0.88      0.78         8

    accuracy                           0.89        45
   macro avg       0.66      0.71      0.68        45
weighted avg       0.84      0.89      0.86        45
```
And the feature importance are:
```plaintext
Feature Importance:
            feature  importance
0             Vf, %       0.330
7           Process       0.259
1  Max. Stress, MPa       0.192
2            E, GPa       0.096
4            Lay-up       0.062
5      0 Deg fabric       0.039
6  Cure / Post Cure       0.022
3     Max. % Strain       0.000
```
![DT 1](/assets/imgs/dt/dt_1.png)

#### Model 2: Entropy, max_depth=None, min_samples_split=2

The second model was trained with the Entropy criterion, a maximum depth of None, and a minimum number of samples required to split a node of 2. The model was evaluated on the testing data and the results are shown below:

```plaintext
Decision Tree Classifier Results:
Accuracy: 0.9111111111111111

Classification Report:
              precision    recall  f1-score   support

           0       0.91      0.97      0.94        33
           1       0.50      0.33      0.40         3
           2       1.00      1.00      1.00         1
           3       1.00      0.88      0.93         8

    accuracy                           0.91        45
   macro avg       0.85      0.79      0.82        45
weighted avg       0.90      0.91      0.91        45
```
And the feature importance are:
```plaintext
Feature Importance:
            feature  importance
6  Cure / Post Cure       0.355
7           Process       0.269
0             Vf, %       0.099
5      0 Deg fabric       0.081
3     Max. % Strain       0.067
2            E, GPa       0.063
4            Lay-up       0.048
1  Max. Stress, MPa       0.019
```
![DT 2](/assets/imgs/dt/dt_2.png)

#### Model 3: Log Loss, max_depth=None, min_samples_split=2

The third model was trained with the Log Loss criterion, a maximum depth of None, and a minimum number of samples required to split a node of 2. The model was evaluated on the testing data and the results are shown below:

```plaintext
Decision Tree Classifier Results:
Accuracy: 0.9111111111111111

Classification Report:
              precision    recall  f1-score   support

           0       0.92      1.00      0.96        33
           1       1.00      0.67      0.80         3
           2       1.00      1.00      1.00         1
           3       0.83      0.62      0.71         8

    accuracy                           0.91        45
   macro avg       0.94      0.82      0.87        45
weighted avg       0.91      0.91      0.90        45
```
And the feature importance are:
```plaintext
Feature Importance:
            feature  importance
7           Process       0.318
3     Max. % Strain       0.223
5      0 Deg fabric       0.110
0             Vf, %       0.109
6  Cure / Post Cure       0.084
1  Max. Stress, MPa       0.069
2            E, GPa       0.053
4            Lay-up       0.035
```
![DT 3](/assets/imgs/dt/dt_3.png)

#### Model 4: Gini, max_depth=4, min_samples_split=5

The fourth model was trained with the Gini criterion, a maximum depth of 4, and a minimum number of samples required to split a node of 5. The model was evaluated on the testing data and the results are shown below:

```plaintext
Decision Tree Classifier Results:
Accuracy: 0.8

Classification Report:
              precision    recall  f1-score   support

           0       0.79      1.00      0.88        33
           1       0.00      0.00      0.00         3
           2       1.00      1.00      1.00         1
           3       1.00      0.25      0.40         8

    accuracy                           0.80        45
   macro avg       0.70      0.56      0.57        45
weighted avg       0.78      0.80      0.74        45
```
And the feature importance are:
```plaintext
Feature Importance:
            feature  importance
7           Process       0.421
5      0 Deg fabric       0.209
0             Vf, %       0.148
4            Lay-up       0.117
2            E, GPa       0.106
1  Max. Stress, MPa       0.000
3     Max. % Strain       0.000
6  Cure / Post Cure       0.000
```
![DT 4](/assets/imgs/dt/dt_4.png)

### Conclusion

The Decision Tree model was trained on the dataset with different hyperparameters and evaluated on the testing data. The results show that the model performs well on the testing data with an accuracy of around 0.9. The feature importance of the model shows that the most important features are the Vf, %, Max. Stress, MPa, and Process. The results show that the Decision Tree model is able to accurately classify the data and provide insights into the importance of the features.

Even with the fourth model, which has a limited depth of 4, the accuracy is still 0.8. This shows that the Decision Tree model is able to perform well even with a limited depth. The feature importance of the fourth model shows that the Process is the most important features.

Overall, the Decision Tree model has been useful to show the importance of the features in the dataset and to classify the data accurately. The processing method of the material is deemed very important and that's helpful when making new materials.