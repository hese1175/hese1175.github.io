---
title: Ensemble Learning
parent: Models/Methods
nav_order: 8
---

# Ensemble Learning
{: .no_toc }


1. TOC
{:toc}

## Overview

Ensemble learning is a machine learning paradigm where multiple models (often called "weak learners") are trained to solve the same problem and combined to get better results. The main hypothesis is that when weak models are correctly combined, they can outperform a single powerful model. The most common ensemble methods are bagging, boosting, and stacking. 

In this project, we will focus on the Gradient Tree Boosting method. Gradient Tree Boosting is an ensemble learning method that builds a series of decision trees, where each tree corrects the errors of the previous one. The final model is an ensemble of these weak learners. Gradient Tree Boosting is a powerful algorithm that can be used for both regression and classification problems. We will use it to predict the modulus of the samples.

![Gradient Tree Boosting](/assets/imgs/ensemble/1.png)

The main idea behind Gradient Tree Boosting is to combine multiple decision trees to create a strong learner. Each tree is built sequentially, where each new tree tries to correct the errors of the previous trees. The final model is an ensemble of these weak learners. The algorithm works by minimizing the loss function, which is the difference between the predicted value and the actual value. The loss function is minimized using gradient descent.

Reference:
1. Deng, Haowen & Zhou, Youyou & Wang, Lin & Zhang, Cheng. (2021). Ensemble learning for the early prediction of neonatal jaundice with genetic features. BMC Medical Informatics and Decision Making. 21. 10.1186/s12911-021-01701-9. 
2. https://scikit-learn.org/stable/modules/ensemble.html

## Data

The data is prepared in a similar fashion to the SVM model. [SVM Data Prep](https://hese1175.github.io/docs/CSCI%205622%20-%20ML/models/svm/#data)

## Code

The code can be accesses on the GitHub repository using the link below:

[Ensemble Code](/assets/code/ensemble.py)

## Results

The accuracy score for the model is:

```
Accuracy: 0.9555555555555556
```

The classification report for the model is:

```
Classification Report:
              precision    recall  f1-score   support

        HIGH       1.00      0.92      0.96        13
         LOW       0.97      0.97      0.97        29
         MED       0.75      1.00      0.86         3

    accuracy                           0.96        45
   macro avg       0.91      0.96      0.93        45
weighted avg       0.96      0.96      0.96        45
```

Finally, the confusion matrix for the model is:

![Confusion Matrix](/assets/imgs/ensemble/cm_ensemble.png)

The model performs well with an accuracy of 95.56%. The model is able to predict the modulus of the samples with high precision and recall. The confusion matrix shows that the model is able to correctly classify most of the samples. The model can be used to predict the modulus of new samples with high accuracy.