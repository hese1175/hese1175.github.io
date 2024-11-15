---
title: Regression
parent: Models/Methods
nav_order: 7
---

# Regression
{: .no_toc }


1. TOC
{:toc}

## Overview

Regression is a supervised machine learning technique that is used to predict the value of a continuous target variable based on the values of one or more predictor variables. The goal of regression is to find the best-fitting line or curve that describes the relationship between the target variable and the predictor variables. There are several types of regression models, including linear regression, polynomial regression, and logistic regression. Each type of regression model has its own strengths and weaknesses, and the choice of model depends on the nature of the data and the problem you are trying to solve.

(a) Define and explain linear regression

Linear regression is a simple and widely used regression technique that is used to model the relationship between a continuous target variable and one or more predictor variables. The goal of linear regression is to find the best-fitting line that describes the relationship between the target variable and the predictor variables. The line is defined by the equation:

$$
y = mx + b
$$

where $y$ is the target variable, $x$ is the predictor variable, $m$ is the slope of the line, and $b$ is the y-intercept. The slope and y-intercept are determined by minimizing the sum of the squared differences between the observed values of the target variable and the values predicted by the line.

(b) Define and explain logistic regression

Logistic regression is a regression technique that is used to model the relationship between a binary target variable and one or more predictor variables. The goal of logistic regression is to find the best-fitting line that describes the relationship between the target variable and the predictor variables. The line is defined by the equation:

$$
y = \frac{1}{1 + e^{-z}}
$$

where $y$ is the target variable, $z$ is the linear combination of the predictor variables, and $e$ is the base of the natural logarithm. The line is fit by maximizing the likelihood of the observed data given the parameters of the model.

(c) How are they similar and how are they different?

Linear regression and logistic regression are similar in that they are both regression techniques that are used to model the relationship between a target variable and one or more predictor variables. 

However, they are different in the type of target variable they are used to model. Linear regression is used to model continuous target variables, while logistic regression is used to model binary target variables. Additionally, the equations that define the lines in linear regression and logistic regression are different. In linear regression, the line is defined by a simple linear equation, while in logistic regression, the line is defined by a sigmoidal function.

(d) Does logistic regression use the Sigmoid function? Explain.

Yes, logistic regression uses the Sigmoid function to model the relationship between the predictor variables and the target variable. The Sigmoid function is defined by the equation:

$$
y = \frac{1}{1 + e^{-z}}
$$

where $y$ is the target variable, $z$ is the linear combination of the predictor variables, and $e$ is the base of the natural logarithm. The Sigmoid function maps the linear combination of the predictor variables to a value between 0 and 1, which can be interpreted as the probability that the target variable is equal to 1.

(e) Explain how maximum likelihood is connected to logistic regression.

Maximum likelihood is a statistical method that is used to estimate the parameters of a model by maximizing the likelihood of the observed data given the parameters of the model. In logistic regression, the likelihood of the observed data given the parameters of the model is defined by the product of the probabilities of the observed outcomes. The goal of logistic regression is to find the parameters of the model that maximize the likelihood of the observed data. This is done by iteratively updating the parameters of the model until the likelihood of the observed data is maximized.

## Data

The data has already been preprocessed and cleaned using the processes shown in the EDA tab. This dataset only contains the numerical columns that are used to predict the target variable `Resin Type`.

Additionally, the target variable was reduced to two classes: `Epoxy` and `Vinylester`.

Here is a quick description of the dataset before regression along with the first 10 rows:

```plaintext
Index: 153 entries, 7 to 1059
Data columns (total 5 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   Vf, %             153 non-null    float64
 1   Max. Stress, MPa  153 non-null    float64
 2   E, GPa            153 non-null    float64
 3   Max. % Strain     153 non-null    float64
 4   Resin Type        153 non-null    object 
dtypes: float64(4), object(1)
memory usage: 7.2+ KB

First 10 rows:
--------------
    Vf, %  Max. Stress, MPa  E, GPa  Max. % Strain Resin Type
7  50.000           943.000 126.000          0.720      Epoxy
8  49.000           910.000 125.000          0.710      Epoxy
9  51.000           966.000 128.000          0.730      Epoxy
10 52.000           967.000 131.000          0.710      Epoxy
11 50.000          1066.000 125.000          0.810      Epoxy
12 51.000           972.000 130.000          0.690      Epoxy
13 52.000          1067.000 131.000          0.780      Epoxy
14 51.000           939.000 134.000          0.710      Epoxy
15 49.000           926.000 124.000          0.720      Epoxy
16 47.000          1035.000 119.000          0.830      Epoxy
```

The data was scaled using the `StandardScaler` class from the `sklearn.preprocessing` module. The data was then split into training and testing sets using the `train_test_split` function from the `sklearn.model_selection` module. The training and testing data was split into 75% training and 25% testing. The data was then used to train and evaluate the regression models.

## Code

The code can be accessed on the GitHub repository using the link below:

[Regression Code](/assets/code/regression.py)

## Results

Logistic regression results are shown below. The dataset was split into training and testing data with a test size of 0.2. The model was trained on the training data and evaluated on the testing data. The results are shown below.

### Model 1: Logistic Regression

```plaintext
Logistic Regression Results:
Accuracy: 0.8461538461538461

Classification Report:
              precision    recall  f1-score   support

       Epoxy       0.85      1.00      0.92        33
  Vinylester       0.00      0.00      0.00         6

    accuracy                           0.85        39
   macro avg       0.42      0.50      0.46        39
weighted avg       0.72      0.85      0.78        39
```

The confusion matrix is shown below:
![Logistic Regression Confusion Matrix](/assets/imgs/lr/cm_lr.png)

This result was compared to that of the Naive Bayes model on the same dataset. 

### Model 2: Naive Bayes

```plaintext
Multinomial Naive Bayes Results:
Accuracy: 0.6666666666666666

Classification Report:
              precision    recall  f1-score   support

       Epoxy       1.00      0.61      0.75        33
  Vinylester       0.32      1.00      0.48         6

    accuracy                           0.67        39
   macro avg       0.66      0.80      0.62        39
weighted avg       0.89      0.67      0.71        39
```

The confusion matrix is shown below:
![Naive Bayes Confusion Matrix](/assets/imgs/lr/cm_nb.png)

## Conclusion

The logistic regression model performed better than the Naive Bayes model on the dataset. The logistic regression model had an accuracy of 0.85, while the Naive Bayes model had an accuracy of 0.67. This suggests that the logistic regression model is better at predicting the resin type of the composite material based on the input features. The logistic regression model can be used to predict the resin type of composite materials with an accuracy of 0.85. The model can be further improved by tuning hyperparameters and using more advanced techniques like feature engineering and feature selection.

However, the logistic regression model is the best model to use in this case. The False positives and True negatives are significant. This is largely due to ignoring the categorical data in the dataset, small size of the dataset as well as the characteristics of the target materials chosen. Unfortunately, the numerical differences in the target materials are not significantly different to make accurate predictions. 