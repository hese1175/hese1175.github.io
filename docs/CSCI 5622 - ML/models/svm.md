---
title: Support Vector Machines <b>(SVM)</b>
parent: Models/Methods
nav_order: 7
---

# Support Vector Machines (SVM)
{: .no_toc }


1. TOC
{:toc}

## Overview
Support Vector Machines or SVM is a supervised machine learning algorithm that can be used for both classification or regression challenges. However, it is mostly used in classification problems. In this algorithm, we plot each data item as a point in n-dimensional space (where n is the number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiates the two classes very well.

![SVM](/assets/imgs/svm/svm_1.png)

SVMs are linear separators and can be used for linearly separable data. However, in practice, most of the data is not linearly separable. In such cases, SVMs can use a kernel trick to transform the input space to a higher dimensional space where the data can be linearly separated. The most common kernels used are the linear, polynomial, and radial basis function (RBF) kernels.

This kernels work by transforming the input space into a higher dimensional space. The linear kernel is used when the data is linearly separable. The polynomial kernel is used when the data is not linearly separable. The RBF kernel is used when the data is not linearly separable and the polynomial kernel is not working well.

The polynomial kernel can be represented as:

$$ 
K(x, y) = (x \cdot y + c)^d
$$

where \(d\) is the degree of the polynomial, $(c)$ is the constant term, and $(x \cdot y)$ is the dot product of the two vectors.

The RBF kernel can be represented as:

$$
K(x, y) = \exp(-\gamma \lVert x - y \rVert^2)
$$

where \(\gamma\) is the kernel coefficient and $$ (\lVert x - y \rVert^2) $$ is the squared Euclidean distance between the two vectors.

![SVM](/assets/imgs/svm/svm_2.png)

To illustrate how a polynomial kernel transforms a 2D point into higher dimensions, consider a point (1 ,2) and use a polynomial kernel with parameters r = 1 and d = 2. The transformation can be computed as follows:

```python
import numpy as np

x = np.array([1, 2])
r = 1
d = 2

def polynomial_kernel(x, r, d):
    return (np.dot(x, x) + r) ** d

polynomial_kernel(x, r, d)
```

The output of the polynomial kernel is 16. This means that the point (1, 2) has been transformed into a higher dimensional space where the data can be linearly separated.

Reference: 
1. https://scikit-learn.org/stable/modules/svm.html
2. https://en.wikipedia.org/wiki/Support_vector_machine
3. https://www.kaggle.com/code/residentmario/kernels-and-support-vector-machine-regularization

## Data

The raw data is available here:

[Optidat_dataset](https://github.com/hese1175/hese1175.github.io/blob/main/assets/data/Optidat_dataset.xls)

[Cleaned Data](https://github.com/hese1175/hese1175.github.io/blob/main/assets/data/cleaned_data_export.csv)

The data is filtered and reduced to just the labeled numerical columns. Here is how the data looks like:

```
    Vf, %  Max. Stress, MPa  E, GPa  Max. % Strain
7  50.000           943.000 126.000          0.720
8  49.000           910.000 125.000          0.710
9  51.000           966.000 128.000          0.730
10 52.000           967.000 131.000          0.710
11 50.000          1066.000 125.000          0.810
```
The modulus is used as the target variable and the other columns are used as features. The data is split into training and testing sets. The training set is used to train the model and the testing set is used to evaluate the model. The data is then scaled using StandardScalar function. Here is what the X and y data looks like before and after scaling for the training set:

```
X_train (before scaling):
     Vf, %  Max. Stress, MPa  Max. % Strain
78  49.000           991.000          0.820
675 58.000           893.000          2.400
938 54.300          1408.000          3.030
29  47.000          1004.000          0.800
677 58.000           885.000          2.400

X_train (after scaling):
array([[-0.94381869, -0.42840733, -1.38327509],
       [ 0.63156814, -0.70307799,  0.39329389],
       [-0.01609089,  0.74034434,  1.10167267],
       [-1.29390465, -0.39197143, -1.40576331],
       [ 0.63156814, -0.72550008,  0.39329389],
       [ 0.28148218, -0.51529295,  0.67439658],
       
y_train (before scaling):
78     HIGH
675     LOW
938     LOW
29     HIGH
677     LOW
```

Splitting data into training and testing sets is important because it helps to evaluate the model on unseen data. The model is trained on the training set and then evaluated on the testing set. This helps to understand how well the model generalizes to new data.

Also, only labeled numerical columns are used in the model. This is because SVMs work well with numerical data. The target variable is the modulus and the other columns are used as features.

## Code

The code can be accesses on the GitHub repository using the link below:

[SVM Code](/assets/code/svm.py)

## Results

Three different kernels are used to train the model: linear, polynomial, and RBF. Additionally, three cost values are used to train the model: 0.1, 1, and 10. The model is then evaluated on the testing set. The results are shown below:

### Linear Kernel

The accuracy of the three variations of the linear kernel is shown below:

```
Linear SVM

Accuracy, C = 0.1: 0.9111111111111111
Accuracy, C = 1: 0.9555555555555556
Accuracy, C = 10: 0.9777777777777777
```

The confusion matrix for C = 1 is shown below:

![CM_Linear](/assets/imgs/svm/cm_linear.png)

### Polynomial Kernel

The accuracy of the three variations of the polynomial kernel is shown below:

```
Polynomial SVM, n = 3

Accuracy, C = 0.1: 0.9555555555555556
Accuracy, C = 1: 0.9777777777777777
Accuracy, C = 10: 0.9777777777777777
```

The confusion matrix for C = 1 is shown below:

![CM_Polynomial](/assets/imgs/svm/cm_poly.png)

### RBF Kernel

The accuracy of the three variations of the RBF kernel is shown below:

```
RBF SVM

Accuracy, C=0.1: 0.9111111111111111
Accuracy, C=1: 0.9555555555555556
Accuracy, C=10: 0.9777777777777777
```

The confusion matrix for C = 1 is shown below:

![CM_RBF](/assets/imgs/svm/cm_rbf.png)

### Comparison

The accuracy of the three kernels is compared in the plot below:

![CM_RBF](/assets/imgs/svm/Comparison.png)


## Conclusion

The accuracy of the three kernels is very similar. The linear kernel has an accuracy of 0.956, the polynomial kernel has an accuracy of 0.978, and the RBF kernel has an accuracy of 0.978. 

The polynomial kernel has the highest accuracy of 0.978. This means that the polynomial kernel is the best kernel for this dataset.

In all three kernels, the accuracy increases as the cost value increases. This means that the model is more accurate when the cost value is higher and is expected. The mis-classification happens for the high modulus samples while the low and medium modulus samples are classified correctly. More datapoints will be needed to improve the model at least on the high modulus side.