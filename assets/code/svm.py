#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:06:26 2024

@author: hemantsethi
"""

##################
# SVM
##################

plot=False

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


svm_df = longitudinal_static_data.copy()

svm_df_cols = [ 'Lay-up', 'Vf, %', '0 Deg fabric', 
               'Cure / Post Cure', 'Process', 'Max. Stress, MPa', 
               'E, GPa', 'Max. % Strain', 'Resin Type',]


svm_df = svm_df[svm_df_cols]

svm_df = svm_df.select_dtypes(include=['int64', 'float64'])
# Dropping columns with na values
svm_df = svm_df.dropna(axis=0, how='any')

svm_target = 'E, GPa'

# Split data into dep and indep 
X = svm_df.drop(svm_target, axis=1)
# y = (svm_df[svm_target] > svm_df[svm_target].mean()).astype(int)
y = pd.cut(svm_df[svm_target], 3, labels=['LOW','MED','HIGH'],
                     duplicates='drop')

# le = LabelEncoder()
# y = le.fit_transform(y)

# le_name_mapping = dict(zip(le.classes_, 
#                            le.transform(le.classes_)))


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=22)
# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

C_list = [0.1, 1, 10]
####### LINEAR
# Train Linear SVM 
print ('Linear SVM\n')
accuracy_linear_list = []
for C in C_list:
    svm_linear = SVC(kernel='linear', C=C, random_state=22)
    svm_linear.fit(X_train, y_train)
    
    # Make predictions
    y_pred_linear = svm_linear.predict(X_test)
    
    
    # Evaluate the model
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    report_linear = classification_report(y_test, y_pred_linear)
    accuracy_linear_list.append(accuracy_linear)
    cf_matrix = confusion_matrix(y_test, y_pred_linear)
    
    if C==1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cf_matrix, annot=True, fmt='d', cbar=False,
                    xticklabels=['Low', 'Med', 'High'], 
                    yticklabels=['Low', 'Med', 'High'])
        ax.set_title(f'Confusion Matrix - Linear, C={C}', fontsize=15)
    
    
    print(f"Accuracy, C = {C}: {accuracy_linear}")
    
print('\n\n')

####### POLY
# Train Polynomial SVM 

print ('Polynomial SVM, n = 3\n')
accuracy_poly_list = []

for C in C_list:
    
    svm_poly = SVC(kernel='poly', C=C, random_state=22)
    svm_poly.fit(X_train, y_train)
    
    # Make predictions
    y_pred_poly = svm_poly.predict(X_test)
    
    # Evaluate the model
    accuracy_poly = accuracy_score(y_test, y_pred_poly)
    report_poly = classification_report(y_test, y_pred_poly)
    accuracy_poly_list.append(accuracy_poly)
    cf_matrix = confusion_matrix(y_test, y_pred_poly)
    
    if C==1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cf_matrix, annot=True, fmt='d', cbar=False,
                    xticklabels=['Low', 'Med', 'High'], 
                    yticklabels=['Low', 'Med', 'High'])
        ax.set_title(f'Confusion Matrix - Poly, n=3, C={C}', fontsize=15)
    
    
    print(f"Accuracy, C = {C}: {accuracy_poly}")

####### RBF
# Train RBF SVM 
print ('RBF SVM\n')
accuracy_rbf_list = []
for C in C_list:
    svm_rbf = SVC(kernel='rbf', C=C, random_state=22)
    svm_rbf.fit(X_train, y_train)
    
    # Make predictions
    y_pred_rbf = svm_rbf.predict(X_test)
    
    # Evaluate the model
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    report_rbf = classification_report(y_test, y_pred_rbf)
    accuracy_rbf_list.append(accuracy_rbf)
    
    cf_matrix = confusion_matrix(y_test, y_pred_rbf)
    
    if C==1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cf_matrix, annot=True, fmt='d', cbar=False,
                    xticklabels=['Low', 'Med', 'High'], 
                    yticklabels=['Low', 'Med', 'High'])
        ax.set_title(f'Confusion Matrix - RBF, C={C}', fontsize=15)
    
    
    print(f"Accuracy, C={C}: {accuracy_rbf}")

fig, ax = plt.subplots(figsize=(6, 5))
plt.plot(C_list, accuracy_linear_list, label='Linear')
plt.plot(C_list, accuracy_poly_list, label='Polynomial, n=3')
plt.plot(C_list, accuracy_rbf_list, label='RBF', ls='--')
ax.set_xscale('log')
ax.set_title('Comparing SVM Kernels with varying Cost function', size=14)
ax.set_xlabel('Cost (C)', size=12)
ax.set_ylabel('Accuracy Score', size=12)
ax.legend()

