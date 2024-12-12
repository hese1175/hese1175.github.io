#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:06:00 2024

@author: hemantsethi
"""

##############
# ENSEMBLE
#############

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


e_df = longitudinal_static_data.copy()

e_df_cols = [ 'Lay-up', 'Vf, %', '0 Deg fabric', 
               'Cure / Post Cure', 'Process', 'Max. Stress, MPa', 
               'E, GPa', 'Max. % Strain', 'Resin Type',]


e_df = e_df[e_df_cols]

e_df = e_df.select_dtypes(include=['int64', 'float64'])
# Dropping columns with na values
e_df = e_df.dropna(axis=0, how='any')

e_target = 'E, GPa'

# Split data into dep and indep 
X = e_df.drop(svm_target, axis=1)
# y = (svm_df[svm_target] > svm_df[svm_target].mean()).astype(int)
y = pd.cut(e_df[e_target], 3, labels=['LOW','MED','HIGH'],
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


# MOdel
gbt_model = GradientBoostingClassifier(random_state=22)
gbt_model.fit(X_train, y_train)

# Make predictions
y_pred = gbt_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

cf_matrix = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cf_matrix, annot=True, fmt='d', cbar=False,
            xticklabels=['Low', 'Med', 'High'], 
            yticklabels=['Low', 'Med', 'High'])
ax.set_title(f'Confusion Matrix - Gradient Boosted Trees', fontsize=15)