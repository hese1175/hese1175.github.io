#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:28:37 2024

@author: hemantsethi
"""


##############
# DECISION TREES
##############

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree



dt_df = longitudinal_static_data.copy()

# dt_df_cols = ['Material', 'Lay-up', 'Vf, %', 'Resin Name', '0 Deg fabric', 
#                'Cure / Post Cure', 'Process', 'Max. Stress, MPa', 
#                'E, GPa', 'Max. % Strain', 'Resin Type',]

dt_df_cols = [ 'Lay-up', 'Vf, %', '0 Deg fabric', 
               'Cure / Post Cure', 'Process', 'Max. Stress, MPa', 
               'E, GPa', 'Max. % Strain', 'Resin Type',]


dt_df = dt_df[dt_df_cols]

dt_df_num = dt_df.select_dtypes(include=['int64', 'float64'])
# Dropping columns with na values
dt_df_num = dt_df_num.dropna(axis=0, how='any')

dt_df = dt_df_num.merge(dt_df, 'left')

cat_cols = ['Lay-up', '0 Deg fabric','Cure / Post Cure', 
            'Process','Resin Type']

num_cols = ['Vf, %', 'Max. Stress, MPa', 'E, GPa', 'Max. % Strain']

# Create preprocessing steps
le = LabelEncoder()
for col in cat_cols:
    dt_df[col] = le.fit_transform(dt_df[col])



# Split data into dep and indep 
X = dt_df.drop('Resin Type', axis=1)
y = dt_df['Resin Type']

le = LabelEncoder()
y = le.fit_transform(y)

le_name_mapping = dict(zip(le.classes_, 
                           le.transform(le.classes_)))


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Initialize and train the Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Evaluate the model
print("Decision Tree Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': dt.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Visualize the decision tree
if plot:
    plt.figure(figsize=(20,10))
    plot_tree(dt, feature_names=X.columns, filled=True, rounded=True, fontsize=6)
    plt.title('Decision Tree: Default - GINI, max_depth=None, min_sample_split=2')





# Initialize and train the Decision Tree Classifier
dt = DecisionTreeClassifier( splitter='random',
                            # max_depth=5, min_samples_split=5,
                            criterion='entropy')
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Evaluate the model
print("Decision Tree Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': dt.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Visualize the decision tree
if plot:
    plt.figure(figsize=(20,10))
    plot_tree(dt, feature_names=X.columns, filled=True, rounded=True, fontsize=6)
    plt.title('Decision Tree: ENTROPY, max_depth=None, min_sample_split=2')




# Initialize and train the Decision Tree Classifier
dt = DecisionTreeClassifier(splitter='random',
                            # max_depth=5, min_samples_split=5,
                            criterion='log_loss')

dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Evaluate the model
print("Decision Tree Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': dt.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Visualize the decision tree
if plot:
    plt.figure(figsize=(20,10))
    plot_tree(dt, feature_names=X.columns, filled=True, rounded=True, fontsize=6)
    plt.title('Decision Tree: LOG_LOSS, max_depth=None, min_sample_split=2')
    
    

    

# Initialize and train the Decision Tree Classifier
dt = DecisionTreeClassifier(splitter='random',
                            max_depth=4, min_samples_split=5,)

dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Evaluate the model
print("Decision Tree Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': dt.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Visualize the decision tree
if plot:
    plt.figure(figsize=(20,10))
    plot_tree(dt, feature_names=X.columns, filled=True, rounded=True, fontsize=12)
    plt.title('Decision Tree: GINI, max_depth=3, min_sample_split=2')
    
