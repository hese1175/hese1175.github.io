#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:44:47 2024

@author: hemantsethi
"""


##########
# Naïve Bayes
##########

##### MultinomialNB

plot = True

from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



mnb_df = longitudinal_static_data.copy()

mnb_df_cols = ['Material', 'Lay-up', 'Vf, %', 'Resin Name', '0 Deg fabric', 
               'Cure / Post Cure', 'Process', 'Max. Stress, MPa', 
               'E, GPa', 'Max. % Strain', 'Resin Type',]

mnb_df = mnb_df[mnb_df_cols]

mnb_df_num = mnb_df.select_dtypes(include=['int64', 'float64'])
# Dropping columns with na values
mnb_df_num = mnb_df_num.dropna(axis=0, how='any')

mnb_df = mnb_df_num.merge(mnb_df, 'left')


cat_cols = ['Material', 'Lay-up', 'Resin Name', '0 Deg fabric', 
                    'Cure / Post Cure', 'Process','Resin Type']

num_cols = ['Vf, %', 'Max. Stress, MPa', 'E, GPa', 'Max. % Strain']

# Create preprocessing steps
le = LabelEncoder()

for col in cat_cols:
    print(col)
    mnb_df[col] = le.fit_transform(mnb_df[col])
    
le_name_mapping = dict(zip(le.classes_, 
                           le.transform(le.classes_)))

map_list = list(le_name_mapping.keys())

if plot:
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(mnb_df.corr(), vmin=-1, vmax = 1, cmap='coolwarm')
    ax.set_title('Correlation Matrix - NB Dataset', fontsize=15)


# Split data into dep and indep 
X = mnb_df.drop('Resin Type', axis=1)
y = mnb_df['Resin Type']

# Scale numerical features
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Train Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Make predictions
y_pred = mnb.predict(X_test)

# Print results
print("Multinomial Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)

if plot:
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cbar=False,
                xticklabels=map_list, yticklabels=map_list)
    ax.set_title('Confusion Matrix - MNB', fontsize=15)



##### Gaussian

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


le = LabelEncoder()

gnb_df = longitudinal_static_data.copy()


gnb_df = gnb_df[['Material', 'Lay-up', 'Vf, %', 'Resin Name', '0 Deg fabric', 
               'Cure / Post Cure', 'Process', 'Max. Stress, MPa', 
               'E, GPa', 'Max. % Strain', 'Resin Type',]]

gnb_df_num = gnb_df.select_dtypes(include=['int64', 'float64'])

categorical_cols = ['Material', 'Lay-up', 'Resin Name', '0 Deg fabric', 
                    'Cure / Post Cure', 'Process', 'Resin Type']

for col in categorical_cols:
    print(col)
    gnb_df[col] = le.fit_transform(gnb_df[col])
    
le_name_mapping = dict(zip(le.classes_, 
                           le.transform(le.classes_)))

map_list = list(le_name_mapping.keys())

# Dropping columns with na values
gnb_df_num = gnb_df_num.dropna(axis=0, how='any')
gnb_df_num.info()

gnb_df = gnb_df_num.merge(gnb_df, 'left')

if plot:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(gnb_df.corr(), vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_title('Correlation Matrix', fontsize=15)

# Split data into dep and indep 
X = gnb_df.drop('Resin Type', axis=1)
y = gnb_df['Resin Type']


# Scale numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Apply Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions
y_pred = gnb.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)

if plot:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cbar=False,
                xticklabels=map_list, yticklabels=map_list)
    ax.set_title('Confusion Matrix - GNB', fontsize=15)



######## CategoricalNB


from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



cnb_df = longitudinal_static_data.copy()

cnb_df_cols = ['Material', 'Lay-up', 'Vf, %', 'Resin Name', '0 Deg fabric', 
               'Cure / Post Cure', 'Process', 'Max. Stress, MPa', 
               'E, GPa', 'Max. % Strain', 'Resin Type',]

cnb_df = cnb_df[cnb_df_cols]

cnb_df_num = cnb_df.select_dtypes(include=['int64', 'float64'])
# Dropping columns with na values
cnb_df_num = cnb_df_num.dropna(axis=0, how='any')

cnb_df = cnb_df_num.merge(cnb_df, 'left')


cat_cols = ['Material', 'Lay-up', 'Resin Name', '0 Deg fabric', 
                    'Cure / Post Cure', 'Process','Resin Type']

num_cols = ['Vf, %', 'Max. Stress, MPa', 'E, GPa', 'Max. % Strain']

# Create preprocessing steps
le = LabelEncoder()
kbin = KBinsDiscretizer(n_bins=5, encode='ordinal', 
                                   strategy='uniform')
    

for col in cat_cols:
    cnb_df[col] = le.fit_transform(cnb_df[col])
    

cnb_df[num_cols] = kbin.fit_transform(cnb_df[num_cols])

# Split data into dep and indep 
X = cnb_df.drop('Resin Type', axis=1)
y = cnb_df['Resin Type']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Initialize and train the CategoricalNB model
cnb = CategoricalNB()
cnb.fit(X_train, y_train)

# Make predictions
y_pred = cnb.predict(X_test)

# Evaluate the model
print("Categorical Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)

if plot:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cbar=False,
                xticklabels=map_list, yticklabels=map_list)
    ax.set_title('Confusion Matrix - CNB', fontsize=15)
