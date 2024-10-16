#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 21:45:44 2024

@author: hemantsethi
"""


###########
# ARM
###########

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


categorical_cols = ['Material', 'Lay-up', 'Resin Name', '0 Deg fabric', '90 deg fabric', 'Cure / Post Cure', 'Process', 'Coupon', 'Resin Type']

# One-hot encoding for categorical columns
encoded_df = pd.get_dummies(static_data, columns=categorical_cols, drop_first=True)
encoded_df = encoded_df.drop(['R-value', 'Freq., Hz', 'Cycles', 'Resin',], axis=1)

# Binarize quantitative data
quantitative_cols = ['Vf, %', 'Max. Stress, MPa', 'Min. Stress, MPa', 'E, GPa', 'Max. % Strain', 'Min. % Strain',]

for col in quantitative_cols:
    binned_col = pd.cut(encoded_df[col], 3, labels=['LOW','MED','HIGH'],
                         duplicates='drop')
    encoded_df[col] = binned_col
    # threshold = encoded_df[col].median()
    # encoded_df[col] = encoded_df[col].apply(lambda x: 1 if x > threshold else 0)

encoded_df = pd.get_dummies(encoded_df, columns=quantitative_cols, drop_first=True)


# Apply the Apriori algorithm with a minimum support of 0.2
frequent_itemsets = apriori(encoded_df, min_support=0.2, use_colnames=True)

# Generate association rules from frequent itemsets with minimum confidence of 0.5
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

print(rules)

rules.sort_values(by='confidence', ascending=False)

