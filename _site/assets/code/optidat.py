#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:01:47 2024

@author: hemantsethi
"""
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 2000)

plot = False

file_name = './data_folder/Optidat_dataset.xls'
loc = os.getcwd()
file_loc = os.path.join(loc, file_name)

raw_data = pd.read_excel(file_loc, sheet_name='OPTIMAT Database', header=4)

print('Raw Data Info: \n')
print(raw_data.info())

# Only keep the following columns

data = raw_data.copy()

# Remove whitespace leading and trailing column names

data.rename(columns=lambda x: x.strip(), inplace=True)

column_list = ['Name', 'Fibre Volume Fraction', 'Material', 'Test type',
               'smax,static', 'smax, fatigue', 'epsstatic', 'epsfatigue',
               'Ncycles', 'Eit', 'Eic',
               'Invalid', 'Incomplete measurement data']

data = data.filter(column_list)
data.info()

# Remove rows for data that is Invalid

data = data[data['Invalid'].isna()]
print('Total Invalid Data:', data['Invalid'].notna().sum())

# Remove rows for data that is Incomplete

data = data[data['Incomplete measurement data'].isna()]
print('Total Incomplete measurement Data:', data['Incomplete measurement data'].notna().sum())

# Drop the two columns

data.drop(['Invalid', 'Incomplete measurement data'], axis=1, inplace=True)

# Remove rows that do not have a test type

data = data[data['Test type'].notna()]
data.info()

# Change data types

data.info()
data['Material'] = data['Material'].astype('category')
data['Test type'] = data['Test type'].astype('category')
data['Ncycles'] = data['Ncycles'].astype('float64')
# data['Eic'] = data['Eic'].astype('float64') # showing an error bc str


# Eic should both be all float or int

for i, u in enumerate(data['Eic']):
    if type(u) == str:
        print(i, u)

# row 325 has an error for Eic data; value has whitespaces '  '
# replacing with np.nan

data['Eic'] = np.where(data['Eic'] == ' ', np.nan, data['Eic'])

# Try changing dtype again

data['Eic'] = data['Eic'].astype('float64')

# check with data.info() or data.describe()
data.info()

# Start by visualizing all Test Types

# tt = data['Test type'].unique()

# Test type pie chart (Top 10)
tt_counts = data['Test type'].value_counts()
print(tt_counts)

plot = False
if plot:
    fig, ax = plt.subplots()
    fig.suptitle('Top 10 test types and total replicates', size=20)
    ax.pie(tt_counts[:10],
           autopct=lambda p: '{:.0f}'.format(p * sum(tt_counts[:10]) / 100),
           labels=tt_counts.index[:10])
    plt.savefig('./imgs/Pie_Chart_T10.png')

# Only keep CA, STT, and STC

tt_list = ['STT', 'STC', 'CA']

data = data[data['Test type'].isin(tt_list)]
print(data.shape)

tt_counts = data['Test type'].value_counts()
print(tt_counts)

plot = False
if plot:
    fig, ax = plt.subplots()
    fig.suptitle('Top 3 test types and total replicates', size=20)
    ax.pie(tt_counts[:3],
           autopct=lambda p: '{:.0f}'.format(p * sum(tt_counts[:3]) / 100),
           labels=tt_counts[:3].index)
    plt.savefig('./imgs/Pie_Chart_T3.png')

# Group data by Test Type for visualizations

tt_group = data.groupby('Test type', observed=True)

tensile_data = tt_group.get_group('STT')
compression_data = tt_group.get_group('STC')
fatigue_data = tt_group.get_group('CA')

# data head for tensile tests

# tensile_data.head(10)

# s_max= raw_data['smax']
# s_max_stt = s_max[s_max > 0]
# s_max_stc = s_max[s_max < 0]


# if plot:
#     fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9,5))

#     axs[0].violinplot(s_max_stt,
#                       showmeans=True,) # Plot tensile
#     axs[1].violinplot(s_max_stc,
#                       showmeans=True,)


# Cleaning tensile data
tensile_data = tensile_data[tensile_data['smax,static'] > 0]
tensile_data.dropna(subset=['smax,static'], inplace=True)

if plot:
    fig, ax = plt.subplots()
    sns.violinplot(data=tensile_data, y='smax,static', ax=ax)
    ax.set_xlabel('Probability Distribution Function (width ~ frequency)', size=10)
    ax.set_ylabel(r'Max Stress - $\sigma_{max}$ (MPa) ', size=10)
    ax.set_title('Tensile Test Max Stress Distribution', size=16)
    plt.savefig('./imgs/tensile_max_stress_dist.png')

# Not clean compression data plot
plot = True
if plot:
    fig, ax = plt.subplots()
    sns.violinplot(data=compression_data, y='smax,static', ax=ax)
    ax.set_xlabel('Probability Distribution Function (width ~ frequency)', size=10)
    ax.set_ylabel(r'Max Stress - $\sigma_{max}$ (MPa) ', size=10)
    ax.set_title('Compression Test Max Stress Distribution', size=16)
    plt.savefig('./imgs/compression_max_stress_dist_not-clean.png')

# Cleaning compression data
compression_data_clean = compression_data[compression_data['smax,static'] < 0]
compression_data_clean.dropna(subset=['smax,static'], inplace=True)

if plot:
    fig, ax = plt.subplots()
    sns.violinplot(data=compression_data_clean, y='smax,static', ax=ax)
    ax.set_xlabel('Probability Distribution Function (width ~ frequency)', size=10)
    ax.set_ylabel(r'Max Stress - $\sigma_{max}$ (MPa) ', size=10)
    ax.set_title('Compression Test Max Stress Distribution', size=16)
    plt.savefig('./imgs/compression_max_stress_dist.png')

# Plot not clean and clean side-by-side

if plot:
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 5), sharey=True)

    sns.violinplot(data=compression_data, y='smax,static', ax=axs[0],
                   color='r')
    sns.violinplot(data=compression_data_clean, y='smax,static', ax=axs[1],
                   color='g', )

    axs[0].set_xticklabels(['Not Clean'])
    axs[1].set_xticklabels(['Clean'])
    axs[0].set_ylabel(r'Max Stress - $\sigma_{max}$ (MPa) ', size=10)

    fig.supxlabel('Probability Distribution Function (width ~ frequency)',
                  size=10)
    plt.suptitle('Compression Test Max Stress Distribution', size=16)
    plt.savefig('./imgs/compression_max_stress_dist_clean-v-not-clean.png')

# Plot fatigue data

# Start with smax, fatigue

if plot:
    fig, ax = plt.subplots()
    sns.violinplot(data=fatigue_data, y='smax, fatigue', ax=ax)
    ax.set_xlabel('Probability Distribution Function (width ~ frequency)', size=10)
    ax.set_ylabel(r'Max Stress (Fatigue) - $\sigma_{max}$ (MPa) ', size=10)
    ax.set_title('Fatigue Test Max Stress Distribution', size=16)
    plt.savefig('./imgs/fatigue_max_stress_dist.png')

# Plot Ncycles as a function of smax, fatigue

if plot:
    fig, ax = plt.subplots()
    sns.scatterplot(data=fatigue_data, x='Ncycles', y='smax, fatigue',
                    ax=ax, )
    ax.set_xscale('log')
    ax.set_xlabel('Number of cycles till failure', size=10)
    ax.set_ylabel(r'Max Stress (Fatigue) - $\sigma_{max}$ (MPa) ', size=10)
    ax.set_title('S-N Curve for Fatigue data', size=16)
    plt.savefig('./imgs/fatigue_SN-curve.png')

# Same plot, Hue for material

if plot:
    fig, ax = plt.subplots()
    sns.scatterplot(data=fatigue_data, x='Ncycles', y='smax, fatigue',
                    ax=ax, hue='Material')
    ax.set_xscale('log')
    ax.set_xlabel('Number of cycles till failure', size=10)
    ax.set_ylabel(r'Max Stress (Fatigue) - $\sigma_{max}$ (MPa) ', size=10)
    ax.set_title('S-N Curve for Fatigue data, Hue-Material', size=16)
    plt.savefig('./imgs/fatigue_SN-curve_Material-hue.png')

# Check relation between Fiber Volume and Modulus

if plot:
    fig, ax = plt.subplots()
    sns.scatterplot(data=tensile_data, x='Fibre Volume Fraction', y='Eit',
                    ax=ax)

    ax.set_xlabel('Fibre Volume Fraction (%)', size=10)
    ax.set_ylabel(r'Elastic Modulus $E_t$ (MPa) ', size=10)
    ax.set_title('Modulus vs FVF', size=16)
    plt.savefig('./imgs/E-vs-FVF.png')