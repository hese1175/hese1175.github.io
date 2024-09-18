---
title: Data Preparation/EDA
parent: CSCI 5622 - ML
nav_order: 2
---

# Data Preparation/EDA
{: .no_toc }

1. TOC
{:toc}

## Introduction

The main data used for analysis is obtained from the [OptiDAT Database](https://web.archive.org/web/20180721153728/http://www.wmc.eu/optimatblades_optidat.php). This data contains information about the mechanical properties of different composite materials used in wind turbine blades. The data is in the form of an Excel file.

Furthermore, the [NewsAPI](https://newsapi.org/docs/endpoints/everything) is used to scrape news articles about the wind energy sector. This data is used to analyze the current trends and show the growing importance of recyclability in the wind energy sector.

## Data Gathering - NewsAPI

The NewsAPI is used to gather news articles about the wind energy sector and analyze the trends over the last couple decades. 

Here is the code that will be used to gather the articles:

```python
import requests
import re
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

# Connecting to NewsAPI and getting data
end = 'https://newsapi.org/v2/everything' 
all_jsons = []
for page in range(1, 6): 
    URL_id = {'apiKey' : 'de147731dffe48aa9effd3380691833b',
              'q' : 'wind energy AND (recycle OR recycling OR recyclable OR recyclability)',
              'from' : '1990-01-01',
              'to' : datetime.now().strftime('%Y-%m-%d'),
              'language' : 'en',
              'pageSize' : 100,
              'page' : page
              }
    
    response = requests.get(end, URL_id)
    jsontxt = response.json()
    
    all_jsons.append(jsontxt)
```
Data from all 5 pages are gathered in a list that will be used to create a DataFrame of articles.

## Data Cleaning and Visualization - NewsAPI

```python
# Create a DataFrame from the articles
df = pd.DataFrame(all_articles)

# Convert the publishedAt column to datetime
df['publishedAt'] = pd.to_datetime(df['publishedAt'])

# Extract the year from the publishedAt column
df['year'] = df['publishedAt'].dt.year

# Group by year and count the number of articles
trends = df.groupby('year').size().reset_index(name='article_count')

# Plot the trends
fig, ax = plt.subplots()

ax.plot(trends['year'], trends['article_count'], marker='o')
ax.set_title('Trends in Wind Energy and Recyclability Articles (1990-Present)')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Articles')
```

This idea did not work as expected due to the restrictions on the free version of the NewsAPI platform. The oldest data available was from 8-17-2024 so only one month. In the next iteration, the data will be gathered from a different source so that the trends can be analyzed.

Regardless, the data from the OptiDAT database has been gathered, cleaned and run through a preliminary analysis as shown below.

## Data Cleaning - OptiDAT Database

The OptiDAT data is in the form of an Excel file. The main data is in the *OPTIMAT Database* sheetname. It consists of 93 columns and 3435 rows. The columns contain information about the material properties, such as density, tensile strength, compressive strength, etc. The rows contain data about different composite materials that were tested as part of the study.

Here is a snippet of the raw data:
![Raw Data](/assets/imgs/raw_data_screenshot.png)

Using python, the raw data is loaded into a pandas dataframe. 
```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

file_name = './data/Optidat_dataset.xls'
loc = os.getcwd()
file_loc = os.path.join(loc,file_name)

raw_data = pd.read_excel(file_loc, sheet_name='OPTIMAT Database', header=4)

print('Raw Data Info: \n')
print(raw_data.info())
```
Here is a look at the data summary.

```plaintext
Raw Data Info: 

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3435 entries, 0 to 3434
Data columns (total 93 columns):
 #   Column                                Non-Null Count  Dtype         
---  ------                                --------------  -----         
 0   Optidat version: February 10, 2011    0 non-null      float64       
 1   optiDAT nr.                           3435 non-null   int64         
 2   Name                                  3435 non-null   object        
 3   Plate                                 3435 non-null   object        
 4   Fibre Volume Fraction                 3031 non-null   float64       
 5   Lab                                   3435 non-null   object        
 6   Laminate                              3435 non-null   object        
 7   Cut angle                             3435 non-null   object        
 8   Manufacturer                          3435 non-null   object        
 9   Geometry                              3435 non-null   object        
 10  Material                              3435 non-null   object        
 11  taverage                              3232 non-null   object        
 12  wmax                                  2084 non-null   float64       
 13  wmin                                  2814 non-null   float64       
 14  waverage                              3181 non-null   float64       
 15  area                                  3181 non-null   float64       
 16  Lnominal                              3435 non-null   object        
 17  Lmeasured                             836 non-null    float64       
 18  Lgauge                                2105 non-null   float64       
 19  Radius (waist)                        92 non-null     float64       
 20  TG and phase                          3435 non-null   object        
 21  Phase                                 3435 non-null   object        
 22  start date                            3253 non-null   datetime64[ns]
 23  end date                              3143 non-null   datetime64[ns]
 24  Test type                             3319 non-null   object        
 25  R-value1                              2003 non-null   object        
 26  Fmax                                  3256 non-null   float64       
 27  Ferror                                0 non-null      float64       
 28  Fstatic                               1788 non-null   float64       
 29  Ffatigue                              2045 non-null   float64       
 30  Fmax, 90°                             122 non-null    float64       
 31  epsmax                                1871 non-null   float64       
 32  epsstatic                             992 non-null    float64       
 33  epsfatigue                            977 non-null    float64       
 34  eps90°                                112 non-null    float64       
 35  eps45°                                136 non-null    float64       
 36  Nu                                    286 non-null    float64       
 37  smax                                  3035 non-null   float64       
 38  smax,static                           1627 non-null   float64       
 39  smax, fatigue                         1976 non-null   float64       
 40  shear strain                          95 non-null     float64       
 41  shear strength                        115 non-null    float64       
 42  shear modulus (near other moduli)     127 non-null    float64       
 43  Torquemax                             1311 non-null   float64       
 44  Torquefatigue                         50 non-null     float64       
 45  Torquestatic                          51 non-null     float64       
 46  Ncycles                               3300 non-null   object        
 47  level                                 1484 non-null   object        
 48  levelcalc                             1984 non-null   object        
 49  Nspectrum                             100 non-null    float64       
 50  failure mode                          1560 non-null   object        
 51  runout                                51 non-null     object        
 52  R-value2                              142 non-null    float64       
 53  Fmax2                                 221 non-null    float64       
 54  epsmax2                               87 non-null     float64       
 55  smax2                                 142 non-null    float64       
 56  Ncycles2                              141 non-null    float64       
 57  ratedisplacement                      1516 non-null   object        
 58  f                                     2038 non-null   object        
 59  f2                                    142 non-null    float64       
 60  Eit                                   1837 non-null   float64       
 61  Eic                                   1416 non-null   object        
 62  Eft                                   398 non-null    float64       
 63  Efc                                   168 non-null    float64       
 64  Elur,front                            9 non-null      object        
 65  Elur, back                            9 non-null      object        
 66  eps_max_t                             1096 non-null   float64       
 67  eps_max_c                             700 non-null    float64       
 68  Machine                               3345 non-null   object        
 69  control                               3125 non-null   object        
 70  grip                                  3108 non-null   object        
 71  ABG                                   2574 non-null   object        
 72  Temp.                                 718 non-null    object        
 73  Temp. control                         1779 non-null   object        
 74  Environment                           3435 non-null   object        
 75  Reference document                    2552 non-null   object        
 76  Remarks                               1847 non-null   object        
 77  Invalid                               165 non-null    object        
 78  Bending                               167 non-null    object        
 79  Buckling                              55 non-null     object        
 80  Overheating                           28 non-null     object        
 81  Tab failure                           262 non-null    object        
 82  Delaminated                           22 non-null     object        
 83  Incomplete measurement data           207 non-null    object        
 84  Strain from E                         29 non-null     object        
 85  LUR                                   217 non-null    object        
 86  TEC                                   50 non-null     float64       
 87  data delivered under name             803 non-null    object        
 88  Repair characteristics                167 non-null    object        
 89  Strain measurement equipment (long)   2209 non-null   object        
 90  Strain measurement equipment (short)  2225 non-null   object        
 91  Grip pressure                         1876 non-null   object        
 92  time per test                         2909 non-null   float64       
 dtypes: datetime64[ns](2), float64(43), int64(1), object(47)
```

Now, the data is cleaned by removing columns that are not needed for analysis. The columns are removed as follows:

```python
# Remove whitespace leading and trailing column names
data.rename(columns=lambda x: x.strip(), inplace=True)
 
column_list = ['Name', 'Fibre Volume Fraction', 'Material','Test type', 
               'smax,static','smax, fatigue','epsstatic','epsfatigue',
               'Ncycles','Eit','Eic',
               'Invalid','Incomplete measurement data']

data = data.filter(column_list)
data.info()
```

```plaintext
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3435 entries, 0 to 3434
Data columns (total 13 columns):
 #   Column                       Non-Null Count  Dtype  
---  ------                       --------------  -----  
 0   Name                         3435 non-null   object 
 1   Fibre Volume Fraction        3031 non-null   float64
 2   Material                     3435 non-null   object 
 3   Test type                    3319 non-null   object 
 4   smax,static                  1627 non-null   float64
 5   smax, fatigue                1976 non-null   float64
 6   epsstatic                    992 non-null    float64
 7   epsfatigue                   977 non-null    float64
 8   Ncycles                      3300 non-null   object 
 9   Eit                          1837 non-null   float64
 10  Eic                          1416 non-null   object 
 11  Invalid                      165 non-null    object 
 12  Incomplete measurement data  207 non-null    object 
dtypes: float64(6), object(7)
```

Two important columns to easily remove data is by looking at the Invalid data column and the Incomplete measurement data column. If the data is invalid or incomplete, it is removed from the dataset. Also, rows that do not have a designated test type are removed.

```python
# Remove rows for data that is Invalid
data = data[data['Invalid'].isna()]
print('Total Invalid Data:',data['Invalid'].notna().sum())

# Remove rows for data that is Incomplete
data = data[data['Incomplete measurement data'].isna()]
print('Total Incomplete measurement Data:',data['Incomplete measurement data'].notna().sum())

# Drop the two columns
data.drop(['Invalid','Incomplete measurement data'],axis=1,inplace=True)

# Remove rows that do not have a test type
data = data[data['Test type'].notna()]
data.info()
```
Here is the current state of columns and data types:

```plaintext
<class 'pandas.core.frame.DataFrame'>
Index: 3068 entries, 0 to 3433
Data columns (total 11 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   Name                   3068 non-null   object 
 1   Fibre Volume Fraction  2729 non-null   float64
 2   Material               3068 non-null   object 
 3   Test type              3068 non-null   object 
 4   smax,static            1518 non-null   float64
 5   smax, fatigue          1752 non-null   float64
 6   epsstatic              961 non-null    float64
 7   epsfatigue             927 non-null    float64
 8   Ncycles                2983 non-null   object 
 9   Eit                    1759 non-null   float64
 10  Eic                    1336 non-null   object 
dtypes: float64(6), object(5)
```
Now we correct the data types for the columns. The Name column is temporarily left as an object dtype. More on this will be discussed later. The Test type and Material columns are also converted to a category data type. Finally, the Ncycles and Eic columns are to be converted to a float data type.

```python
# Change data types
data['Material'] = data['Material'].astype('category')
data['Test type'] = data['Test type'].astype('category')
data['Ncycles'] = data['Ncycles'].astype('float64')
data['Eic'] = data['Eic'].astype('float64')
```
The dtype conversion of `data['Eic']` is showing an error because of a string value in the column. 

First we need to find all the string values in the column.

```python
# Eic should both be all float or int
for i,u in enumerate(data['Eic']):
    if type(u) == str:
        print(i, u)
```
Row 325 has an error for Eic data; value has whitespaces `'  '`. This is corrected by converting the string to a `np.nan` value.

```python
data['Eic'] = np.where(data['Eic'] == ' ', np.nan, data['Eic'])

# Try changing dtype again
data['Eic'] = data['Eic'].astype('float64')

# check with data.info() or data.describe()
data.info()
```

```plaintext
<class 'pandas.core.frame.DataFrame'>
Index: 3068 entries, 0 to 3433
Data columns (total 11 columns):
 #   Column                 Non-Null Count  Dtype   
---  ------                 --------------  -----   
 0   Name                   3068 non-null   object  
 1   Fibre Volume Fraction  2729 non-null   float64 
 2   Material               3068 non-null   category
 3   Test type              3068 non-null   category
 4   smax,static            1518 non-null   float64 
 5   smax, fatigue          1752 non-null   float64 
 6   epsstatic              961 non-null    float64 
 7   epsfatigue             927 non-null    float64 
 8   Ncycles                2983 non-null   float64 
 9   Eit                    1759 non-null   float64 
 10  Eic                    1335 non-null   float64 
dtypes: category(2), float64(8), object(1)
```

```plaintext
       Fibre Volume Fraction  smax,static  smax, fatigue  epsstatic  epsfatigue      Ncycles      Eit      Eic
count               2729.000     1518.000       1752.000    961.000     927.000     2983.000 1759.000 1335.000
mean                  45.004      113.787        127.534      0.672       0.454   345061.136   26.331   29.376
std                   18.919      476.216        221.863      1.872       0.698  1514505.719    9.902    8.981
min                    0.000    -5274.270       -511.466     -3.846      -1.666        0.000    0.778   10.750
25%                   50.868     -387.185         42.627     -1.332       0.173        1.000   14.670   25.800
50%                   52.480      102.466        181.912      1.300       0.608     2452.000   27.530   28.619
75%                   53.730      505.058        264.490      2.121       0.910    91196.000   34.600   38.200
max                   56.660     1156.504        761.461      8.080       1.970 39393907.000   49.300   57.160
```


## Prelim Exploratory Data Analysis - OptiDAT Database

The data looks relatively clean and ready for some exploratory data analysis with visualizations.

Visualizing the data can be helpful to clean the data further. Looking at the raw data, the test type column has a lot of unique values. It might be helpful to visualize the top 10 test types.

```python
# Start by visualizing all Test Types

# tt = data['Test type'].unique()

# Test type pie chart (Top 10)
tt_counts = data['Test type'].value_counts()
print(tt_counts)

if plot:
    fig, ax = plt.subplots()
    fig.suptitle('Top 10 test types and total replicates', size=20)
    ax.pie(tt_counts[:10],
            autopct=lambda p: '{:.0f}'.format(p * sum(tt_counts[:10]) / 100),
            labels = tt_counts.index[:10])
```

`tt_counts` shows that the test type column has 83 unique values. The top 10 test types are shown in the pie chart.

To narrow the analysis and only include the important test types, top 3 test types are selected. The data is filtered to only include the top 3 test types - STT (Static Tensile), STC (Static Compression) and CA (Constant Amplitude Fatigue).

```python
# Only keep CA, STT, and STC

tt_list = ['STT','STC','CA']
data = data[data['Test type'].isin(tt_list)]

# Plot the top 3 test types
if plot:
    
    fig, ax = plt.subplots()
    fig.suptitle('Top 3 test types and total replicates', size=20)
    ax.pie(tt_counts[:3],
            autopct=lambda p: '{:.0f}'.format(p * sum(tt_counts[:3]) / 100),
            labels = tt_counts[:3].index)
    plt.savefig('./imgs/Pie_Chart_T3.png')
```

The preliminary data can be further grouped and analyzed by the major differtiator i.e. Test type. 

```python
# Group data by Test Type for visualizations
tt_group = data.groupby('Test type', observed=True)

tensile_data = tt_group.get_group('STT')
compression_data = tt_group.get_group('STC')
fatigue_data = tt_group.get_group('CA')

# data head for tensile tests
tensile_data.head(10)
```
Here is the data for the initial 10 tensile tests. Similar data are grouped for compression and fatigue (CA) tests.

```plaintext
                  Name  Fibre Volume Fraction Material Test type  smax,static  smax, fatigue  epsstatic  epsfatigue  Ncycles    Eit  Eic
5   GEV201_D0100_0016                    NaN      GE1       STT      812.248            NaN      2.104         NaN    1.000 44.578  NaN
9   GEV201_D0100_0031                    NaN      GE1       STT      859.098            NaN      2.148         NaN    1.000 44.911  NaN
10  GEV201_R0100_0001                    NaN      GE1       STT      879.473            NaN        NaN         NaN    1.000 39.430  NaN
14  GEV201_R0100_0011                    NaN      GE1       STT      841.836            NaN        NaN         NaN    1.000 39.523  NaN
20  GEV201_R0100_0026                    NaN      GE1       STT      751.744            NaN     -1.960         NaN    1.000 39.190  NaN
26  GEV202_D0100_0001                    NaN      GE1       STT      712.281            NaN        NaN         NaN    1.000    NaN  NaN
28  GEV202_D0100_0006                    NaN      GE1       STT      688.875            NaN        NaN         NaN    1.000    NaN  NaN
38  GEV202_R0100_0006                    NaN      GE1       STT      689.609            NaN        NaN         NaN    1.000    NaN  NaN
40  GEV202_R0100_0011                    NaN      GE1       STT      675.225            NaN        NaN         NaN    1.000    NaN  NaN
59  GEV203_R0100_0001                    NaN      GE1       STT      483.126            NaN        NaN         NaN    1.000    NaN  NaN
```

`np.nan` values in certain columns are expected as fatigue related test data will not be available for tensile and compression tests and vice versa.

Some easy visualizations can be done to understand the data better. For example, a box or violin plot to show the distribution of the data. 

But first, the stress data for tensile must be cleaned to remove invalid values (negative values) and outliers. 
```python
# Cleaning tensile data
tensile_data = tensile_data[tensile_data['smax,static']>0]
tensile_data.dropna(subset=['smax,static'], inplace=True)

# Visualize the data
if plot:
    fig, ax = plt.subplots()
    sns.violinplot(data=tensile_data, y='smax,static', ax=ax)
    ax.set_xlabel('Probability Distribution Function (width ~ frequency)', size=10)
    ax.set_ylabel(r'Max Stress - $\sigma_{max}$ (MPa) ', size=10)
    ax.set_title('Tensile Test Max Stress Distribution', size=16)
    plt.savefig('./imgs/tensile_max_stress_dist.png')
```



Similarly the data for compression and fatigue tests can be cleaned and visualized.

```python
if plot:
    fig, ax = plt.subplots()
    sns.violinplot(data=compression_data, y='smax,static', ax=ax)
    ax.set_xlabel('Probability Distribution Function (width ~ frequency)', size=10)
    ax.set_ylabel(r'Max Stress - $\sigma_{max}$ (MPa) ', size=10)
    ax.set_title('Compression Test Max Stress Distribution', size=16)
    plt.savefig('./imgs/compression_max_stress_dist_not-clean.png')
```

However, we see that the compression data has some values that are positive. This is not expected for compression tests. The data is cleaned by removing the positive values.

```python
# Cleaning compression data
compression_data = compression_data[compression_data['smax,static']<0]
compression_data.dropna(subset=['smax,static'], inplace=True)

if plot:
    fig, ax = plt.subplots()
    sns.violinplot(data=compression_data, y='smax,static', ax=ax)
    ax.set_xlabel('Probability Distribution Function (width ~ frequency)', size=10)
    ax.set_ylabel(r'Max Stress - $\sigma_{max}$ (MPa) ', size=10)
    ax.set_title('Compression Test Max Stress Distribution', size=16)
    plt.savefig('./imgs/compression_max_stress_dist.png')
```

Finally, the fatigue max stress data `smax, fatigue` is plotted. 

```python
if plot:
    fig, ax = plt.subplots()
    sns.violinplot(data=fatigue_data, y='smax, fatigue', ax=ax)
    ax.set_xlabel('Probability Distribution Function (width ~ frequency)', size=10)
    ax.set_ylabel(r'Max Stress (Fatigue) - $\sigma_{max}$ (MPa) ', size=10)
    ax.set_title('Fatigue Test Max Stress Distribution', size=16)
    plt.savefig('./imgs/fatigue_max_stress_dist.png')
```

There is more to the fatigue data than just max stress. The values can be negative if the ratio of min to max stress is not 1. The ratio data is available in the `R-value1` column and will be utilized in the analysis later.

Now, the fatigue Ncycles data is visualized. Generally, this is done in a log plot. 

```python
# Plot Ncycles as a function of smax, fatigue

if plot:
    fig, ax = plt.subplots()
    sns.scatterplot(data=fatigue_data, x='Ncycles', y='smax, fatigue',
                    ax=ax,)
    ax.set_xscale('log')
    ax.set_xlabel('Number of cycles till failure', size=10)
    ax.set_ylabel(r'Max Stress (Fatigue) - $\sigma_{max}$ (MPa) ', size=10)
    ax.set_title('S-N Curve for Fatigue data', size=16)
    plt.savefig('./imgs/fatigue_SN-curve.png')
```

There might be a relation to the material so the same plot is done with a hue for the material. 

```python
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
```

Moving back to the tensile data, the tensile modulus data `Eit` is visualized as a function of the Fiber Volume Fraction in the material. 

```python
# Check relation between Fiber Volume and Modulus for tensile data
if plot:
    fig, ax = plt.subplots()
    sns.scatterplot(data=tensile_data, x='Fibre Volume Fraction', y='Eit', 
                    ax=ax)

    ax.set_xlabel('Fibre Volume Fraction (%)', size=10)
    ax.set_ylabel(r'Elastic Modulus $E_t$ (MPa) ', size=10)
    ax.set_title('Modulus vs FVF', size=16)
    plt.savefig('./imgs/E-vs-FVF.png')
```

Looks like there are two distinct groups in the data. This will be investigated later in the study.

All the plots are available in the Introduction section of the project as per the guidelines. 

