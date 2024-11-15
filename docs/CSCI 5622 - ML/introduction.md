---
title: Introduction
parent: CSCI 5622 - ML
nav_order: 1
---

# Introduction
{: .no_toc }

1. TOC
{:toc}

## Wind Energy 

Wind energy is a key component in the mitigation strategy for climate change. It has grown into a significant sector in the global renewable energy landscape. However, the growth has not been fast enough to meet the climate change targets. The wind energy sector is facing challenges in terms of cost, reliability, efficiency and more recently, recyclability. Wind blades are one of the most expensive components of the turbine as well as the most difficult to recycle. 
![Wind Turbine](/assets/imgs/wind_turbine.png)


Recyclable wind turbines will help reduce cost by enabling the reuse of expensive materials while reducing waste. This will help wind energy become more competitive in the renewable energy space.

![Wind Energy Growth](/assets/imgs/sweetwater-wind-blades.jpg)
*Wind Turbines in Landfill as the materials cannot be recycled*

This project will aim to address the recyclability of wind blades. The project will focus on the development of a machine learning model to analyze characteristics of current renewable, recyclable composite materials, and compare their properties to that of traditional materials (epoxy and fiberglass). By analyzing the results from current materials, the model will be able to predict which characteristics are most important for recyclable materials. 

In this project, two datasets will be used. The first dataset is the OptiDAT dataset which contains material properties of wind turbine blades. The second dataset is the SNL/DOE/MSU dataset which also has recent material properties of wind turbine blades. The datasets will be used to train the machine learning model to predict the most important characteristics of recyclable materials.

By identifying and predicting the most important features of the data, the model will help researchers and engineers develop new materials faster and in an efficient manner. There is a need for new technologies and innovations to address these challenges and make wind energy more competitive in the renewable energy space. 

### Raw Data Image

![Raw Data](/assets/imgs/raw_data_screenshot.png)

### Cleaned Data Image

![Cleaned Data](/assets/imgs/cleaned_data_screenshot.png)

### Visualization Images

All the images below are visualizations of the data. The code is available in the date_prep tab as well as the process of data cleaning and visualization. Please refer to that tab for more information.

![Pie Chart T10](/assets/imgs/Pie_Chart_T10.png)
\
*Pie chart shows the top 10 test types in the whole dataset*

![Pie Chart T3](/assets/imgs/Pie_Chart_T3.png)
\
*Pie chart shows the top 3 most important test types in the whole dataset*

![Tensile Test Max Stress Dist](/assets/imgs/tensile_max_stress_dist.png)
\
*Violin plot showing the distribution of max stress in tensile data for all materials*

![Compression Test Max Stress Dist Not Clean](/assets/imgs/compression_max_stress_dist_not-clean.png)
\
*Violin plot of max stress in compression data. This showed some positive values which needed to be removed*

![Compression Test Max Stress Dist Clean](/assets/imgs/compression_max_stress_dist.png)
\
*Violin plot of max stress in compression data after removing positive values*

![Side by side vis](/assets/imgs/compression_max_stress_dist_clean-v-not-clean.png)
\
*Side by side comparison showing the raw and clean data*

![Fatigue Test Max Stress Dist](/assets/imgs/fatigue_max_stress_dist.png)
\
*Violin plot of max stress in fatigue data*

![Fatigue SN curve](/assets/imgs/fatigue_SN-curve.png)
\
*Scatter plot showing SN curve relation for fatigue tests*

![Fatigue SN curve Material Hue](/assets/imgs/fatigue_SN-curve_Material-hue.png)
\
*Scatter plot showing the S-N curve for fatigue tests with material hue*

![Tensile Modulus v FVF](/assets/imgs/E-vs-FVF.png)
\
*Scatter plot showing the relationship between tensile modulus and fiber volume fraction. The plot shows two distinct groups which will be investigated later.*

Raw data is available on the GitHub repository.
\
[OptiDAT dataset](/assets/data/Optidat_dataset.xls)

Code used is also available on the GitHub repository.
\
[Data Prep Code](/assets/code/optidat.py)
\
[Prelim NewsAPI code](/assets/code/newapi.py)

Images are available in the assets/imgs folder.
\
[Images and Plots](/assets/imgs)

References: \
[1]: https://www.texasmonthly.com/news-politics/sweetwater-wind-turbine-blades-dump/ \
[2]: Wiser, R.; Bolinger, M.; Hoen, B. Land-Based Wind Market Report: 2022 Edition, Department of Energey Report, 2022. \
[3]: Komusanac, I.; Brindley, G.; Fraile, D.; Ramirez, L. Wind Energy in Europe: 2021 Statistics and the Outlook for 2022-2026, WindEurope Report, 2022.