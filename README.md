# ECE-143-Project-Group-17

This work addresses the problem of predicting which features will decide a good client
<br>
<p align="center">
    <img src='together.png' height="500" >
</p>

## Installation

```shell
git clone https://github.com/sepehrfrgh/ece143_direct_marketing.git
cd ece143_direct_marketing
```

### Dependencies

- Python>=3.6
- pandas
- matplotlib
- seaborn
- scikit-learn
- jupyter


## Overview
Repository contains:
- Downloaded data from the UCI Bank Marketing dataset
- Data analysis code
- Visualizations in Jupyter notebook

### Data
The UCI dataset can be found in the `./dataset/` directory. the`.txt` file is an introduction to the dataset.

The original data can be found at https://archive.ics.uci.edu/ml/datasets/Bank+Marketing `./dataset/`


### Jupyter Notebook
Get started with the Jupyter notebook `analysis-notebook.ipynb`

### Data Analysis

```shell
#to analyse features, such as age, month, etc.
python xxx.py
```

-pre_processing.py : dataframe,mapping,load function
-analysis.py : age,month analysis
-feature_importance.py : use random forest to show relations between features

### Result 
The plots of data analysis will be saved in `./result_plots/`.




