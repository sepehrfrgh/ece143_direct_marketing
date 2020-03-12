# ECE-143-Project-Group-17

This work addresses the problem of predicting which features will decide a good client
<br>
<p align="center">
    <img src='together.png' height="500" >
</p>

## Dependencies
Our work is implemented in Python, here are the dependencies.

- Python 3
- pandas
- matplotlib
- seaborn
- mpl_toolkits
- sklearn
- pre_processing
- scipy
- warnings

```shell
# Download this code
git clone https://github.com/sepehrfrgh/ece143_direct_marketing.git
cd ece143_direct_marketing
```

## Overview
We provide:
- Downloaded data from uci dataset
- Code for data analysis
- Jupyter notebook which shows all the visualizations
- Pdf file of presentation

### Data
The data in `./dataset/` is already downloaded from uci dataset. the`.txt` file is an introduction of our dataset.

You are highly recommended to use our downloaded data because the origin dataset has other versions.

You can also downloaded the data youself from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing and put them in `./dataset/`


### Jupyter Notebook
We provide the notebook version of of code in xxx.ipynb

-xxx.ipynb: 

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




