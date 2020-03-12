# ECE-143-Project-Group-17

How do telemarketers decide who to target with their calls?
This project addresses this question with the analysis of the data
from a direct marketing campaign undertaken by a Portuguese bank.

<br>
<p align="center">
    <img src='together.png' height="500" >
</p>

## Installation

```shell script
git clone https://github.com/sepehrfrgh/ece143_direct_marketing.git
cd ece143_direct_marketing
pip install -r requirements.txt
```

### Dependencies

- Python>=3.6
- pandas
- matplotlib
- seaborn
- scikit-learn
- jupyter

## Usage

```shell script
jupyter notebook analysis-notebook.ipynb 
```

## Overview

Repository contains:
- Downloaded data from the UCI Bank Marketing dataset
- Data analysis code
- Visualizations in Jupyter notebook

### Data

The UCI dataset can be found in the `./dataset/` directory. the`.txt` file is an introduction to the dataset.

The original data can be found at https://archive.ics.uci.edu/ml/datasets/Bank+Marketing `./dataset/`

### Jupyter Notebook

Visualizations were produced in `analysis-notebook.ipynb`

### Data Analysis

- pre_processing.py : Data loading and processing
- analysis.py : Analysis tools
