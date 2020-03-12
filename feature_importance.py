# Data Manipulation 
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Feature Selection and Encoding
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from pre_processing import DfBankAdditional

# Machine learning 
import sklearn.ensemble as ske
from sklearn import datasets, model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
#import tensorflow as tf

# Grid and Random Search
import scipy.stats as st
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Metrics
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

# Managing Warnings 
import warnings
warnings.filterwarnings('ignore')

#mission: predict if the client will subscribe (yes/no) a term deposit (variable y).
# Load Training and Test Data Sets
headers = ['age', 'job', 'marital', 
           'education', 'default', 
           'housing', 'loan', 
           'contact', 'month', 'day_of_week', 
           'duration', 'campaign', 
           'pdays', 'previous', 
           'poutcome','emp.var.rate','cons.price.idx',
           'cons.conf.idx','euribor3m','nr.employed','y']
training_raw = DfBankAdditional(pd.read_csv('dataset/bank-additional-full.csv', 
                       header=None, 
                       names=headers, 
                       sep=';', 
                       na_values=["?"], 
                       engine='python',
                          skiprows=1,
                          index_col=False))
test_raw = DfBankAdditional(pd.read_csv('dataset/bank-additional.csv', 
                      header=None, 
                      names=headers, 
                      sep=';', 
                      na_values=["?"], 
                      engine='python', 
                      skiprows=1,
                      index_col=False))

training_raw.process_all()
test_raw.process_all()

training_raw = training_raw.df
test_raw = test_raw.df

# Join Datasets
dataset_raw = training_raw.append(test_raw)
dataset_raw.reset_index(inplace=True)
dataset_raw.drop('index',inplace=True,axis=1)

dataset_bin = pd.DataFrame() # To contain our dataframe with our discretised continuous variables 
dataset_con = pd.DataFrame() # To contain our dataframe with our continuous variables 

dataset_bin['y'] = dataset_raw['y']
dataset_con['y'] = dataset_raw['y']

#Feature:age
dataset_bin['age'] = pd.cut(dataset_raw['age'], 12) 
dataset_con['age'] = dataset_raw['age'] # non-discretised

#Feature:job
dataset_bin['job'] = dataset_raw['job']
dataset_con['job'] = dataset_raw['job']


#feature marital
dataset_bin['marital'] = dataset_raw['marital']
dataset_con['marital'] = dataset_raw['marital']

#feature education
dataset_bin['education'] = dataset_raw['education']
dataset_con['education'] = dataset_raw['education']


#feature housing
dataset_bin['housing'] = dataset_raw['housing']
dataset_con['housing'] = dataset_raw['housing']


#feature loan
dataset_bin['loan'] = dataset_raw['loan']
dataset_con['loan'] = dataset_raw['loan']



#feature contact
dataset_bin['contact'] = dataset_raw['contact']
dataset_con['contact'] = dataset_raw['contact']


#feature month
dataset_bin['month'] = dataset_raw['month']
dataset_con['month'] = dataset_raw['month']

# One Hot Encodes 
one_hot_cols = dataset_bin.columns.tolist()
one_hot_cols.remove('y')
dataset_bin_enc = pd.get_dummies(dataset_bin, columns=one_hot_cols)

dataset_bin_enc.head()

# Label Encode 
dataset_con_test = dataset_con
dataset_con_enc = dataset_con_test.apply(LabelEncoder().fit_transform)
dataset_con_enc.head()

# create the relation-graph between 2 dataset
def show():
   plt.style.use('seaborn-whitegrid')
   fig = plt.figure(figsize=(25,10)) 

   plt.subplot(1, 2, 1)

   mask = np.zeros_like(dataset_bin_enc.corr(), dtype=np.bool)
   mask[np.triu_indices_from(mask)] = True
   sns.heatmap(dataset_bin_enc.corr(), 
               vmin=-1, vmax=1, 
               square=True, 
               cmap=sns.color_palette("rainbow", 100), 
               mask=mask, 
               linewidths=.5);

   mask = np.zeros_like(dataset_con_enc.corr(), dtype=np.bool)
   mask[np.triu_indices_from(mask)] = True
   corrmat = dataset_con_enc.corr()
   f,ax = plt.subplots(figsize=(12,9))
   ax.set_xticklabels(corrmat,rotation='horizontal')
   #ax.xaxis.tick_top()
   ax.set_facecolor('xkcd:salmon')
   ax.set_facecolor((1.0, 1.0, 1.0))
   
   sns.heatmap(corrmat, 
               vmin=-1, vmax=1, 
               square=True, 
               annot =True,
               cmap=sns.color_palette("RdPu", 100), 
               mask=mask, 
               linewidths=.5,
            linecolor = 'grey');
   label_y = ax.get_yticklabels()
   plt.setp(label_y , rotation = 360)
   label_x = ax.get_xticklabels()
   plt.setp(label_x , rotation = 90)
   plt.title('correlation coefficients',fontsize='large')
   f.savefig('./result_plots/relation.jpg', dpi=100, bbox_inches='tight')

   # Using Random Forest to gain an insight on Feature Importance
   clf = RandomForestClassifier()
   clf.fit(dataset_con_enc.drop('y', axis=1), dataset_con_enc['y'])

   plt.style.use('seaborn-whitegrid')
   importance = clf.feature_importances_
   importance = pd.DataFrame(importance, index=dataset_con_enc.drop('y', axis=1).columns, columns=["Importance"])
   importance.sort_values(by='Importance', ascending=True).plot(kind='barh', figsize=(5,len(importance)/2),edgecolor='black');
   plt.title('feature importance',fontsize='large')
   plt.xlim((0, 0.5))
   plt.savefig('./result_plots/importance.jpg',figuresize=(18.5,10),dpi=100, bbox_inches='tight')