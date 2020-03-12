import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from analysis import FeatureAnalysis
import config

feature_analysis = FeatureAnalysis(config.bank_additional_path)


plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(25, 10))

plt.subplot(1, 2, 1)

mask = np.zeros_like(feature_analysis.df_encoded.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(feature_analysis.df_encoded.corr(),
            vmin=-1, vmax=1,
            square=True,
            cmap=sns.color_palette("rainbow", 100),
            mask=mask,
            linewidths=.5)

correlation_matrix = feature_analysis.df_encoded.corr()
f, ax = plt.subplots(figsize=(12, 9))
ax.set_xticklabels(correlation_matrix, rotation='horizontal')
ax.set_facecolor('xkcd:salmon')
ax.set_facecolor((1.0, 1.0, 1.0))

sns.heatmap(correlation_matrix,
            vmin=-1, vmax=1,
            square=True,
            annot=True,
            cmap=sns.color_palette("RdPu", 100),
            mask=mask,
            linewidths=.5,
            linecolor='grey')
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360)
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=90)
plt.title('correlation coefficients', fontsize='large')

# Using Random Forest to gain an insight on Feature Importance
importance = feature_analysis.get_feature_importance()
importance.plot(kind='barh', figsize=(5, len(importance) / 2), edgecolor='black');
plt.title('feature importance', fontsize='large')
plt.xlim((0, 0.5))
plt.show()
