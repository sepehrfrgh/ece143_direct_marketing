import calendar
from typing import Union, Iterable, Callable, List

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import pre_processing as pp


class Analysis:
    
    def __init__(self, csv_path):
        '''
        Loads the Dataframe from the csv_path.
        Uses the user defined pre_processing module's load_data() function to load the csv file.
        After loading the entries in the various columns of the dataframe are processed and validated.
        Processing involves replacing unknown values with np.NaN.
        Validation checks if the column values in the data frame are compliant with the mappings of the data provided.
        :param csv_path: a string that contains the path to the dataframe.
        '''
        assert isinstance(csv_path, str)
        self.processing = pp.load_data(csv_path)
        self.processing.process_all()
        self.df = self.processing.df

    def get_probabilities(self, column) -> pd.DataFrame:
        '''
        Returns the probability of a customer saying yes based on the column attribute passed
        :param column: column index for which we want to compute probability
        '''
        
        assert column in self.df.keys()
        return self.df[[column, 'y']].groupby(by=column).mean().reset_index()

    def get_success_count(self, column) -> pd.DataFrame:
        '''
        Returns the number of a customers who say yes based on the column attribute passed
        :param column: column index for which we want to compute success count(number of yes)
        '''
        
        assert column in self.df.keys()
        return self.df[[column, 'y']].groupby(column)['y'].sum().reset_index()

    def get_count(self, column) -> pd.DataFrame:
        '''
        Returns the number total number of a customers who have been contacted
        :param column: column index for which we want to compute total count
        '''
        
        assert column in self.df.keys()
        return self.df[[column, 'y']].groupby(column)['y'].count().reset_index()

    def get_yes_no_count(self, column) -> pd.DataFrame:
        '''
        Returns the number total number of a customers who said yes and total number of customers who said no
        :param column: column index for which we want to compute
        '''
        assert column in self.df.keys()
        return self.df[column].groupby
    
    def get_column(self, column) -> pd.DataFrame:
        '''
        Returns a specific column from the dataframe
        :param column: column index which we want to be queried
        '''
        
        assert column in self.df.keys()
        return self.df[column]
        
    def percentage_of_population(self, column):
        '''
        Returns the number total number of a customers who have been contacted
        :param column: column index for which we want to compute probability
        '''
    
        assert column in self.df.keys()
        total = self.df[column].count()
        percentages = [np.around(i*100/total, 2) for i in self.df[column].value_counts()]
        return percentages
    
    
    def map_age(self, csv_path):
        '''
        Returns the mapping of age in the dataset to age groups for instance a 17 year old will be mapped to (16-20) category
        :param csv_path: a string that contains the path to the dataframe.
        '''
        labels = ['(16, 20)','(21, 30)','(31, 40)','(41, 50)','(51, 60)','(61, 70)','(71, 80)','(81, 90)','(91, 100)']
        dataset_con = pd.DataFrame()
        
        dataset_con['y'] = self.df['y']
        dataset_con['age'] = self.df['age']
        dataset_con['y']= np.round(dataset_con['age'])
        dataset_con['interval'] = dataset_con['y'].map(self.processing.age_dict)
        counts = []

        for i in labels:
            counts.append(self.df.loc[dataset_con['interval'] == i]['y'].value_counts())
    
        return counts, labels

    def get_age_prob_success(self, data):
        '''
        Returns the probability that a particular age group would subscribe to the term deposit.
        :param data: its the output of map_age
        '''
        result = []
        for item in data:
            result.append(item[1]/(item[1]+item[0])* 100)
        return result

    @property
    def column_list(self):
        return self.df.columns.tolist()


class MaritalAnalysis(Analysis):
    def __init__(self, csv_path):
        super(MaritalAnalysis, self).__init__(csv_path)
        self.df = self.df.loc[self.df['marital'] != 'unknown', ['marital', 'y']]


class FeatureAnalysis(Analysis):
    age_bin_width = 12
    features = ['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'y']

    def __init__(self, csv_path):
        super(FeatureAnalysis, self).__init__(csv_path)
        self.df = self.df[self.features]

        self.df_binned = self.df
        self.df_binned['age'] = pd.cut(self.df['age'], self.age_bin_width)
        self.df_encoded = self.df.apply(LabelEncoder().fit_transform)

    def get_feature_importance(self):
        clf = RandomForestClassifier()
        clf.fit(self.df_encoded.drop('y', axis=1), self.df_encoded['y'])
        importance = pd.DataFrame(clf.feature_importances_, index=self.df_encoded.drop('y', axis=1).columns,
                                  columns=["Importance"])
        return importance.sort_values(by='Importance', ascending=True)

    @property
    def one_hot_columns(self):
        return self.df.columns.tolist().remove('y')



def number_to_day_of_week(df: Union[pd.DataFrame, pd.Series, Iterable]) -> Union[pd.DataFrame, pd.Series, Iterable]:
    '''
    Returns a DataFrame, Series, or Iterable with integers converted to the appropriate abbreviated day of the week.
    0 returns an empty string. Values outside [0 6] will raise an `IndexError`

    :param df: a `pandas.DataFrame` or `pandas.Series` object with integer values ranging from 0 to 6
   '''
    def func(x):
        return calendar.day_abbr[x]

    result = _apply(df, func)
    return result


def number_to_month(df: Union[pd.DataFrame, pd.Series, Iterable]) -> Union[pd.DataFrame, pd.Series, Iterable]:
    '''
    Returns a DataFrame, Series, or Iterable with integers converted to the appropriate abbreviated month. 0 returns
    an empty string. Values outside [0 12] will raise an `IndexError`

    :param df: a `pandas.DataFrame` or `pandas.Series` object with integer values ranging from 0 to 12
    '''
    def func(x):
        return calendar.month_abbr[x]

    result = _apply(df, func)
    return result


def _apply(x: Union[pd.DataFrame, pd.Series, Iterable], func: Callable) -> List:
    '''
    Iteratively applies function `func` to `x`.

    :param x:  A pandas DataFrame, Series or an iterable
    :param func: a callable to be applied to x without arguments
    :return: a list containing the results
    '''
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        result = x.apply(func)
    elif isinstance(x, Iterable):
        result = map(func, x)
    else:
        raise TypeError(f"_apply takes Dataframe, Series, or Iterables, not {type(x)}")
    return list(result)
