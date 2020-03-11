import calendar
import pandas as pd
import numpy as np
from typing import Union, Iterable, Callable, List

import pre_processing as pp


class TimeAnalysis:
    
    def __init__(self, csv_path):
        self.df = pp.load_data(csv_path)
        self.df.process_all()

    def get_probabilities(self, column) -> pd.DataFrame:
        assert column in self.df.keys()
        return self.df[[column, 'y']].groupby(by=column).mean().reset_index()

    def get_success_count(self, column) -> pd.DataFrame:
        assert column in self.df.keys()
        return self.df[[column, 'y']].groupby(column)['y'].sum().reset_index()

    def get_count(self, column) -> pd.DataFrame:
        assert column in self.df.keys()
        return self.df[[column, 'y']].groupby(column)['y'].count().reset_index()

    def filter_count(self, column) -> pd.DataFrame:
        assert column in self.df.keys()
        return self.df[[column, 'y']].groupby(column)['y'].count().reset_index()

    def get_yes_no_count(self, column) -> pd.DataFrame:
        assert column in self.df.keys()
        return self.df[column].groupby
    
    def get_column(self, column) -> pd.DataFrame:
        assert column in self.df.keys()
        return self.df[column]
    
    
    
def map_age(csv_path):
    labels = ['(16, 20)','(21, 30)','(31, 40)','(41, 50)','(51, 60)','(61, 70)','(71, 80)','(81, 90)','(91, 100)']
    dataset_con = pd.DataFrame() 
    dataset_raw = pp.load_data(csv_path)
    dataset_con['y'] = dataset_raw['y']
    dataset_con['age'] = dataset_raw['age']
    dataset_con['y']= np.round(dataset_con['age'])
    dataset_con['interval'] = dataset_con['y'].map(dataset_raw.age_dict)
    counts = []

    for i in labels:
        counts.append(dataset_raw.loc[dataset_con['interval'] == i]['y'].value_counts())
    
    return counts, labels

def get_age_prob_success(data):
    result = []
    for item in data:
        result.append(item[1]/(item[1]+item[0])* 100)
    return result

def number_to_day_of_week(df: Union[pd.DataFrame, pd.Series, Iterable]) -> Union[pd.DataFrame, pd.Series, Iterable]:
    """
    Returns a DataFrame, Series, or Iterable with integers converted to the appropriate abbreviated day of the week.
    0 returns an empty string. Values outside [0 6] will raise an `IndexError`

    :param df: a `pandas.DataFrame` or `pandas.Series` object with integer values ranging from 0 to 6
    """
    def func(x):
        return calendar.day_abbr[x]

    result = _apply(df, func)
    return result


def number_to_month(df: Union[pd.DataFrame, pd.Series, Iterable]) -> Union[pd.DataFrame, pd.Series, Iterable]:
    """
    Returns a DataFrame, Series, or Iterable with integers converted to the appropriate abbreviated month. 0 returns
    an empty string. Values outside [0 12] will raise an `IndexError`

    :param df: a `pandas.DataFrame` or `pandas.Series` object with integer values ranging from 0 to 12
    """
    def func(x):
        return calendar.month_abbr[x]

    result = _apply(df, func)
    return result


def _apply(x: Union[pd.DataFrame, pd.Series, Iterable], func: Callable) -> List:
    """
    Iteratively applies function `func` to `x`.

    :param x:  A pandas DataFrame, Series or an iterable
    :param func: a callable to be applied to x without arguments
    :return: a list containing the results
    """
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        result = x.apply(func)
    elif isinstance(x, Iterable):
        result = map(func, x)
    else:
        raise TypeError(f"_apply takes Dataframe, Series, or Iterables, not {type(x)}")
    return list(result)