import pandas as pd
import numpy as np
import calendar
from typing import Union, Iterable, Callable, List


class DfBankAdditional(pd.DataFrame):
    """
    To be used with the bank-additional dataset from:
    https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#

    Pass a pandas.DataFrame to the constructor to get started
    """
    NO_INCOME = 'no income'
    LOWER_INCOME = 'lower income'
    HIGHER_INCOME = 'higher income'

    DROPOUT = 'Dropout'

    mappings = {
        'y': {
            'yes': 1,
            'no': 0
        },
        'poutcome': {
            'nonexistent': np.NaN,
            'failure': 0,
            'success': 1
        },
        'job': {
            'housemaid': LOWER_INCOME,
            'services': LOWER_INCOME,
            'blue-collar': LOWER_INCOME,
            'self-employed': LOWER_INCOME,
            'retired': NO_INCOME,
            'student': NO_INCOME,
            'unemployed': NO_INCOME,
            'admin': HIGHER_INCOME,
            'admin.': HIGHER_INCOME,
            'technician': HIGHER_INCOME,
            'management': HIGHER_INCOME,
            'entrepreneur': HIGHER_INCOME,
            'unknown': 'unknown',
        },
        'education': {
            'basic.4y': DROPOUT,
            'high.school': DROPOUT,
            'basic.6y': DROPOUT,
            'basic.9y': DROPOUT,
            'professional.course': 'professional.course',
            'unknown': 'unknown',
            'university.degree': 'university.degree',
            'illiterate': 'illiterate',
        },
        'day_of_week': dict(zip(map(str.lower, calendar.day_abbr), range(7))),
        'month': dict(zip(map(str.lower, calendar.month_abbr), range(0, 13))),
    }

    def process_all(self):
        """
        The dataframe is modified in-place, replacing unknown values with np.NaN, false values with `0`, and true
        values with `1`.

        This behavior can be modified by changing the class attribute `mappings`
        """
        for c in self.keys():
            if c in self.mappings.keys():
                self.re_map_column(c)

        self._validate_all()

    def re_map_column(self, column: str):
        for k, v in self.mappings[column].items():
            self[column].replace(k, v, inplace=True)

    def _validate_all(self):
        """
        Checks that our assumptions about the structure of the data are correct. Raises and AssertionError if an
        unexpected datatype or value is found.
        """
        for c in self.keys():
            if c in self.mappings.keys():
                self._validate(c)

    def _validate(self, column):
        if not self[column].isin(self.mappings[column].values()).all():
            map_value_set = set(self.mappings[column].values())
            self_value_set = set(self[column].values)
            missing_values = self_value_set.difference(map_value_set)
            raise ValueError(f'{column} contains values not found in mapping: {missing_values}')


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


def load_data(path: str) -> DfBankAdditional:
    """
    Loads the csv located at `path`
    :param path: a string path to the bank-additional-full.csv file
    :return: a DfBankAdditional
    """
    return DfBankAdditional(pd.read_csv(path, sep=';'))
