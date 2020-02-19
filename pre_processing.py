import pandas as pd
import numpy as np
import calendar
from typing import Union, Iterable


class DfBankAdditional(pd.DataFrame):
    """
    To be used with the bank-additional dataset from:
    https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#

    Pass a pandas.DataFrame to the constructor to get started
    """
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
            'housemaid': 'lower income',
            'services': 'lower income',
            'blue-collar': 'lower income',
            'unknown': 'lower income',
            'self-employed': 'lower income',
            'retired': 'no income',
            'student': 'no income',
            'admin': 'higher income',
            'technician': 'higher income',
            'management': 'higher income',
            'entrepreneur': 'higher income'
        },
        'education': {
            'basic.4y': 'Dropout',
            'high.school': 'Dropout',
            'basic.6y': 'Dropout',
            'basic.9y': 'Dropout',
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
        self.re_map_column('y')
        self.re_map_column('poutcome')
        self.re_map_column('day_of_week')
        self.re_map_column('month')
        self.re_map_column('job')
        self.re_map_column('education')

        self._validate_all()

    def re_map_column(self, column: str):
        for k, v in self.mappings[column].items():
            self[column].replace(k, v, inplace=True)

    def _validate_all(self):
        """
        Checks that our assumptions about the structure of the data are correct. Raises and AssertionError if an
        unexpected datatype or value is found.
        """
        self._validate('month')
        self._validate('poutcome')
        self._validate('day_of_week')
        self._validate('month')

    def _validate(self, column: str):
        assert self[column].isin(self.mappings[column].values()).all()


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


def _apply(df, func):
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        result = df.apply(func)
    elif isinstance(df, Iterable):
        result = map(func, df)
    else:
        raise TypeError(f"_apply takes Dataframe, Series, or Iterables, not {type(df)}")
    return list(result)


def load_data(path: str) -> DfBankAdditional:
    """

    :param path:
    :return:
    """
    return DfBankAdditional(pd.read_csv(path, sep=';'))
