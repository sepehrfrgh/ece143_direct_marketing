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
    y_mapping = {'yes': 1,
                 'no': 0}
    poutcome_mapping = {'nonexistent': np.NaN,
                        'failure': 0,
                        'success': 1}
    day_of_week_mapping = dict(zip(map(str.lower, calendar.day_abbr), range(7)))
    month_mapping = dict(zip(map(str.lower, calendar.month_abbr), range(0, 13)))

    def process_all(self):
        """
        The dataframe is modified in-place, replacing unknown values with np.NaN, false values with `0`, and true
        values with `1`.

        This behavior can be modified by changing the class attributes:
            `y_mapping`
            `poutcome_mapping`
            `days_of_week_mapping`
        """
        self.process_y()
        self.process_poutcome()
        self.process_day_of_week()
        self.process_month()

        self._validate_all()

    def process_month(self):
        for k, v in self.month_mapping.items():
            self['month'].replace(k, v, inplace=True)

    def process_day_of_week(self):
        for k, v in self.day_of_week_mapping.items():
            self['day_of_week'].replace(k, v, inplace=True)

    def process_poutcome(self):
        for k, v in self.poutcome_mapping.items():
            self['poutcome'].replace(k, v, inplace=True)

    def process_y(self):
        for k, v in self.y_mapping.items():
            self['y'].replace(k, v, inplace=True)

    def _validate_all(self):
        """
        Checks that our assumptions about the structure of the data are correct. Raises and AssertionError if an
        unexpected datatype or value is found.
        """
        self._validate_y()
        self._validate_poutcome()
        self._validate_day_of_week()
        self._validate_month()

    def _validate_month(self):
        assert self['month'].isin(self.month_mapping.values()).all()

    def _validate_day_of_week(self):
        assert self['day_of_week'].isin(self.day_of_week_mapping.values()).all()

    def _validate_poutcome(self):
        assert self['poutcome'].isin(self.poutcome_mapping.values()).all()

    def _validate_y(self):
        assert self['y'].isin(self.y_mapping.values()).all()


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
        raise TypeError(f"_apply takes Datframe, Series, or Iterables, not {type(df)}")
    return list(result)
