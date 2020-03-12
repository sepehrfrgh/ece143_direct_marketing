import pandas as pd
import numpy as np
import calendar


class DfBankAdditional:
    '''
    To be used with the bank-additional dataset from:
    https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#

    Pass a pandas.DataFrame to the constructor to get started
    '''
    NO_INCOME = 'no income'
    LOWER_INCOME = 'lower income'
    HIGHER_INCOME = 'higher income'

    
    DROPOUT = 'Dropout'
    age_dict = {}
    i=0
    for i in range(15, 99):
        if i >= 15 and i <= 20:
            age_dict[i] = '(16, 20)'
        elif i >= 21 and i <= 30:
            age_dict[i] = '(21, 30)'
        elif i >= 31 and i <= 40:
            age_dict[i] = '(31, 40)'
        elif i >= 41 and i <= 50:
            age_dict[i] = '(41, 50)'
        elif i >= 51 and i <= 60:
            age_dict[i] = '(51, 60)'
        elif i >= 61 and i <= 70:
            age_dict[i] = '(61, 70)'
        elif i >= 71 and i <= 80:
            age_dict[i] = '(71, 80)'
        elif i >= 81 and i <= 90:
            age_dict[i] = '(81, 90)'
        else:
            age_dict[i] = '(91, 100)'
    mappings = {

        'marital_status_mapping' : {'single'  : 'single',
        'married' : 'married',
        'divorced': 'divorced',
        'unknown' : np.NaN},
        
        'age1' : age_dict,
      

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

    def __init__(self, filepath):
        self.df = pd.DataFrame(filepath)

    def process_all(self):
        '''
        The dataframe is modified in-place, replacing unknown values with np.NaN, false values with `0`, and true
        values with `1`.

        This behavior can be modified by changing the class attribute `mappings`
        '''
        for c in self.df.keys():
            if c in self.mappings.keys():
                self.re_map_column(c)

        self._validate_all()

    def re_map_column(self, column: str):
        '''
        Applies the mapping in the `self.mappings` dict to the column name in `column`
        
        :param column: name of the `pd.DataFrame` column to re-map
        '''
        assert isinstance(column, str)
        for k, v in self.mappings[column].items():
            self.df[column].replace(k, v, inplace=True)

    def _validate_all(self):
        '''
        Checks that our assumptions about the structure of the data are correct. Raises and AssertionError if an
        unexpected datatype or value is found.
        '''
        for c in self.df.keys():
            if c in self.mappings.keys():
                self._validate(c)

    def _validate(self, column: str):
        '''
        Checks that all data entries in `column` can be found in the `self.mappings` dict

        :param column: The DataFrame column to validate
        '''
        assert isinstance(column, str)
        if not self.df[column].isin(self.mappings[column].values()).all():
            map_value_set = set(self.mappings[column].values())
            self_value_set = set(self.df[column].values)
            missing_values = self_value_set.difference(map_value_set)
            raise ValueError(f'{column} contains values not found in mapping: {missing_values}')


def load_data(path: str) -> DfBankAdditional:
    '''
    Loads the csv located at `path`
    :param path: a string path to the bank-additional-full.csv file
    :return: a DfBankAdditional
    '''
    return DfBankAdditional(pd.read_csv(path, sep=';'))
