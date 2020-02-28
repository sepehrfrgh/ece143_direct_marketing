import pandas as pd
import numpy as np
from typing import Union, Iterable, Callable, List

import pre_processing as pp
class maritalStatus:
    def __init__(self, csv_path):
        '''
        Loads the Dataframe from the csv_path.
        Uses the user defined pre_processing module's load_data() function to load the csv file.
        After loading the entries in the various columns of the dataframe are processed and validated.
        Processing involves replacing unknown values with np.NaN.
        Validation checks if the column values in the data frame are compliant with the mappings of the data provided.
        
        Input: csv_path which is a string.
        '''
        
        assert isinstance(csv_path, str)
        self.df = pp.load_data(csv_path)
        self.df.process_all()
        
    def get_probabilities(self, column) -> pd.DataFrame:
        '''
        Returns the probability of a customer saying yes based on the column attribute passed
        Input: column index for which we want to compute probability
        '''
        
        assert column in self.df.keys()
        return self.df[[column, 'y']].groupby(by=column).mean().reset_index()
    def get__total_probabilities(self, column) -> pd.DataFrame:
        '''
        Returns the probability of a customer saying yes based on the column attribute passed
        Input: column index for which we want to compute probability
        '''
        
        assert column in self.df.keys()
        total = self.df.shape[0]
        col_vals = self.df[column].unique()
        print(col_vals)
        prob_list = []
        for i in col_vals:
            if i == 'nan':
                break
            #m.loc[m['marital'] == 'divorced']['y']
            prob_list.append(self.df.loc[self.df[column]== i]['y'].value_counts()[1]/total)
    
        return prob_list
        #return self.df[[column, 'y']].groupby(by=column).mean().reset_index()    

    def get_success_count(self, column) -> pd.DataFrame:
        '''
        Returns the number of a customers who say yes based on the column attribute passed
        Input: column index for which we want to compute probability
        '''
        
        assert column in self.df.keys()
        return self.df[[column, 'y']].groupby(column)['y'].sum().reset_index()

    def get_count(self, column) -> pd.DataFrame:
        '''
        Returns the number total number of a customers who have been contacted
        Input: column index for which we want to compute probability
        '''
        
        assert column in self.df.keys()
        return self.df[[column, 'y']].groupby(column)['y'].count().reset_index()

    def percentage_of_population(self, column):
        '''
        Returns the number total number of a customers who have been contacted
        Input: column index for which we want to compute probability
        ''' 
        
        assert column in self.df.keys()
        total = self.df[column].count()
        percentages = [np.around(i*100/total, 2) for i in self.df[column].value_counts()]
        return percentages