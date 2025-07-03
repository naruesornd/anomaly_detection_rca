import pandas as pd
import numpy as np
from itertools import combinations
from data_processor import DataProcessor
from datetime import datetime
class FeatureEngineering:
    def __init__(self,dp:DataProcessor):
        self.dp = dp
        self.df = dp.df
        self.cross_features = []
        self.features = self.dp.list_columns(print_columns=False)
        self.features = [f for f in self.features if f not in ['timestamp']]
    def generate_cross_features(self, drop_features=[]):
        features = [f for f in self.features if f not in drop_features]
        new_cols = {}
        for f1,f2 in combinations(features,2):
            new_cols[f1 + '_x_' + f2] = self.df[f1] * self.df[f2]
        self.cross_features = new_cols.keys()
        self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)
        return self.cross_features
    
    def cycle_time_engineer(self):
        new_cols = {
        'hour_sin': np.sin(2 * np.pi * self.df['timestamp'].dt.hour / 24),
        'hour_cos': np.cos(2 * np.pi * self.df['timestamp'].dt.hour / 24),
        'day_sin':  np.sin(2 * np.pi * self.df['timestamp'].dt.dayofyear / 365),
        'day_cos':  np.cos(2 * np.pi * self.df['timestamp'].dt.dayofyear / 365),
        }
        self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)

    def lag_engineer(self):
        new_cols = {}
        for f in self.features:
            for lag in [1,2,3,6,12]:
                new_cols[f + '_lag_' + str(lag)] = self.df[f].shift(lag)
        self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)

    def rolling_mean_engineer(self):
        new_cols = {}
        for f in self.features:
            for window in [1,2,3,6,12]:
                new_cols[f + '_rolling_mean_' + str(window)] = self.df[f].rolling(window).mean()
        self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)

    def rolling_std_engineer(self):
        new_cols = {}
        for f in self.features:
            for window in [1,2,3,6,12]:
                new_cols[f + '_rolling_std_' + str(window)] = self.df[f].rolling(window).std()
        self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)


