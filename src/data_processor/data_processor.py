import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = self.load_data()
        self.df = self.df.reset_index()
        self.columns = self.df.columns.tolist()
    """load data from file"""
    def load_data(self):
        if self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.xlsx'):
            df = pd.read_excel(self.data_path)
        else:
            raise ValueError('Unsupported file format')
        return df
    
    """summary of data"""
    def list_columns(self,print_columns=True):
        if self.df is not None:
            if print_columns:
                print(self.columns)
            return self.columns
        else:
            if print_columns:
                print('No data loaded')
            return None

    def summary(self):
        if self.df is not None:
            print(self.df.describe())
        else:
            print('No data loaded')

    def head(self):
        if self.df is not None:
            print(self.df.head())
        else:
            print('No data loaded')

    def change_pivot(self,index,columns,values):
        if self.df is not None:
            self.df = pd.pivot_table(self.df, index=index, columns=columns, values=values)
            self.df = self.df.reset_index()
            self.columns = self.df.columns.tolist()
        else:
            print('No data loaded')

    def drop_NA_with_feature(self, features):
        if self.df is not None:
            self.df = self.df.dropna(subset=features)
        else:
            print('No data loaded')
    
    def sum_of_NA_rows(self,features):
        if self.df is not None:
            print(self.df[features].isnull().sum())
        else:
            print('No data loaded')
    
    def num_of_row(self):
        if self.df is not None:
            print(self.df.shape[0])
        else:
            print('No data loaded')

    def rename_column_to_timestamp(self, original_column_name):
        """
        将指定列名重命名为 'timestamp'
        """
        if original_column_name not in self.df.columns:
            raise ValueError(f"列 '{original_column_name}' 不在 DataFrame 中")
        self.df = self.df.rename(columns={original_column_name: 'timestamp'})
        self.df['timestamp'] = self.df['timestamp'].astype(str).str.strip()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        self.columns = self.df.columns.tolist()  # 更新列名缓存

    def export_to_csv(self, output_path):
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
        else:
            print('No data loaded')