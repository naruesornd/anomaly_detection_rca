import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CycleProcessor:
    def __init__(self, column_name, df, min_cycle_length=70, threshold=0.05):
        self.cycles = []
        self.min_cycle_length = min_cycle_length
        self.threshold = threshold
        self.column_name = column_name
        self.df = df

    def identify_cycles(self):
        if self.df is not None:
            target_feature = self.df[self.column_name].values
            diffs = np.abs(np.diff(target_feature))
            change_points = np.where(diffs > self.threshold)[0] + 1

            cycles = []
            start = 0
            for cp in change_points:
                if cp - start >= self.min_cycle_length:
                    cycles.append((start, cp))
                    start = cp
                else:
                    start = cp
            if len(target_feature) - start >= self.min_cycle_length:
                cycles.append((start, len(target_feature)))
            self.cycles = cycles
        else:
            print('No data loaded')
    def assign_cycle_features(self, cycle_offset=0):
        if self.df is not None:
            self.df["cycle_id"] = -1
            self.df["cycle_time"] = -1

            for i, (start, end) in enumerate(self.cycles):
                self.df.loc[self.df.index[start:end], "cycle_id"] = i + cycle_offset
                self.df.loc[self.df.index[start:end], "cycle_time"] = np.arange(end - start)
            return self.df
        else:
            print('No data loaded')
            return None
        
    def export_files(self, file_path):
        if self.df is not None:
            self.df.reset_index(inplace=True)
            self.df.to_csv(file_path, index=False)
        else:
            print('No data loaded')