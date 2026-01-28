import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CycleProcessor:
    """
    Identify cycles on a 1D signal (e.g. FeedFlow) and assign:
      - cycle_id: which cycle it belongs to (0,1,2,...)
      - cycle_time: index within that cycle (0,1,2,...)

    Cycles are split whenever the absolute change between
    consecutive points exceeds `threshold`, and each cycle
    must have length >= min_cycle_length.
    """
    def __init__(self, df, column_name,
                 threshold,                   # e.g. 10 for FeedFlow
                 min_cycle_length=70):
        
        self.df = df
        self.column_name = column_name
        self.threshold = threshold
        self.min_cycle_length = min_cycle_length
        self.cycles = []   # list of (start_idx, end_idx)

    def identify_cycles(self):
        """
        Find cycle boundaries and populate self.cycles
        as a list of (start_idx, end_idx) on df.index positions.
        """
        if self.df is None or self.column_name not in self.df.columns:
            raise ValueError("DataFrame or column_name is not valid")

        values = self.df[self.column_name].values

        # 1-step differences
        diffs = np.abs(np.diff(values))
        # indices where we consider a new cycle starts
        change_points = np.where(diffs > self.threshold)[0] + 1

        cycles = []
        start = 0
        for cp in change_points:
            if cp - start >= self.min_cycle_length:
                cycles.append((start, cp))
                start = cp
            else:
                # too short, just move the start forward
                start = cp

        # last cycle until end
        if len(values) - start >= self.min_cycle_length:
            cycles.append((start, len(values)))

        # map from position to actual df index
        index_array = self.df.index.to_numpy()
        self.cycles = [(index_array[s], index_array[e-1]) for s, e in cycles]
        return self.cycles

    def assign_cycle_features(self, cycle_offset=0):
        """
        Adds two columns to self.df:
          - cycle_id
          - cycle_time
        cycle_offset lets you continue numbering from a previous file.
        """
        if not self.cycles:
            # if user forgot to call identify_cycles()
            self.identify_cycles()

        # initialize as unassigned
        self.df["cycle_id"] = -1
        self.df["cycle_time"] = -1

        for i, (start_idx, end_idx) in enumerate(self.cycles):
            mask = (self.df.index >= start_idx) & (self.df.index <= end_idx)
            cycle_len = mask.sum()
            self.df.loc[mask, "cycle_id"] = i + cycle_offset + 1
            self.df.loc[mask, "cycle_time"] = np.arange(1, cycle_len+1)

        # drop unassinged rows (cycle_id is still NA)
        self.df = self.df.dropna(subset=['cycle_id']).reset_index(drop=True)
        
        return self.df

    @staticmethod
    def plot_cycle(df, cycle_id, column_name='FeedFlow'):
        cdf = df[df['cycle_id'] == cycle_id].copy()
        if cdf.empty:
            print(f"No rows found for cycle_id={cycle_id}")
            return
        plt.figure(figsize=(10, 4))
        plt.plot(cdf['cycle_time'], cdf[column_name], marker='o', linewidth=1.5)
        plt.title(f'{column_name} for cycle_id = {cycle_id}')
        plt.xlabel('cycle_time')
        plt.ylabel(column_name)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()


    def export_files(self, path):
        self.df.to_csv(path, index=False)



########### OLD CODE ###########
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# class CycleProcessor:
#     def __init__(self, column_name, df, min_cycle_length=70, threshold=0.05):
#         self.cycles = []
#         self.min_cycle_length = min_cycle_length
#         self.threshold = threshold
#         self.column_name = column_name
#         self.df = df

#     def identify_cycles(self):
#         if self.df is not None:
#             target_feature = self.df[self.column_name].values
#             diffs = np.abs(np.diff(target_feature))
#             change_points = np.where(diffs > self.threshold)[0] + 1

#             cycles = []
#             start = 0
#             for cp in change_points:
#                 if cp - start >= self.min_cycle_length:
#                     cycles.append((start, cp))
#                     start = cp
#                 else:
#                     start = cp
#             if len(target_feature) - start >= self.min_cycle_length:
#                 cycles.append((start, len(target_feature)))
#             self.cycles = cycles
#         else:
#             print('No data loaded')
#     def assign_cycle_features(self, cycle_offset=0):
#         if self.df is not None:
#             self.df["cycle_id"] = pd.NA
#             self.df["cycle_time"] = pd.NA

#             for i, (start, end) in enumerate(self.cycles):
#                 self.df.loc[self.df.index[start:end], "cycle_id"] = i + cycle_offset
#                 self.df.loc[self.df.index[start:end], "cycle_time"] = np.arange(end - start)
#             return self.df
#         else:
#             print('No data loaded')
#             return None
        
#     def export_files(self, file_path):
#         if self.df is not None:
#             self.df.reset_index(inplace=True)
#             self.df.to_csv(file_path, index=False)
#         else:
#             print('No data loaded')
