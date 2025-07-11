import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
class TimeSeriesDataset(Dataset):
        def __init__(self, X, y, seq_len):
            self.X = X
            self.y = y
            self.seq_len = seq_len

        def __len__(self):
            return len(self.X) - self.seq_len + 1

        def __getitem__(self, idx):
            X_seq = self.X[idx:idx+self.seq_len]
            y_seq = self.y[idx+self.seq_len-1]
            return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)


class TestTimeSeriesDataset(Dataset):
        def __init__(self, X, y, cycle_id, cycle_time, seq_len):
            self.X = X
            self.y = y
            self.cycle_id = cycle_id
            self.cycle_time = cycle_time
            self.seq_len = seq_len

        def __len__(self):
            return len(self.X) - self.seq_len + 1

        def __getitem__(self, idx):
            X_seq = self.X[idx:idx+self.seq_len]
            y_seq = self.y[idx+self.seq_len-1]
            cid = self.cycle_id[idx+self.seq_len-1]
            ctime = self.cycle_time[idx+self.seq_len-1]
            return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32), cid, ctime
    
