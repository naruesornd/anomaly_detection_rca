import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
class TimeSeriesDatasets(Dataset):

    def __init__(self, X,y, seq_len=12):
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len
    
    def __getitem__(self, idx:int):
        x = self.X[idx:idx+self.seq_len]
        y = self.y[idx+self.seq_len]
        return x,y
    
