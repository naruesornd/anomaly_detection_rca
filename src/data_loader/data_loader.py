import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .data_set import TimeSeriesDatasets
def time_series_loader(X, y, seq_len, batch_size=64, shuffle=True) -> DataLoader:

    dataset = TimeSeriesDatasets(X=X, y=y, seq_len=seq_len)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)