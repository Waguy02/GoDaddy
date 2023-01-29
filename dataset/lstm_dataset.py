import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from constants import DATA_DIR
from  import DatasetType


class LstmDataset(Dataset):
    def __init__(self, type, seq_len, stride=1):
        self.type = type
        self.seq_len = seq_len
        self.stride = stride

        self.file = os.path.join(DATA_DIR, f"train_with_census_{'train' if type==DatasetType.TRAIN else 'val' if type==DatasetType.VALID  else 'test'}.csv")
        self.load_data()

    def init_transforms(self):
        """
        Initialize transforms.Might be different for each dataset type
        """

    def load_data(self):
        """
        Load data from the data items if necessary
        """
        self.data = pd.read_csv(self.file)
        self.data['first_day_of_month'] = pd.to_datetime(self.data['first_day_of_month'])

    def __len__(self):
        return len(self.data) // self.stride

    def __getitem__(self, item):
        """
        Retrieving seq_len data
        1. The county (CFIPS) should be the same
        2. And the difference between the date(first_day_of_month) should be at most 3 months
        """
        i = item * self.stride
        county = self.data.iloc[i]['cfips']

        rows_data=self.data.iloc[i:i+self.seq_len]

        #Check if the county is the same
        is_valid = len(rows_data)==self.seq_len and (rows_data['cfips'].unique()[0]==county) and (rows_data['first_day_of_month'].diff().max()<pd.Timedelta(days=90))

        if not is_valid:
            ##Find a random item that is valid
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())

        #Taking seq_len rows and considering the following features
        #pct_bb,pct_college,pct_foreign_born,pct_it_workers,median_hh_inc, active,microbusiness_density
        features_tensor = torch.tensor(
            rows_data[['pct_bb', 'pct_college', 'pct_foreign_born', 'pct_it_workers', 'median_hh_inc','year', 'active',
                        'microbusiness_density']].values, dtype=torch.float32)

        #return the iterator
        return features_tensor




