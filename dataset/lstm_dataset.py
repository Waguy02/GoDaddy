import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from constants import DATA_DIR, N_CENSUS_FEATURES, USE_CENSUS
from my_utils import DatasetType

TRAIN_END_DATE = ""


class LstmDataset(Dataset):
    def __init__(self, type, seq_len, stride=1,use_census=USE_CENSUS):
        self.type = type
        self.seq_len = seq_len
        self.stride = stride
        self.use_census = use_census
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
        rows_data=self.data.iloc[i:i+self.seq_len]

        #Check if the county is the same
        max_time_diff=rows_data['first_day_of_month'].diff().max()
        min_time_diff = rows_data['first_day_of_month'].diff().min()
        is_valid = len(rows_data)==self.seq_len and (len(rows_data['cfips'].unique())==1) and (max_time_diff<pd.Timedelta(days=90)) and (min_time_diff>pd.Timedelta(days=0))

        if not is_valid:
            ##Find a random item that is valid
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())

        #Taking seq_len rows and considering the following features
        #pct_bb,pct_college,pct_foreign_born,pct_it_workers,median_hh_inc, active,microbusiness_density

        if self.use_census:
            features_tensor = torch.tensor(
                rows_data[['cfips','pct_bb', 'pct_college', 'pct_foreign_born', 'pct_it_workers', 'median_hh_inc','year',
                            'microbusiness_density']].values, dtype=torch.float32)

            #Normalize the cfips and the year
            features_tensor[:,0]=features_tensor[:,0]/1000
            features_tensor[:,6]=features_tensor[:,6]/1000



            # noramlize the microbusiness_density max=3
            features_tensor[:, 0] = features_tensor[:, 0] / 3
        else :
            features_tensor = torch.tensor(
                rows_data[['microbusiness_density']].values, dtype=torch.float32)  # Not considering the census features

            # noramlize the microbusiness_density max=3
            features_tensor[:,0]=features_tensor[:,0]/3
        #return the iterator
        return features_tensor




