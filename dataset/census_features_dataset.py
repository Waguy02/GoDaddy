import json
import os
from unicodedata import category
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from constants import DATA_DIR


from enum import Enum

from my_utils import DatasetType, extract_census_features, get_cfips_index


class CensusDataset(Dataset):
    def __init__(self, type):
        self.type=type
        self.load_data()
        self.cfips_index =get_cfips_index() ## For one-hot encoding of cfips
        pass

    def load_data(self):
        """
        Load data from the data items if necessary
        Returns:

        """
        self.data_file=os.path.join(DATA_DIR,f"census_interpolated_{'train' if self.type == DatasetType.TRAIN else 'val'}.csv")
        self.data = pd.read_csv(self.data_file)




    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        """
        cfips, pct_bb,pct_college,pct_foreign_born,pct_it_workers,median_hh_inc,year .
        Retrieve the following features from the dataset and return the corresponding tensor

        Returns:
        """
        row=self.data.iloc[idx]
        features_tensor =extract_census_features(row, self.cfips_index)
        return features_tensor




