import json
import os
from enum import Enum
from itertools import islice
import numpy as np
import pandas as pd
import torch

from constants import DATA_DIR, N_COUNTY


def read_json(path_json):
    with open(path_json, encoding='utf8') as json_file:
        return json.load(json_file)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
def chunks(data, SIZE):
    """Split a dictionnary into parts of max_size =SIZE"""
    it = iter(data)
    for _ in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}

def sorted_dict(x, ascending=True):
    """
    Sort dict according to value.
    x must be a primitive type: int,float, str...
    @param x:
    @return:
    """
    return dict(sorted(x.items(), key=lambda item: (1 if ascending else -1) * item[1]))
def reverse_dict(input_dict):
    """
    Reverse a dictonary
    Args:
        input_dict:

    Returns:

    """
    inv_dict = {}
    for k, v in input_dict.items():
        inv_dict[v] = inv_dict.get(v, []) + [k]

    return inv_dict

def save_matrix(matrix,filename):
    with open(filename,'wb') as output:
        np.save(output,matrix)
def load_matrix(filename,auto_delete=False):
    with open(filename,'rb') as input:
        matrix=np.load(input)

    if auto_delete:
        os.remove(filename)
    return matrix



class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0



from enum import Enum
class DatasetType(Enum):
    TRAIN="train"
    VALID="valid"
    TEST="test"



def extract_census_features(row,cfips_index):
    """

    @param row: Row of the dataframe
    @param cfips_index: index of the cfips for one-hot encoding
    @return:
    """


    features_tensor = torch.tensor( [row['pct_bb'],
                                    row['pct_college'],
                                    row['pct_foreign_born'],
                                    row['pct_it_workers'],
                                    row['median_hh_inc'],
                                    row['year']], dtype=torch.float32)
    #Min-max normalization
    features_tensor[0] = (features_tensor[0]- 24.5/ (97.6-24.5))
    features_tensor[1] = (features_tensor[1] /48)
    features_tensor[2] = (features_tensor[2]/ 54)
    features_tensor[3] = (features_tensor[3] / 17.4)
    features_tensor[4] = (features_tensor[4]- 17109/(1586821-17109))
    features_tensor[5] = (features_tensor[5] / 2023)

    cfips_one_hot = get_cfips_one_hot(row['cfips'], cfips_index)

    ##Add one-hot encoding of cfips
    features_tensor = torch.cat((cfips_one_hot,features_tensor), 0)

    return features_tensor






def get_cfips_index():
    """
    Return a dictionary with key=cfips and value=index for using a one-hot encoding
    """
    df= pd.read_csv(os.path.join(DATA_DIR, "census_ae.csv"))
    cfips = df['cfips'].unique()
    cfips.sort()
    #Sort cfips
    return {cfips[i]: i for i in range(len(cfips))}


def get_cfips_one_hot(cfips,cfips_index):
    """
    Return a one-hot  encoding of the cfips_index using torch
    """
    assert len(cfips_index)==N_COUNTY,"The length of cfips_index must be equal to N_COUNTY"
    one_hot=torch.zeros(N_COUNTY)
    one_hot[cfips_index[cfips]]=1
    return one_hot
