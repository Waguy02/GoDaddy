import json
import os
from enum import Enum
from itertools import islice
import numpy as np
import pandas as pd
import torch

from constants import DATA_DIR, N_COUNTY, N_DIMS_COUNTY_ENCODING, CENSUS_FEATURES, CENSUS_YEARS, \
    CENSUS_FEATURES_MIN_MAX, QUERY_CENSUS_DIMS


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

def extract_census_features(row,  cfips_index):
    """
    @param row: Row of the dataframe
    @param cfips_index: Index of cfips
    @param active: The number of active cases for the county
    @return:
    """
    ##If series :
    #N_YEARS*N_FEATURES+ N_DIMS_COUNTY_ENCODING+1
    features_tensor= torch.zeros(QUERY_CENSUS_DIMS)

    idx=0
    for feature in CENSUS_FEATURES:
        start=idx
        features_values=[]
        for year in CENSUS_YEARS:
            min,max=CENSUS_FEATURES_MIN_MAX[feature]
            features_values.append((row[f"{feature}_{year}"]-min)/(max-min))
            idx+=1

        #Fill nan by interpolation
        features_values=np.array(features_values)
        nan_idx=np.where(np.isnan(features_values))[0]
        if len(nan_idx)>0:
            #Interpolate
            features_values[nan_idx]=np.interp(nan_idx,np.where(~np.isnan(features_values))[0], features_values[~np.isnan(features_values)])
        features_tensor[start:idx]=torch.tensor(features_values,dtype=torch.float32)





    #Add cfips encoding
    cfips= row['cfips']
    cfips_one_hot= get_cfips_encoding(cfips,cfips_index)

    features_tensor[idx :idx + N_DIMS_COUNTY_ENCODING] = cfips_one_hot

    # #Add active
    # min_active,max_active=CENSUS_FEATURES_MIN_MAX['active']
    # features_tensor[-1]=(active- min_active)/(max_active-min_active)


    return features_tensor.float()
def get_cfips_index():
    """
    Return a dictionary with key=cfips and value=index for using a one-hot encoding
    """
    df= pd.read_csv(os.path.join(DATA_DIR, "census_interpolated.csv"))
    cfips = df['cfips'].unique()
    cfips.sort()
    #Sort cfips
    return {cfips[i]: i for i in range(len(cfips))}
def get_cfips_encoding(cfips,cfips_index):
    """
     return the base 2 encoding of cfips
    """

    #n_dims is the number of bits needed to represent the cfips

    bin_index=np.binary_repr(cfips_index[cfips],width=N_DIMS_COUNTY_ENCODING)
    enc = torch.tensor([int(x) for x in bin_index],dtype=torch.float32)
    return enc



