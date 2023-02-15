import math
import os

import numpy as np
import torch

ROOT_DIR=os.path.dirname(os.path.realpath(__file__))
DATA_DIR=os.path.join(ROOT_DIR,"data","godaddy-microbusiness-density-forecasting") ##Directory of dataset

EXPERIMENTS_DIR=os.path.join(ROOT_DIR, "logs/experiments")
use_cuda = torch .cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")


N_CENSUS_FEATURES= 6 #pct_bb,pct_college,pct_foreign_born,pct_it_workers,median_hh_inc,active
#cfips is not considered as a feature we use a one-hot encoding for it


USE_CENSUS= False #Without census features

AE_LATENT_DIM= 32

LSTM_HIDDEN_DIM = 8

SEQ_LEN=6
SEQ_STRIDE= 1

N_COUNTY=3142
N_DIMS_COUNTY_ENCODING=  math.ceil(math.log(N_COUNTY,2))

FEATURES_AE_CENSUS_DIR=os.path.join(EXPERIMENTS_DIR, "features_ae_2_dims")
FEATURES_AE_LATENT_DIM= 2

TRAIN_FILE= os.path.join(DATA_DIR, "train.csv")
TEST_FILE= os.path.join(DATA_DIR, "test.csv")

CENSUS_FILE =os.path.join(DATA_DIR, "census_interpolated.csv")

NB_FUTURES= 10 #Number of days to predict


#Scaling factors for microbusiness density
MEAN_MB= 3.817671
STD_MB= 4.991087

MAX_MB= 300
MIN_MB= 0.0