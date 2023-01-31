import os
import torch

ROOT_DIR=os.path.dirname(os.path.realpath(__file__))
DATA_DIR=os.path.join(ROOT_DIR,"data","godaddy-microbusiness-density-forecasting") ##Directory of dataset

EXPERIMENTS_DIR=os.path.join(ROOT_DIR, "logs/experiments")
use_cuda = torch .cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")


N_CENSUS_FEATURES= 6 #pct_bb,pct_college,pct_foreign_born,pct_it_workers,median_hh_inc, microbusiness_density
#cfips is not considered as a feature we use a one-hot encoding for it


USE_CENSUS= False #Without census features

AE_LATENT_DIM= 32

LSTM_HIDDEN_DIM = 8

SEQ_LEN=6
SEQ_STRIDE= 1

N_COUNTY=3142

FEATURES_AE_CENSUS_DIR=os.path.join(EXPERIMENTS_DIR, "base_features_ae")