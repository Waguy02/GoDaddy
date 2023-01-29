import os
import torch

ROOT_DIR=os.path.dirname(os.path.realpath(__file__))
DATA_DIR=os.path.join(ROOT_DIR,"data","godaddy-microbusiness-density-forecasting") ##Directory of dataset

EXPERIMENTS_DIR=os.path.join(ROOT_DIR, "logs/experiments")
use_cuda = torch .cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")

# N_CENSUS_FEATURES= 6
N_CENSUS_FEATURES= 0  #Without census features

AE_LATENT_DIM= 16
LSTM_HIDDEN_DIM = 6

SEQ_LEN=6
SEQ_STRIDE= 1