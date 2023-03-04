import math
import os
import torch

ROOT_DIR=os.path.dirname(os.path.realpath(__file__))
DATA_DIR=os.path.join(ROOT_DIR,"data","godaddy-microbusiness-density-forecasting") ##Directory of dataset

EXPERIMENTS_DIR=os.path.join(ROOT_DIR, "logs/experiments")
use_cuda = torch .cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")


CENSUS_FEATURES = ['pct_bb', 'pct_college', 'pct_foreign_born', 'pct_it_workers', 'median_hh_inc']
N_CENSUS_FEATURES= len(CENSUS_FEATURES) #pct_bb,pct_college,pct_foreign_born,pct_it_workers,median_hh_inc,active
CENSUS_YEARS= [2019,2020,2021,2022,2023]
CENSUS_FEATURES_MIN_MAX= {
    'pct_bb': (24.5, 97.6),
    'pct_college': (0, 48),
    'pct_foreign_born': (0,54),
    'pct_it_workers': (0,17.4 ),
    'median_hh_inc': (17109,1586621),
    'active': (0,1167744)
}
MAX_NAN_PER_FEATURE=2 #Maximum number of nan values per census feature
#cfips is not considered as a feature we use a one-hot encoding for it


USE_CENSUS= False #Without census features
AE_LATENT_DIM= 32
LSTM_HIDDEN_DIM = 8
SEQ_LEN=6
SEQ_STRIDE= 1
N_COUNTY=3142
N_DIMS_COUNTY_ENCODING=  math.ceil(math.log(N_COUNTY,2))
QUERY_CENSUS_DIMS= len(CENSUS_FEATURES) * len(CENSUS_YEARS) + N_DIMS_COUNTY_ENCODING
FEATURES_AE_CENSUS_DIR=os.path.join(EXPERIMENTS_DIR, "features_ae_2_dims")
FEATURES_AE_LATENT_DIM= 2
TRAIN_FILE= os.path.join(DATA_DIR, "train.csv")
VALID_FILE= os.path.join(DATA_DIR, "revealed_test.csv")
TEST_FILE= os.path.join(DATA_DIR, "test.csv")
CENSUS_FILE =os.path.join(DATA_DIR, "census_starter_shifted.csv")
NB_FUTURES= 10 #Number of days to predict


#Scaling factors for microbusiness density
MEAN_MB= 3.817671
STD_MB= 4.991087
MAX_MB= 300
MIN_MB= 0.0