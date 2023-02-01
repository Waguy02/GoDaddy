import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from constants import DATA_DIR, N_CENSUS_FEATURES, USE_CENSUS, TEST_FILE, CENSUS_FILE, MEAN_MB, STD_MB
from my_utils import DatasetType, extract_census_features, get_cfips_index

EVAL_START_DATE = "2022-05-01"
TEST_START_DATE =  "2022-11-01"


class LstmDataset(Dataset):
    def __init__(self, type, seq_len, stride=1,use_census=USE_CENSUS):
        self.type = type
        self.seq_len = seq_len
        self.stride = stride
        self.use_census = use_census
        self.load_data()
        self.prepare_sequences()


    def init_transforms(self):
        """
        Initialize transforms.Might be different for each dataset type
        """

    def load_data(self):
        """
        Load data from the data items if necessary
        """

        self.main_file = os.path.join(DATA_DIR, "train.csv")
        self.main_df = pd.read_csv(self.main_file)

        if self.type == DatasetType.TEST:
            self.test_df = pd.read_csv(TEST_FILE)
            self.test_df["microbusiness_density"] = [0 for _ in range(len(self.test_df))]
            self.test_df["county"] =["NAN" for _ in range(len(self.test_df))]
            self.test_df["state"] =["NAN" for _ in range(len(self.test_df))]

            self.main_df = pd.concat([self.main_df, self.test_df], ignore_index=True)

        self.main_df['year']=pd.to_datetime(self.main_df['first_day_of_month'].str.split("-", expand=True)[0])


        if self.use_census:
            #Merge the census features
            self.cfips_index=get_cfips_index()
            self.census_df = pd.read_csv(CENSUS_FILE)
            self.census_df["year"]=pd.to_datetime(self.census_df["year"],format="%Y")
            self.main_df=pd.merge(self.main_df,self.census_df,on=["cfips","year"],how="left")


        ##Group by cfips and sort by date
        self.main_df=self.main_df.sort_values(by=["cfips","first_day_of_month"])
        self.main_df["id"] =list(range(len(self.main_df)))



    def prepare_sequences(self):
        """
        Prepare the sequences for the LSTM:
        Build a list of (id(offset), id(seq_len+offset)) tuples
        """
        self.sequences=[]

        if self.type == DatasetType.TRAIN:
            ##Train data are dates before EVAL_START_DATE
            df=self.main_df[self.main_df['first_day_of_month']<EVAL_START_DATE]

            for i in tqdm(range(0, len(df)-self.seq_len, self.stride), desc="Preparing sequences of dataset of type train"):

                ##The cfips should be the same for the whole sequence(just check the first and last rows)
                if df.iloc[i]["cfips"] != df.iloc[i + self.seq_len - 1]["cfips"]:
                    continue

                if i + self.seq_len > len(df) :
                    break

                #Get the corresponding ids
                self.sequences.append((self.main_df.iloc[i]["id"], self.main_df.iloc[i + self.seq_len]["id"]))



        else :
            if self.type == DatasetType.VALID:
                df = self.main_df[self.main_df['first_day_of_month'] >= EVAL_START_DATE]

            else:
                df = self.main_df[self.main_df['first_day_of_month'] >= TEST_START_DATE]


            for i in tqdm(range(0, len(df),self.stride), desc="Preparing sequences of dataset of type {}".format("eval" if self.type == DatasetType.VALID else "test")):
                ## In eval and test sequences, the step to predict should always be the last one of the sequence

                ##Find the offest of the start in the main df


                offset=df.iloc[i]["id"]

                offset = offset - self.seq_len + 1

                ##check if the cfips is the same
                if self.main_df.iloc[offset]["cfips"] != self.main_df.iloc[offset + self.seq_len - 1]["cfips"]:
                    #Warning
                    print("Warning: cfips is not the same for the whole sequence . Offsets :",offset,offset + self.seq_len - 1)

                self.sequences.append((offset, offset + self.seq_len))

        ##Update the type of the year column to int
        self.main_df["year"]=self.main_df["year"].dt.year



    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        """
        Retrieving seq_len data
        1. The county (CFIPS) should be the same
        2. And the difference between the date(first_day_of_month) should be at most 3 months
        """
        start,end=self.sequences[item]
        rows_data=self.main_df.iloc[start:end]



        tensor = torch.tensor(rows_data[['microbusiness_density']].values,
                                       dtype=torch.float32)  # Not considering the census features

        tensor = (tensor - MEAN_MB)/STD_MB #Normalize the microbusiness density

        if self.use_census:
            censur_features_tensor = extract_census_features(rows_data, cfips_index=self.cfips_index,single_row=False)
            tensor = torch.cat((censur_features_tensor,tensor), dim=1)

        return tensor




