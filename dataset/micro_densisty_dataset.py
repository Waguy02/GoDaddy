import logging
import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from constants import DATA_DIR, N_CENSUS_FEATURES, USE_CENSUS, TEST_FILE, CENSUS_FILE, MEAN_MB, STD_MB, CENSUS_FEATURES
from my_utils import DatasetType, extract_census_features, get_cfips_index
random.seed(42)
EVAL_START_DATE = "2022-10-01"
TEST_START_DATE =  "2022-11-01"

SEED=42
class MicroDensityDataset(Dataset):
    def __init__(self, type, seq_len, stride=1,use_census=USE_CENSUS):
        self.type = type
        self.seq_len = seq_len
        self.stride = stride if type == DatasetType.TRAIN else 1
        self.use_census = use_census
        self.load_data()
        self.tensor_list = dict()
        self.prepare_sequences()
        if type!=DatasetType.TEST:
            self.prepare_tensors()

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

            self.test_df =self.test_df.sort_values(by=["cfips","first_day_of_month"])
            self.test_df = self.test_df.reset_index(drop=True)

            # Fill The missing value of active with last known value (for each cfips)
            self.main_df["active"] = self.main_df.groupby("cfips")["active"].apply(lambda x: x.fillna(method="ffill"))
            pass

        if self.use_census:
            #Merge the census features
            self.cfips_index=get_cfips_index()
            self.census_df=pd.read_csv(CENSUS_FILE)

            #Create a dictionary of the census data for each cfips
            self.census_dict=dict()
            for i in range(len(self.census_df)):
                cfips=self.census_df.iloc[i]["cfips"]
                self.census_dict[cfips]=self.census_df.iloc[i]





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
                self.sequences.append((df.iloc[i]["id"], df.iloc[i]["id"]+ self.seq_len))

                ##DEBUG


        else :


            if self.type == DatasetType.VALID:
                df = self.main_df[self.main_df['first_day_of_month'] >= EVAL_START_DATE]

            else:
                df = self.main_df[self.main_df['first_day_of_month'] >= TEST_START_DATE]


            for i in tqdm(range(0, len(df),self.stride), desc="Preparing sequences of dataset of type {}".format("eval" if self.type == DatasetType.VALID else "test")):
                ## In eval and test sequences, the step to predict should always be the last one of the sequence

                ##Find the offest of the start in the main df


                offset=df.iloc[i]["id"]

                offset = offset - self.seq_len+1  # The step to predict is the last one of the sequence


                ##check if the cfips is the same
                if self.main_df.iloc[offset]["cfips"] != self.main_df.iloc[offset + self.seq_len - 1]["cfips"]:
                    #Warning
                    print("Warning: cfips is not the same for the whole sequence . Offsets :",offset,offset + self.seq_len - 1)

                self.sequences.append((offset, offset + self.seq_len))


    def prepare_tensors(self):

        for i in tqdm(range(len(self.sequences)), desc="Prefetching tensors..."):
            start,end=self.sequences[i]
            self.tensor_list[(start,end)]=self.__getitem__(i)



    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        """
        Retrieving seq_len data
        1. The county (CFIPS) should be the same
        2. And the difference between the date(first_day_of_month) should be at most 3 months
        """
        start,end=self.sequences[item]

        if (start,end) in self.tensor_list:
            return self.tensor_list[(start,end)]

        rows_data=self.main_df.iloc[start:end]


        #ensure unique cfips
        # assert len(rows_data["cfips"].unique())==1

        tensor = torch.tensor(rows_data[['microbusiness_density']].values,
                                       dtype=torch.float32)  # Not considering the census features

        #FEatures scaling


        if self.use_census:
            census_data=self.census_dict[rows_data.iloc[0]["cfips"]]
            #Get the last value of active
            active=int(rows_data.iloc[-1]["active"])
            census_tensor=extract_census_features(census_data, active, self.cfips_index)

            return {"density":tensor,"census":census_tensor}

        return {"density":tensor}


    def mix_with(self, other_dataset, size=0.8):
        """
        Combine two datasets exemple a train dataset and test dataset
        @param other_dataset:
        @param size:
        @return:
        """

        all_sequences= self.sequences + other_dataset.sequences
        random.shuffle(all_sequences)
        self.sequences=all_sequences[:int(len(all_sequences)*size)]

        all_tensor_list = self.tensor_list.copy().update(other_dataset.tensor_list)
        self.tensor_list={k:all_tensor_list[k] for k in self.sequences}

        other_dataset.sequences=all_sequences[int(len(all_sequences)*size):]
        other_dataset.tensor_list={k:all_tensor_list[k] for k in other_dataset.sequences}
        logging.info("Combined dataset: {} sequences for train and {} sequences for test".format(len(self.sequences),len(other_dataset.sequences)))


