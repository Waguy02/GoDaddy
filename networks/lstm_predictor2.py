import logging
import os
import torch
import torchvision.models
from torch import nn
from constants import ROOT_DIR, DEVICE, AE_LATENT_DIM, LSTM_HIDDEN_DIM, N_CENSUS_FEATURES, USE_CENSUS, EXPERIMENTS_DIR, \
    FEATURES_AE_CENSUS_DIR
from networks.features_autoencoder import FeaturesAENetwork
from networks.lstm_predictor import LstmPredictor


class LstmPredictor2(LstmPredictor):
    """
    Lstm Predictor that do not directly incorporate census but only on the regression part
    """
    def __init__(self, hidden_dim=LSTM_HIDDEN_DIM,
                 n_hidden_layers=1,
                 use_encoder=USE_CENSUS,
                 experiment_dir="my_model", reset=False, load_best=True):
        """
        @param features_encoder :
        @param input_dim:
        @param hidden_dim:
        @param ues_encoder:²
        @param experiment_dir:
        @param reset:
        @param load_best:
        """

        super(LstmPredictor2, self).__init__(hidden_dim=hidden_dim,
                                             n_hidden_layers=n_hidden_layers,
                                             use_encoder=use_encoder,
                                             experiment_dir=experiment_dir,
                                             reset=reset,
                                             load_best=load_best)

        self.variante_num=1

    ##1. Defining network architecture
    def setup_network(self):
        """
        Initialize the network  architecture here
        @return:
        """
        self.input_dim = 1

        if self.use_census_encoder:
            # Freeze the encoder weights
            # for param in self.features_encoder.parameters():
            #     param.requires_grad = False
            pass


        self.regressor_dim = self.hidden_dim + (0 if not self.use_census_encoder else self.features_encoder.hidden_dim)

        self.lstm=nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,num_layers=self.n_hidden_layers,batch_first=True)

        self.regressor=nn.Sequential(
            nn.Linear(self.regressor_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1))







    #4. Forward call
    def forward(self, input):
        """
        Forward call here.
        It a time series, so we need the full sequence output (strided by 1)
        @param input:
        @return:
        """


        #1. First apply the encoder to the first N_CENSUS8FEAUTRES features of each element in the sequence
        if self.use_census_encoder:
            encoded_features = self.features_encoder.encode(input[:, :, :self.features_encoder.input_dim])
            ##The encoded features do not go through the LSTM, so we need to concatenate them with the rest of the input
            lstm_input= input[:, :, self.features_encoder.input_dim:]
        else:
            lstm_input = input

        #2. Then apply the LSTM
        lstm_output, _ = self.lstm(lstm_input)

        #3. Concatenate the encoded features with the LSTM output
        if self.use_census_encoder:
            output = torch.cat((encoded_features, lstm_output), dim=2)
        else :
            output = lstm_output

        #3. Finally apply the regressor to get the predictions.
        output = self.regressor(output)

        return output





