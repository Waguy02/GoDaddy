import logging
import os
import torch
import torchvision.models
from torch import nn
from constants import ROOT_DIR, DEVICE, AE_LATENT_DIM, LSTM_HIDDEN_DIM, N_CENSUS_FEATURES, USE_CENSUS, EXPERIMENTS_DIR, \
    FEATURES_AE_CENSUS_DIR, N_DIMS_COUNTY_ENCODING
from networks.features_autoencoder import FeaturesAENetwork
from networks.lstm_predictor import LstmPredictor
from networks.lstm_predictor2 import LstmPredictor2


class LstmPredictorWithAttention(LstmPredictor2):
    def __init__(self, hidden_dim=LSTM_HIDDEN_DIM,
                 n_hidden_layers=1,
                 use_census=USE_CENSUS,
                 use_derivative=False,
                 experiment_dir="my_model", reset=False, load_best=True):
        self.use_derivative=use_derivative
        super(LstmPredictorWithAttention, self).__init__(hidden_dim=hidden_dim,
                                                         n_hidden_layers=n_hidden_layers,
                                                         use_census=use_census,
                                                         experiment_dir=experiment_dir,
                                                         reset=reset,
                                                         load_best=load_best)
        self. variante_num=2


    def setup_network(self):
        """
                Initialize the network  architecture here
                @return:
                """
        self.input_dim = 1

        if self.use_derivative:
            self.input_dim += 2# 2 for the derivative at left and right

        if self.use_census:
            # Freeze the encoder weights
            self.input_dim+=self.features_encoder.hidden_dim
            for param in self.features_encoder.parameters():
                param.requires_grad = False


        self.regressor_dim = self.hidden_dim + (0 if not self.use_census else self.features_encoder.hidden_dim)

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.n_hidden_layers,
                            batch_first=True)

        self.regressor = nn.Sequential(
            nn.Linear(self.regressor_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1))
        self.attention=nn.Linear(self.hidden_dim,1)

    def forward(self, input):

        #Remove the target (last token)
        query = input[:, -1, :]
        input = input[:, :-1, :]

        if self.use_census:
            encoded_features = self.features_encoder.encode(input[:,:,:self.features_encoder.input_dim])
            encoded_query = self.features_encoder.encode(query[:,  :self.features_encoder.input_dim])
            input = torch.cat((encoded_features, input[:, :, self.features_encoder.input_dim:]), dim=2)



        if self.use_derivative:
            d_left= torch.zeros((input.shape[0],input.shape[1],1), device=DEVICE)
            d_left[:,1:, -1] = input[:, 1:, -1] - input[:, :-1, -1]

            d_right= torch.zeros((input.shape[0],input.shape[1],1), device=DEVICE)
            d_right[:,:-1, -1] = input[:, 1:, -1] - input[:, :-1, -1]

            input = torch.cat((d_left, input, d_right), dim=-1) ## Adding the derivative to the input as a new feature


        lstm_output, _ = self.lstm(input[:, :, -self.input_dim:])

        # Apply attention to the LSTM output
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_output = attention_weights * lstm_output



        context_vector = weighted_output

        context_vector=context_vector.sum(dim=1)
        if self.use_census:
            context_vector = torch.cat((context_vector, encoded_query), dim=1)

        # Apply the regressor to get the predictions
        output = self.regressor(context_vector)


        return output