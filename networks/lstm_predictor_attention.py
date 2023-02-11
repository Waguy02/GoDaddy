import logging
import os
import torch
import torchvision.models
from torch import nn
from constants import ROOT_DIR, DEVICE, AE_LATENT_DIM, LSTM_HIDDEN_DIM, N_CENSUS_FEATURES, USE_CENSUS, EXPERIMENTS_DIR, \
    FEATURES_AE_CENSUS_DIR
from networks.features_autoencoder import FeaturesAENetwork
from networks.lstm_predictor import LstmPredictor
from networks.lstm_predictor2 import LstmPredictor2


class LstmPredictorWithAttention(LstmPredictor2):
    def __init__(self, hidden_dim=LSTM_HIDDEN_DIM,
                 n_hidden_layers=1,
                 use_encoder=USE_CENSUS,
                 use_derivative=False,
                 experiment_dir="my_model", reset=False, load_best=True):
        self.use_derivative=use_derivative
        super(LstmPredictorWithAttention, self).__init__(hidden_dim=hidden_dim,
                                                         n_hidden_layers=n_hidden_layers,
                                                         use_encoder=use_encoder,
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

        if self.use_census_encoder:
            # Freeze the encoder weights
            for param in self.features_encoder.parameters():
                param.requires_grad = False
            pass

        self.regressor_dim = self.hidden_dim + (0 if not self.use_census_encoder else self.features_encoder.hidden_dim)

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.n_hidden_layers,
                            batch_first=True)

        self.regressor = nn.Sequential(
            nn.Linear(self.regressor_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1))
        self.attention=nn.Linear(self.hidden_dim,1)

    def forward(self, input):

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





        # Concatenate the context vector with the encoded features (if applicable)
        if self.use_census_encoder:
            encoded_features = self.features_encoder.encode(input[:, :, :self.features_encoder.input_dim])
            context_vector = torch.cat((encoded_features, weighted_output), dim=2)
        else:
            context_vector = weighted_output

        context_vector=context_vector.sum(dim=1)

        # Apply the regressor to get the predictions
        output = self.regressor(context_vector)

        # Unsqueeze the output to have the same shape as the input(batch_size,seq_len,1)

        return output