import json
import logging
import os

import numpy as np
import torch
from torch import nn
from constants import DEVICE, USE_CENSUS, FEATURES_AE_LATENT_DIM, FEATURES_AE_CENSUS_DIR, N_CENSUS_FEATURES, \
    N_DIMS_COUNTY_ENCODING
from networks.features_autoencoder import FeaturesAENetwork


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        # Sin/Cos positional encoding
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1).squeeze(1)
        # COnvert to nn.Parameter
        self.pe = nn.Parameter(self.pe, requires_grad=False)

    def forward(self, x):
        # Add positional encoding to the input (Pay attention to the dimensions (the pe does not have the batch dimension))
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerPredictor(nn.Module):

    def __init__(self,
                 emb_dim=32,
                 n_layers=3,
                 n_head=8,
                 max_seq_len=100,
                 dim_feedforward=128,
                 use_derivative=True,
                 use_census=USE_CENSUS,
                 n_dims_census_emb=2,
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

        super(TransformerPredictor, self).__init__()
        self.variante_num = 4
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.use_census = use_census
        self.max_seq_len = max_seq_len
        self.census_features_encoder = None
        self.n_dims_census_emb = n_dims_census_emb
        self.input_dim = 1
        self.use_derivative = use_derivative
        if self.use_derivative:
            self.input_dim += 2  # 2 for derivative

        if self.use_census:
            self.input_dim = self.input_dim + self.n_dims_census_emb

        self.experiment_dir = experiment_dir
        self.model_name = os.path.basename(self.experiment_dir)
        self.reset = reset
        self.load_best = load_best
        self.setup_dirs()
        self.setup_network()

        if not reset: self.load_state()

    ##1. Defining network architecture
    def setup_network(self):
        """
        Initialize the network  architecture here
        @return:
        """
        # Input encoder from self.input_dim to self.emb_dim along with positional encoding
        if self.use_census:
            self.query_encoder = nn.Sequential(nn.Linear(N_DIMS_COUNTY_ENCODING + N_CENSUS_FEATURES, self.emb_dim))
            self.census_features_encoder = nn.Sequential(
                nn.Linear(N_CENSUS_FEATURES, self.n_dims_census_emb),
            )

        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.emb_dim),
        )

        ##Positional encoding
        self.positional_encoding = PositionalEncoding(self.emb_dim, max_len=self.max_seq_len)
        self.dropout = nn.Dropout(p=0.1)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=self.n_head, dim_feedforward=self.dim_feedforward,
                                       dropout=0,
                                       batch_first=True),
            num_layers=self.n_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.emb_dim, nhead=self.n_head, dim_feedforward=self.dim_feedforward,
                                       dropout=0,
                                       batch_first=True),
            num_layers=self.n_layers,

        )

        if self.use_census:
            self.regressor = nn.Sequential(
                nn.Linear(2 * self.emb_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1)
            )

        else:
            self.regressor = nn.Sequential(
                nn.Linear(self.emb_dim, 1)
            )

    ##2. Model Saving/Loading
    def load_state(self, best=False):
        """
        Load model
        :param self:
        :return:
        """
        if best and os.path.exists(self.save_best_file):
            logging.info(f"Loading best model state : {self.save_file}")
            self.load_state_dict(torch.load(self.save_file, map_location=DEVICE))
            return

        if os.path.exists(self.save_file):
            logging.info(f"Loading model state : {self.save_file}")
            self.load_state_dict(torch.load(self.save_file, map_location=DEVICE))

    def save_state(self, best=False):
        if best:
            logging.info("Saving best model")
            torch.save(self.state_dict(), self.save_best_file)
        torch.save(self.state_dict(), self.save_file)

    ##3. Setupping directories for weights /logs ... etc
    def setup_dirs(self):
        """
        Checking and creating directories for weights storage
        @return:
        """
        self.save_file = os.path.join(self.experiment_dir, f"{self.model_name}.pt")
        self.save_best_file = os.path.join(self.experiment_dir, f"{self.model_name}_best.pt")
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    # 4. Forward call
    def forward(self, X_input):
        """
        +Forward call here.
        It a time series, so we need the full sequence output (strided by 1)
        @param X:
        @return:
        """
        # 0. Preparing the input (Removing the target from the input)
        X = X_input[:, :-1, :]  # Removing the target from the input (Only required when using census features)

        if self.use_derivative:
            d_left = torch.zeros((X.shape[0], X.shape[1], 1), device=DEVICE)
            d_left[:, 1:, -1] = X[:, 1:, -1] - X[:, :-1, -1]

            d_right = torch.zeros((X.shape[0], X.shape[1], 1), device=DEVICE)
            d_right[:, :-1, -1] = X[:, 1:, -1] - X[:, :-1, -1]

            X = torch.cat((d_left, X, d_right), dim=-1)  ## Adding the derivative to the input as a new feature

        if self.use_census:
            target = X[:, -1, :]  # Last element of the sequence is the target .
            query = self.query_encoder(target[:, :N_DIMS_COUNTY_ENCODING + N_CENSUS_FEATURES])

            enc_census = self.census_features_encoder(
                X[:, :, N_DIMS_COUNTY_ENCODING:N_CENSUS_FEATURES + N_DIMS_COUNTY_ENCODING])
            X = torch.cat((X[:, :, N_CENSUS_FEATURES + N_DIMS_COUNTY_ENCODING:], enc_census), dim=-1)

        # 2. Apply the input encoder to the input
        X = self.input_embedding(X)

        # 3. Add the positional encoding
        X = self.positional_encoding(X)

        # 4. Add a query token to the input. Encoding of the cfips. (It is the same for all the sequence)
        if self.use_census:
            X = torch.cat((query.unsqueeze(1), X), dim=1)

        # 4. Apply the transformer encoder to get the memory
        X = self.transformer_encoder(X)

        if self.use_census:
            query_enc = X[:, 0, :]
            X = X[:, 1:, :]  # Removing the query token

        # .5 Apply the transformer decoder to get the next item in the sequence
        tgt_sequence = torch.zeros(X.shape[0], 1, X.shape[-1]).to(DEVICE)
        tgt_mask = torch.ones(1, 1).to(DEVICE)

        # 6. Then apply the transformer to get the next item in the sequence
        output = self.transformer_decoder(tgt_sequence, memory=X,
                                          tgt_mask=tgt_mask)  # We want to predict the next item in the sequence

        # 7.We only want the last output of the sequence
        output = output[:, -1, :]
        if self.use_census:
            output = torch.cat((output, query_enc), dim=-1)

        # 3. Finally apply the regressor to get the predictions.
        output = self.regressor(output)

        return output





