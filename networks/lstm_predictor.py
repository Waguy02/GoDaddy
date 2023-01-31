import json
import logging
import os
import torch
import torchvision.models
from torch import nn
from constants import ROOT_DIR, DEVICE, LSTM_HIDDEN_DIM, N_CENSUS_FEATURES, USE_CENSUS, EXPERIMENTS_DIR, \
    FEATURES_AE_CENSUS_DIR
from networks.features_autoencoder import FeaturesAENetwork


class LstmPredictor(nn.Module):

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

        super(LstmPredictor, self).__init__()

        self.variante_num=0

        self.use_encoder = use_encoder
        if self.use_encoder:
            ##Get the hidden dimension of the encoder
            config_encoder=os.path.join(FEATURES_AE_CENSUS_DIR,"model.json")
            with open(config_encoder) as f:
                config = json.load(f)
                ae_hidden_dim = config["hidden_dim"]

            self.features_encoder = FeaturesAENetwork(experiment_dir=FEATURES_AE_CENSUS_DIR,hidden_dim=ae_hidden_dim,load_best=True).to(DEVICE)
            self.input_dim = self.features_encoder.hidden_dim + 1

            #Freeze the encoder weights
            for param in self.features_encoder.parameters():
                param.requires_grad = False

        else :
            self.features_encoder = None
            self.input_dim =1

        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
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
        self.lstm=nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,num_layers=self.n_hidden_layers,batch_first=True)

        self.regressor=nn.Sequential(
            nn.Linear(self.hidden_dim,1)
            )

        if self.use_encoder:
            # Freeze the encoder weights
            for param in self.features_encoder.parameters():
                param.requires_grad = False


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

    #4. Forward call
    def forward(self, input):
        """
        Forward call here.
        It a time series, so we need the full sequence output (strided by 1)
        @param input:
        @return:
        """
        #1. First apply the encoder to the first N_CENSUS8FEAUTRES features of each element in the sequence
        if self.use_encoder:
            encoded_features = self.features_encoder.encode(input[:, :, :self.features_encoder.input_dim])
            input = torch.cat((encoded_features, input[:, :, self.features_encoder.input_dim:]), dim=-1)

        #2. Then apply the LSTM
        output, _ = self.lstm(input)

        #3. Finally apply the regressor to get the predictions.
        output = self.regressor(output)

        return output





