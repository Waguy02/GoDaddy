import logging
import os
import torch
from torch import nn
from constants import DEVICE, N_CENSUS_FEATURES, AE_LATENT_DIM, N_COUNTY


class FeaturesAENetwork(nn.Module):
    """
    The features are
    pct_bb,pct_college,pct_foreign_born,pct_it_workers,median_hh_income and the date
    Autoencoder network for features representation of the input data
    """

    def __init__(self, experiment_dir="my_model", reset=False, load_best=True, input_dim=N_CENSUS_FEATURES + N_COUNTY,
                 hidden_dim=AE_LATENT_DIM):
        super(FeaturesAENetwork, self).__init__()
        self.experiment_dir = experiment_dir
        self.model_name = os.path.basename(self.experiment_dir)
        self.reset = reset
        self.load_best = load_best
        self.setup_dirs()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.setup_network()
        if not reset: self.load_state()

    ##1. Defining network architecture
    def setup_network(self):
        """
        Initialize the network  architecture here
        @return:
        """

        # 1. Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_dim, ))

        # 2. Decoder
        self.decoder_bone = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.Sigmoid(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 256),
            nn.Sigmoid(),
        )

        # Decoder for census one-hot encoded features
        self.decoder_cfips = nn.Sequential(
            nn.Linear(256, N_COUNTY),
            nn.Softmax(dim=1),
        )

        ## Decoder for census continuous features
        self.decoder_census = nn.Sequential(
            nn.Linear(256, N_CENSUS_FEATURES))

    ##2. Model Saving/Loading
    def load_state(self, best=False):
        """
        Load model
        :param self:
        :return:
        """
        if best and os.path.exists(self.save_best_file):
            logging.info(f"Loading features encoder : {self.save_best_file}")
            self.load_state_dict(torch.load(self.save_file, map_location=DEVICE))
            return

        if os.path.exists(self.save_file):
            logging.info(f"Loading features encoder : {self.save_file}")
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
    def forward(self, input):
        """
        Forward call here during training.
        Return the reconstructed input
        """
        hidden_state = self.encoder(input)
        output = self.decoder_bone(hidden_state)
        output_census = self.decoder_census(output)
        output_cfips = self.decoder_cfips(output)
        return hidden_state,  output_cfips, output_census


    # 5. Inference call (Just encoding)
    def encode(self, input):
        """
        Forward call here during inference.
        Return the hidden state
        """
        hidden_state = self.encoder(input)
        return hidden_state
