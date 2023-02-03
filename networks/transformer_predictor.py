import logging
import os
import torch
from torch import nn
from constants import DEVICE, USE_CENSUS, FEATURES_AE_LATENT_DIM
from networks.features_autoencoder import FeaturesAENetwork


class TransformerPredictor(nn.Module):

    def __init__(self,
                 emb_dim=32,
                 n_layers=3,
                 n_head=8,
                 max_seq_len=100,
                 dim_feedforward=128,
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

        super(TransformerPredictor, self).__init__()
        self.variante_num=3
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.use_census_encoder = use_encoder
        self.max_seq_len = max_seq_len
        if self.use_census_encoder:
            #Get the hidden dimension of the encoder
            # config_encoder=os.path.join(FEATURES_AE_CENSUS_DIR,"model.json")
            # with open(config_encoder) as f:
            #     config = json.load(f)
            #     ae_hidden_dim = config["hidden_dim"]
            # self.features_encoder = FeaturesAENetwork(experiment_dir=FEATURES_AE_CENSUS_DIR,hidden_dim=ae_hidden_dim).to(DEVICE)
            self.census_features_encoder = FeaturesAENetwork(hidden_dim=FEATURES_AE_LATENT_DIM).to(DEVICE)
            self.input_dim = self.census_features_encoder.hidden_dim + 1
            # # Freeze the encoder weights
            # for param in self.features_encoder.parameters():
            #     param.requires_grad = False
        else :
            self.census_features_encoder = None
            self.input_dim =1


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
        #Input encoder from self.input_dim to self.emb_dim along with positional encoding
        self.input_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        )

        ##Positional encoding
        P = torch.zeros((1, self.max_seq_len, self.emb_dim))
        X = torch.arange(self.max_seq_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange( 0, self.emb_dim, 2, dtype=torch.float32) / self.emb_dim)
        P[0, :, 0::2] = torch.sin(X)
        P[0, :, 1::2] = torch.cos(X)
        self.positional_encoding = nn.Parameter(P, requires_grad=False)
        self.dropout = nn.Dropout(p=0.1)


        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.emb_dim, nhead=self.n_head, dim_feedforward=self.dim_feedforward,
                                       batch_first=True),
            num_layers=self.n_layers
        )

        self.regressor=nn.Sequential(
            nn.Linear(self.emb_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
            )

        if self.use_census_encoder:
            # Freeze the encoder weights/
            for param in self.census_features_encoder.parameters():
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
        if self.use_census_encoder:
            encoded_features = self.census_features_encoder.encode(input[:, :, :self.census_features_encoder.input_dim])
            input = torch.cat((encoded_features, input[:, :, self.census_features_encoder.input_dim:]), dim=-1)

        #2. Apply the input encoder to the input
        input = self.input_encoder(input)

        #3. Add the positional encoding
        input = input + self.positional_encoding[:, :input.shape[1], :]
        input = self.dropout(input)


        #.4 Setting mask to avoid looking at future values
        tgt_sequence = torch.zeros(input.shape[0],1,input.shape[-1]).to(DEVICE )
        tgt_mask = torch.ones(1,1).to(DEVICE)

        #3. Then apply the transformer to get the next item in the sequence
        output = self.transformer_decoder(tgt_sequence,memory=input,tgt_mask= tgt_mask)#We want to predict the next item in the sequence

        #3. Finally apply the regressor to get the predictions.
        output = self.regressor(output[:, -1, :])#We only want the last output of the sequence

        return output





