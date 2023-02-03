import argparse
import logging
import os

import torch.utils.data
from torch import nn
from torch.optim import Adam

from constants import EXPERIMENTS_DIR, DEVICE, N_CENSUS_FEATURES, AE_LATENT_DIM
from dataset.census_features_dataset import CensusDataset
from my_utils import DatasetType
from logger import setup_logger
from networks.features_autoencoder import FeaturesAENetwork
from networks.network import CustomNetwork
from training.trainer import Trainer
from training.trainer_features_ae import TrainerFeaturesAE


def cli():
    """
   Parsing args
   @return:
   """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reset", "-r", action='store_true', default=False   , help="Start retraining the model from scratch")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate of Adam optimized")
    parser.add_argument("--nb_epochs", "-e", type=int, default=40, help="Number of epochs for training")
    parser.add_argument("--model_name", "-n",help="Name of the model. If not specified, it will be automatically generated")
    parser.add_argument("--num_workers", "-w", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--batch_size", "-bs", type=int, default=64, help="Batch size for training")
    parser.add_argument("--log_level", "-l", type=str, default="INFO")
    parser.add_argument("--autorun_tb","-tb",default=False,action='store_true',help="Autorun tensorboard")
    parser.add_argument("--hidden_dim", "-hd", type=int, default=2, help="Hidden dimension of the autoencoder")
    return parser.parse_args()

def main(args):
    model_name = f"features_ae_{args.hidden_dim}_dims" if args.model_name is None else args.model_name
    experiment_dir = os.path.join(EXPERIMENTS_DIR, model_name)
    network=FeaturesAENetwork(
                load_best=False,
                reset=args.reset,
                experiment_dir=experiment_dir,
                hidden_dim=args.hidden_dim).to(DEVICE)

    optimizer = Adam(network.parameters(), lr=args.learning_rate)

    #LOSS function for cfip : hinge loss
    criterion_cfips = nn.CrossEntropyLoss()

    #LOSS function for features(MSE)
    criterion_features = nn.MSELoss()


    logging.info("Training : "+model_name)
    trainer = TrainerFeaturesAE(network,
                      criterion_cfips,
                      criterion_features,
                      optimizer=optimizer,
                      scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True),
                      nb_epochs= args.nb_epochs,
                      batch_size=args.batch_size,
                      reset=args.reset,
                      )


    train_dataset=CensusDataset(type=DatasetType.TRAIN)
    val_dataset=CensusDataset(type=DatasetType.VALID)

    train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,num_workers=args.num_workers)


    trainer.fit(train_dataloader,val_dataloader)
    
    

if __name__ == "__main__":
    args = cli()
    setup_logger(args)
    main(args)
 