import argparse
import logging
import os

import torch.utils.data
from torch.optim import Adam
from torchmetrics import SymmetricMeanAbsolutePercentageError

from constants import EXPERIMENTS_DIR, SEQ_LEN, SEQ_STRIDE, DEVICE
from dataset.dataset import DatasetType
from dataset.lstm_dataset import LstmDataset
from logger import setup_logger
from networks.features_autoencoder import FeaturesAENetwork

from networks.lstm_predictor import LstmPredictor
from networks.network import CustomNetwork
from training.trainer import Trainer
from training.trainer_lstm import TrainerLstmPredictor


def cli():
    """
   Parsing args
   @return:
   """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reset", "-r", action='store_true', default=False   , help="Start retraining the model from scratch")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.05, help="Learning rate of Adam optimized")
    parser.add_argument("--nb_epochs", "-e", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--model_name", "-n",help="Name of the model. If not specified, it will be automatically generated")
    parser.add_argument("--num_workers", "-w", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="Batch size for training")
    parser.add_argument("--log_level", "-l", type=str, default="INFO")
    parser.add_argument("--autorun_tb","-tb",default=False,action='store_true',help="Autorun tensorboard")
    return parser.parse_args()

def main(args):
    model_name = "base_lstm_ae" if args.model_name is None else args.model_name

    features_encoder = None
    # features_encoder = FeaturesAENetwork(experiment_dir=os.path.join(EXPERIMENTS_DIR, "base_features_ae")).to(DEVICE)

    experiment_dir = os.path.join(EXPERIMENTS_DIR, model_name)

    network=LstmPredictor(features_encoder=None, experiment_dir=experiment_dir).to(DEVICE)
    # network = LstmPredictor(features_encoder=features_encoder, experiment_dir=experiment_dir).to(DEVICE)

    optimizer = Adam(network.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    loss_fn= SymmetricMeanAbsolutePercentageError().to(DEVICE)
    criterion= lambda y_pred,y_true: loss_fn(y_pred,y_true)*100
    logging.info("Training : "+model_name)
    trainer = TrainerLstmPredictor(network,
                      criterion,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      nb_epochs= args.nb_epochs,
                      batch_size=args.batch_size,
                      reset=args.reset,
                      )

    train_dataset=LstmDataset(type=DatasetType.TRAIN,seq_len=SEQ_LEN,stride=SEQ_STRIDE)
    val_dataset=LstmDataset(type=DatasetType.VALID,seq_len=SEQ_LEN,stride=SEQ_STRIDE)

    train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True,drop_last=True,)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,num_workers=args.num_workers,drop_last=True)


    trainer.fit(train_dataloader,val_dataloader)
    
    

if __name__ == "__main__":
    args = cli()
    setup_logger(args)
    main(args)
 