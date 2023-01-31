import argparse
import logging
import os

import torch.utils.data

from constants import EXPERIMENTS_DIR, SEQ_LEN, SEQ_STRIDE, DEVICE
from losses.smape import SmapeCriterion
from my_utils import DatasetType
from dataset.lstm_dataset import LstmDataset
from logger import setup_logger
from networks.lstm_predictor import LstmPredictor
from networks.lstm_predictor_2 import LstmPredictor2
from training.trainer_lstm import TrainerLstmPredictor


def cli():
    """
   Parsing args
   @return:
   """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reset", "-r", action='store_true', default=False   , help="Start retraining the model from scratch")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.05, help="Learning rate of Adam optimized")
    parser.add_argument("--nb_epochs", "-e", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--model_name", "-n",help="Name of the model. If not specified, it will be automatically generated")
    parser.add_argument("--num_workers", "-w", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--batch_size", "-bs", type=int, default=512, help="Batch size for training")
    parser.add_argument("--log_level", "-l", type=str, default="INFO")
    parser.add_argument("--autorun_tb","-tb",default=False,action='store_true',help="Autorun tensorboard")
    parser.add_argument("--use_census","-c",default=False,action='store_true',help="Use census data")
    parser.add_argument("--seq_len", "-sl", type=int, default=SEQ_LEN, help="Sequence length")
    parser.add_argument("--seq_stride", "-ss", type=int, default=SEQ_STRIDE, help="Sequence stride")
    parser.add_argument("--hidden_dim", "-hd", type=int, default=4, help="Hidden dimension of the LSTM")
    parser.add_argument("--n_hidden_layers", "-nl", type=int, default=1, help="Number of hidden layers of the LSTM")
    parser.add_argument("--variante","-v",type=int,default=0,help="Variante of the model")
    return parser.parse_args()

def main(args):

    #Format the model name

    if args.model_name is None:
        model_name = f"lstm_hd.{args.hidden_dim}_nl.{args.n_hidden_layers}_sl.{args.seq_len}_ss.{args.seq_stride}_lr.{args.learning_rate}_bs.{args.batch_size}"
    else :
        model_name=args.model_name



    experiment_dir = os.path.join(EXPERIMENTS_DIR, model_name)


    NetworkClass = LstmPredictor if args.variante==0 else LstmPredictor2

    network = NetworkClass(experiment_dir=experiment_dir, hidden_dim=4, n_hidden_layers=1, use_encoder=args.use_census).to(DEVICE)

    #Adam optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    criterion= SmapeCriterion().to(DEVICE)


    logging.info("Training : "+model_name)
    trainer = TrainerLstmPredictor(network,
                      criterion,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      nb_epochs= args.nb_epochs,
                      batch_size=args.batch_size,
                      reset=args.reset,
                      )

    train_dataset=LstmDataset(type=DatasetType.TRAIN,seq_len=args.seq_len,stride=args.seq_stride,use_census=args.use_census)
    val_dataset=LstmDataset(type=DatasetType.VALID,seq_len=args.seq_len,stride=args.seq_stride,use_census=args.use_census)
    test_dataset=LstmDataset(type=DatasetType.TEST,seq_len=args.seq_len,stride=args.seq_stride,use_census=args.use_census)

    logging.info(f"Nb sequences : Train {len(train_dataset)} - Val {len(val_dataset)} - Test {len(test_dataset)}")

    train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True,drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,num_workers=0,drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,num_workers=0,drop_last=False,shuffle=False)

    ##Train
    trainer.fit(train_dataloader,val_dataloader)

    ##Load best model
    trainer.network.load_state(best=True)
    trainer.run_test(test_dataloader=test_dataloader)
    

if __name__ == "__main__":
    args = cli()
    setup_logger(args)
    main(args)
 