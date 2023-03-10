import argparse
import json
import logging
import os
import pickle

import torch.utils.data

from constants import EXPERIMENTS_DIR, SEQ_LEN, SEQ_STRIDE, DEVICE, ROOT_DIR
from losses.smape import SmapeCriterion
from my_utils import DatasetType
from dataset.micro_densisty_dataset import MicroDensityDataset
from logger import setup_logger
from networks.lstm_predictor import LstmPredictor
from networks.lstm_predictor2 import LstmPredictor2
from networks.lstm_predictor_attention import LstmPredictorWithAttention
from networks.transformer_predictor import TransformerPredictor
from training.trainer_lstm import TrainerLstmPredictor
from training.trainer_transformer import TrainerTransformerPredictor


def cli():
    """
   Parsing args
   @return:
   """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reset", "-r", action='store_true', default=False, help="Start retraining the model from scratch")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate of Adam optimized")
    parser.add_argument("--nb_epochs", "-e", type=int, default=1000, help="Number of epochs for training")
    parser.add_argument("--model_name", "-n",help="Name of the model. If not specified, it will be automatically generated")
    parser.add_argument("--num_workers", "-w", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--batch_size", "-bs", type=int, default=256, help="Batch size for training")
    parser.add_argument("--log_level", "-l", type=str, default="INFO")
    parser.add_argument("--autorun_tb","-tb",default=True,action='store_true',help="Autorun tensorboard")
    parser.add_argument("--use_census","-c",default=False, action='store_true',help="Use census data")
    parser.add_argument("--use_derivative", "-dv", default=False, action='store_true', help="Use derivate")
    parser.add_argument("--seq_len", "-sl", type=int, default=10, help="Sequencedee length")
    parser.add_argument("--seq_stride", "-ss", type=int, default=1, help="Sequence stride")

    ## Transformer arg
    parser.add_argument("--emb_dim", "-de", type=int, default=18, help="Embedding dimension of the transformer")
    parser.add_argument("--n_layers", "-nl", type=int, default=4, help="Number of layers of the transformer")
    parser.add_argument("--n_head", "-nh", type=int, default=3, help="Number of heads of the transformer")
    parser.add_argument("--dim_feedforward", "-df", type=int, default=256, help="Feedforward dimension of the transformer")
    parser.add_argument("--dropout_rate", "-do", type=float, default=0.05, help="Dropout of the transformer")
    parser.add_argument("--census_emb_dim", "-dce", type=int, default=4, help="Census embedding dimension")

    parser.add_argument("--test_only", "-i", default=False, action='store_true', help="Inference mode")
    return parser.parse_args()

def main(args):

    #Format the model nam

    if args.model_name is None:
        model_name = TransformerPredictor.format_model_name(
            use_census=args.use_census,
            use_derivative=args.use_derivative,
            emb_dim=args.emb_dim,
            census_emb_dim=args.census_emb_dim,
            n_layers=args.n_layers,
            n_head=args.n_head,
            dim_feedforward=args.dim_feedforward,
            seq_len=args.seq_len,
            seq_stride=args.seq_stride,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            dropout_rate=args.dropout_rate
        )
    else :
        model_name=args.model_name

    experiment_dir = os.path.join(EXPERIMENTS_DIR, model_name)
    #Try to load an existing config
    if not args.reset and os.path.exists(os.path.join(experiment_dir,"model.json")):
        logging.info("Loading existing config")
        config=json.load(open(os.path.join(experiment_dir,"model.json")))
        network = TransformerPredictor.build_from_config(config, experiment_dir, reset=False, load_best=False).to(DEVICE)
        args.seq_len=config["seq_len"]
        args.seq_stride=config["stride"]
        args.use_census=config["use_census"]
    else :
        network =TransformerPredictor(experiment_dir=experiment_dir,
                                      emb_dim=args.emb_dim,
                                      n_layers=args.n_layers,
                                      n_head=args.n_head,
                                      dim_feedforward=args.dim_feedforward,
                                      use_census=args.use_census,
                                      max_seq_len=args.seq_len-1,
                                      reset=args.reset,
                                      dropout_rate=args.dropout_rate,
                                      census_emb_dim=args.census_emb_dim).to(DEVICE)

    #Adam optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(30*args.batch_size/32),gamma=0.7,verbose=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4,cooldown=4, verbose=True,min_lr=1e-5)
    criterion= SmapeCriterion().to(DEVICE)


    logging.info("Training : "+model_name)
    trainer = TrainerTransformerPredictor(network,
                      criterion,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      nb_epochs= args.nb_epochs,
                      batch_size=args.batch_size,
                      reset=args.reset,
                      )

    # Save  the dataset according to type, seq_len_stride and use_census: using pickle
    if not os.path.exists(os.path.join(ROOT_DIR,"dataset","pickle")):
        os.makedirs(os.path.join(ROOT_DIR,"dataset","pickle"))

    datasets_pickle_path = os.path.join(ROOT_DIR,"dataset","pickle",f"prepared_dataset_{args.seq_len}_{args.seq_stride}_{args.use_census}.pickle")


    if not os.path.exists(datasets_pickle_path):

        train_dataset = MicroDensityDataset(type=DatasetType.TRAIN, seq_len=args.seq_len, stride=args.seq_stride,
                                            use_census=args.use_census)

        val_dataset = MicroDensityDataset(type=DatasetType.VALID, seq_len=args.seq_len, stride=args.seq_stride,
                                          use_census=args.use_census)

        test_dataset = MicroDensityDataset(type=DatasetType.TEST, seq_len=args.seq_len, stride=args.seq_stride,
                                           use_census=args.use_census)

        # train_dataset.mix_with(val_dataset,size=0.8) #Mix train and val dataset to avoid disparity between the two in terms of dates distribution


        with open(datasets_pickle_path,"wb") as f:
            logging.info(f"Saving datasets to {datasets_pickle_path}")
            pickle.dump((train_dataset,val_dataset,test_dataset),f)
    else:
        with open(datasets_pickle_path,"rb") as f:
            logging.info(f"Loading datasets  from {datasets_pickle_path}")
            train_dataset,val_dataset,test_dataset = pickle.load(f)
            test_dataset.tensor_list=dict()


    logging.info(f"Nb sequences : Train {len(train_dataset)} - Val {len(val_dataset)} - Test {len(test_dataset)}")

    train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True,drop_last=False,persistent_workers=args.num_workers>0,pin_memory=True)


    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,num_workers=0,drop_last=True,pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,num_workers=0,drop_last=False,shuffle=False)

    ##Train
    if not args.test_only:
        trainer.fit(train_dataloader,val_dataloader)

    ##Load best model
    trainer.network.load_state(best=True)
    trainer.run_test(test_dataloader=test_dataloader)


if __name__ == "__main__":
    args = cli()
    setup_logger(args)
    main(args)
 