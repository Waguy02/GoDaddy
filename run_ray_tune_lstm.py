import json
import os

import ray
import torch.optim
from ray import tune
from torch.optim import Adam
from torchmetrics import SymmetricMeanAbsolutePercentageError

from constants import DEVICE, EXPERIMENTS_DIR
from dataset.lstm_dataset import LstmDataset
from losses.smape import SmapeCriterion
from my_utils import DatasetType
from networks.lstm_predictor import LstmPredictor
from networks.lstm_predictor_attention import LstmPredictorWithAttention
from training.trainer_lstm import TrainerLstmPredictor

ray.init()


criterion= SmapeCriterion().to(DEVICE)


def train_fn(config, experiment_dir = os.path.join(EXPERIMENTS_DIR,"ray_tune")):
    model_name = "lstm_hd.{}_nl.{}_bs.{}_lr.{}_seq.{}_census.{}_derivative.{}".format(config["hidden_dim"],config["n_hidden_layers"],config["batch_size"],config["lr"],config["seq_len"],config["use_census"],config["use_derivative"])
    experiment_dir= os.path.join(experiment_dir,model_name)
    network = LstmPredictorWithAttention(
        hidden_dim=config["hidden_dim"],
        use_encoder=config["use_census"],
        use_derivative=config["use_derivative"],
        n_hidden_layers=config["n_hidden_layers"],
        experiment_dir= experiment_dir,

    ).to(DEVICE)


    train_dataset = LstmDataset(type=DatasetType.TRAIN, seq_len=config["seq_len"] ,use_census=config["use_census"])
    val_dataset = LstmDataset(type=DatasetType.VALID, seq_len=config["seq_len"], use_census=config["use_census"])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    optimizer=Adam(network.parameters(), lr=config["lr"])
    trainer = TrainerLstmPredictor(network,
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3 , min_lr=1e-5),
                                  nb_epochs=30,
                                  batch_size=config["batch_size"])
    trainer.fit(train_dataloader, val_dataloader)
    return trainer.best_val_loss



if __name__ == '__main__':
        analysis = tune.run(train_fn, config={"lr": tune.choice([0.01,0.001]),
                               "use_census": tune.choice([True, False]),
                                "use_derivative": tune.choice([True, False]),
                                "batch_size": tune.choice([512]),
                                "hidden_dim": tune.choice([2,4,8,12]),
                                "n_hidden_layers": tune.choice([1,2,3]),
                                "seq_len": tune.choice([6,10,15])}
                            ,max_concurrent_trials=2,
                            resources_per_trial={"cpu": 8, "gpu": 1},
                            log_to_file=True,
                            local_dir=os.path.join(EXPERIMENTS_DIR,"ray_tune_lstm"),
                            _experiment_checkpoint_dir=os.path.join(EXPERIMENTS_DIR,"ray_tune_lstm","checkpoints"),

             )





