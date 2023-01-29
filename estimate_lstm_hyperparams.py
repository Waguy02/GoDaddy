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
from training.trainer_lstm import TrainerLstmPredictor

ray.init()


criterion= SmapeCriterion().to(DEVICE)


def train_fn(config, experiment_dir = os.path.join(EXPERIMENTS_DIR,"ray_tune")):

    model_name="lstm_h.{}_bs.{}_lr.{}_seq.{}_census.{}".format(config["hidden_dim"],config["batch_size"],config["lr"],config["seq_len"],config["use_census"])
    experiment_dir= os.path.join(experiment_dir,model_name)
    network = LstmPredictor(
        experiment_dir= experiment_dir,
        use_encoder= config["use_census"],
        hidden_dim = config["hidden_dim"],
    ).to(DEVICE)


    train_dataset = LstmDataset(type=DatasetType.TRAIN, seq_len=config["seq_len"] ,use_census=config["use_census"])
    val_dataset = LstmDataset(type=DatasetType.VALID, seq_len=config["seq_len"], use_census=config["use_census"])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    optimizer=Adam(network.parameters(), lr=config["lr"])
    trainer = TrainerLstmPredictor(network,
                                  criterion=criterion,
                                  optimizer=optimizer,
                                    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5,min_lr=1e-5),
                                  nb_epochs=20,
                                  batch_size=config["batch_size"])
    trainer.fit(train_dataloader, val_dataloader)
    return trainer.best_val_loss



if __name__ == '__main__':
        tune.run(train_fn, config={"lr": tune.loguniform(1e-4, 1e-1),
                               "use_census": tune.choice([True, False]),
                                "batch_size": tune.choice([32,256]),
                                "hidden_dim": tune.choice([2, 4,6,8]),
                                "seq_len": tune.choice([4, 5, 6])}

             )

        best_config=tune.get_best_config(metric="loss", mode="min")

        with open(os.path.join(EXPERIMENTS_DIR,"base_lstm_no_ae","best_hyper_config.json"), "w") as f:
            json.dump(best_config, f, indent=4)

