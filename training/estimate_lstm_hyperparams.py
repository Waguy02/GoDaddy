import json
import os

import ray
import torch.optim
from ray import tune
from torchmetrics import SymmetricMeanAbsolutePercentageError

from constants import DEVICE, EXPERIMENTS_DIR
from dataset.lstm_dataset import LstmDataset
from my_utils import DatasetType
from networks.lstm_predictor import LstmPredictor
from training.trainer_lstm import TrainerLstmPredictor

ray.init()

loss_fn= SymmetricMeanAbsolutePercentageError().to(DEVICE)
criterion= lambda y_pred,y_true: loss_fn(y_pred,y_true)*100


def train_fn(config, experiment_dir = os.path.join(EXPERIMENTS_DIR,"base_lstm_no_ae")):

    network = LstmPredictor(
        experiment_dir=experiment_dir,
        hidden_dim=config["hidden_size"]
    ).to(DEVICE)


    train_dataset = LstmDataset(type=DatasetType.TRAIN, seq_len=config["seq_len"])
    val_dataset = LstmDataset(type=DatasetType.VALID, seq_len=config["seq_len"])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    trainer = TrainerLstmPredictor(network,
                                   experiment_dir=experiment_dir,
                                  criterion=criterion,
                                  optimizer=torch.optim.SGD(network.parameters(), lr=config["lr"]),
                                  nb_epochs=20,
                                  batch_size=config["batch_size"])
    trainer.fit(train_dataloader, val_dataloader)
    return trainer.best_val_loss



if __name__ == '__main__':
        tune.run(train_fn, config={"lr": tune.loguniform(1e-4, 1e-1),
                                "batch_size": tune.choice([16, 32, 64]),
                                "hidden_size": tune.choice([8, 16, 32]),
                                "seq_len": tune.choice([3, 5, 6])}
             )

        best_config=tune.get_best_config(metric="loss", mode="min")

        with open(os.path.join(EXPERIMENTS_DIR,"base_lstm_no_ae","best_hyper_config.json"), "w") as f:
            json.dump(best_config, f, indent=4)

