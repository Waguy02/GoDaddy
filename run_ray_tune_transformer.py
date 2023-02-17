import json
import os
import pickle

import ray
import torch.optim
from ray import tune
from torch.optim import Adam
from torchmetrics import SymmetricMeanAbsolutePercentageError

from constants import DEVICE, EXPERIMENTS_DIR, ROOT_DIR
from dataset.micro_densisty_dataset import MicroDensityDataset
from losses.smape import SmapeCriterion
from my_utils import DatasetType
from networks.lstm_predictor import LstmPredictor
from networks.lstm_predictor_attention import LstmPredictorWithAttention
from networks.transformer_predictor import TransformerPredictor
from training.trainer_lstm import TrainerLstmPredictor
from training.trainer_transformer import TrainerTransformerPredictor

criterion = SmapeCriterion().to(DEVICE)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def train_fn(config, experiment_dir=os.path.join(EXPERIMENTS_DIR, "ray_tune_trf")):
    model_name = TransformerPredictor.format_model_name(
        config["use_census"],
        config["use_derivative"],
        config["emb_dim"],
        config["census_emb_dim"],
        config["n_layers"],
        config["n_head"],
        config["dim_feedforward"],
        config["seq_len"],
        config["seq_stride"],
        config["learning_rate"],
        config["batch_size"],
        config["dropout_rate"]
    )
    experiment_dir = os.path.join(experiment_dir, model_name)
    network = TransformerPredictor.build_from_config(
        config,
        experiment_dir,
        reset=True,
        load_best=False
    ).to(DEVICE)

    # Prepare dataset
    if not os.path.exists(os.path.join(ROOT_DIR, "dataset", "pickle_ray_tune")):
        os.makedirs(os.path.join(ROOT_DIR, "dataset", "pickle_ray_tune"))
    datasets_pickle_path = os.path.join(ROOT_DIR, "dataset", "pickle_ray_tune",
                                        f"prepared_dataset_{config['seq_len']}_{config['seq_stride']}_{config['use_census']}.pickle")
    if not os.path.exists(datasets_pickle_path):
        train_dataset = MicroDensityDataset(type=DatasetType.TRAIN, seq_len=config["seq_len"],
                                            stride=config["seq_stride"], use_census=config["use_census"])
        val_dataset = MicroDensityDataset(type=DatasetType.VALID, seq_len=config["seq_len"], stride=config["seq_stride"],
                                          use_census=config["use_census"])
        # train_dataset.mix_with(val_dataset,size=0.8) #Mix train and val dataset to avoid disparity between the two in terms of dates distribution
        with open(datasets_pickle_path, "wb") as f:
            pickle.dump((train_dataset, val_dataset), f)
    else:
        with open(datasets_pickle_path, "rb") as f:
            train_dataset, val_dataset = pickle.load(f)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                                                   drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                                                 drop_last=False)

    optimizer = Adam(network.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30,cooldown=25, verbose=True,
                                                           min_lr=1e-5)
    trainer = TrainerTransformerPredictor(network,
                                          criterion,
                                          optimizer=optimizer,
                                          scheduler=scheduler,
                                          nb_epochs=config["nb_epochs"],
                                          batch_size=config["batch_size"],
                                          reset=False,
                                          )

    trainer.fit(train_dataloader, val_dataloader)
    tune.report(loss=trainer.best_val_loss)


if __name__ == '__main__':
    ray.init()
    analysis = tune.run(train_fn,
                        config={
                            "use_census": tune.grid_search([True]),
                            "use_derivative": tune.grid_search([True]),
                            "emb_dim": tune.grid_search([ 128,64, 32]),
                            "census_emb_dim": tune.grid_search([3, 5]),
                            "n_layers": tune.grid_search([6, 8]),
                            "n_head": tune.grid_search([8,4]),
                            "dim_feedforward": tune.grid_search([256, 512, 1024]),
                            "seq_len": tune.grid_search([30,20,10]),
                            "seq_stride": tune.grid_search([1]),
                            "learning_rate": tune.grid_search([0.001]),
                            "batch_size": tune.grid_search([256]),
                            "dropout_rate": tune.grid_search([0, 0.05, 0.001]),
                            "nb_epochs": 500,
                        },
                        resources_per_trial={"cpu": 1},
                        max_concurrent_trials=6
                        )

    # Get a dataframe for analyzing trial results.
    df = analysis.dataframe()
    df.to_csv(os.path.join(EXPERIMENTS_DIR, "ray_tune_trf", "analysis_results.csv"))
