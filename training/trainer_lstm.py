import csv
import json
import logging
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from constants import DEVICE, STD_MB, MEAN_MB, NB_FUTURES
from my_utils import Averager



class TrainerLstmPredictor:
    """
    Class to manage the full training pipeline
    """
    def __init__(self, network,
                 criterion,
                 optimizer,
                 scheduler=None,
                 nb_epochs=10, batch_size=128, reset=False):
        """
        @param network:
        @param dataset_name:
        @param images_dirs:
        @param loss:
        @param optimizer:
        @param nb_epochs:
        @param nb_workers: Number of worker for the dataloader
        """
        self.network = network
        self.batch_size = batch_size
        self.loss_fn=criterion

        self.optimizer = optimizer
        self.scheduler =scheduler if scheduler else\
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=10,min_lr=1e-5)

        self.nb_epochs = nb_epochs
        self.experiment_dir = self.network.experiment_dir
        self.model_info_file = os.path.join(self.experiment_dir, "model.json")
        self.model_info_best_file = os.path.join(self.experiment_dir, "model_best.json")

        if reset:
            if os.path.exists(self.experiment_dir):
                shutil.rmtree(self.experiment_dir)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.start_epoch = 0
        if not reset and os.path.exists(self.model_info_file):
            with open(self.model_info_file, "r") as f:
                self.start_epoch = json.load(f)["epoch"] + 1
                self.nb_epochs += self.start_epoch
                logging.info("Resuming from epoch {}".format(self.start_epoch))


    def save_model_info(self, infos, best=False):
        json.dump(infos, open(self.model_info_file, 'w'),indent=4)
        if best: json.dump(infos, open(self.model_info_best_file, 'w'),indent=4)

    def fit(self,train_dataloader,val_dataloader):
        logging.info("Launch training on {}".format(DEVICE))
        if self.network.use_encoder:
            logging.info("Using encoder census data")

        self.summary_writer = SummaryWriter(log_dir=self.experiment_dir)
        itr = self.start_epoch * len(train_dataloader) * self.batch_size  ##Global counter for steps

        #Save model graph
        self.summary_writer.add_graph(self.network, next(iter(train_dataloader)).to(DEVICE))

        self.best_val_loss = 1e20  # infinity
        if os.path.exists(self.model_info_file):
            with open(self.model_info_file, "r") as f:
                model_info = json.load(f)
                lr=model_info["lr"]
                logging.info(f"Setting lr to {lr}")
                for g in self.optimizer.param_groups:
                    g['lr'] = lr

        if os.path.exists(self.model_info_best_file):
            with open(self.model_info_best_file, "r") as f:
                best_model_info = json.load(f)
                self.best_val_loss = best_model_info["val_loss"]


        for epoch in range(self.start_epoch, self.nb_epochs):  # Training loop
            self.network.train()
            """"
            0. Initialize loss and other metrics
            """
            running_loss=Averager()
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.nb_epochs}")
            for _, batch in enumerate(pbar):
                """
                Training lopp
                """
                self.optimizer.zero_grad()
                itr += self.batch_size
                """
                1.Forward pass
                """
                batch=batch.to(DEVICE)
                y_pred = self.network(batch)
                ## The output is the values of the density for each time step

                """
                2.Loss computation and other metrics
                """
                # The density is the last item of the batch
                y_true = batch[:,:,-1]

                nb_futures = min(train_dataloader.dataset.seq_len - 1, NB_FUTURES)
                if self.network.variante_num ==2:#Attention model: (single output)
                    loss = self.loss_fn(y_pred, y_true[:, -nb_futures:])
                else :
                    y_pred=y_pred.squeeze()
                    loss = self.loss_fn(y_pred[:, -1 - nb_futures:-1], y_true[:, -nb_futures:])



                """
                3.Optimizing
                """
                loss.backward()
                self.optimizer.step()


                running_loss.send(loss.cpu().item())
                pbar.set_postfix(current_loss=loss.cpu().item(), current_mean_loss=running_loss.value)

                """
                4.Writing logs and tensorboard data, loss and other metrics
                """
                self.summary_writer.add_scalar("Train/loss", loss.item(), itr)

                self.scheduler.step(loss.item())


            epoch_val_loss =self.eval(val_dataloader,epoch)

            infos = {
                "epoch": epoch,
                "train_loss":running_loss.value,
                "val_loss":epoch_val_loss.value,
                "lr": self.optimizer.param_groups[0]['lr'],
                "input_dim": self.network.input_dim,
                "hidden_dim": self.network.hidden_dim,
                "n_hidden_lstm_layers": self.network.n_hidden_layers,
                "seq_len": train_dataloader.dataset.seq_len,
                "batch_size": train_dataloader.batch_size,
                "stride": train_dataloader.dataset.stride,
                "use_census": self.network.use_encoder,
                "variante": self.network.variante_num,
                "census_dim": -1 if not self.network.use_encoder else self.network.features_encoder.hidden_dim
            }

            logging.info("Epoch {} - Train loss: {:.4f} - Val loss: {:.4f}".format(epoch, running_loss.value, epoch_val_loss.value))

            if epoch_val_loss.value < self.best_val_loss:
                self.best_val_loss = epoch_val_loss.value
                best = True
            else:
                best = False

            self.network.save_state(best=best)
            self.save_model_info(infos, best=best)

             # if scheduler is StepLR
            # if isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR):
            #     self.scheduler.step()
            # else:
            #     self.scheduler.step(epoch_val_loss.value)

            self.summary_writer.add_scalar("Epoch_train/loss", running_loss.value, epoch)
            self.summary_writer.add_scalar("Epoch_val/loss", epoch_val_loss.value, epoch)



    def eval(self, val_dataloader,epoch):
        """
        Compute loss and metrics on a validation dataloader
        @return:
        """
        with torch.no_grad():
            self.network.eval()
            running_loss=Averager()
            pbar = tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}/{self.nb_epochs}")
            for _, batch in enumerate(pbar):

                """
                Training lopp
                """
                """
                1.Forward pass
                """
                batch=batch.to(DEVICE)
                y_pred = self.network(batch)
                """ 
                2.Loss computation and other metrics
                """
                y_true = batch[:,:,-1]

                nb_futures = min(val_dataloader.dataset.seq_len - 1, NB_FUTURES)
                if self.network.variante_num == 2:  # Attention model: (single output)
                    loss = self.loss_fn(y_pred, y_true[:, -nb_futures:])
                else:
                    y_pred = y_pred.squeeze()
                    loss = self.loss_fn(y_pred[:, -1 - nb_futures:-1], y_true[:, -nb_futures:])

                running_loss.send(loss.item())

                pbar.set_postfix(current_loss=loss.item(), current_mean_loss=running_loss.value)

        return running_loss



    def run_test(self, test_dataloader):
        """
        Compute loss and metrics on a validation dataloader
        @return:
        """
        assert test_dataloader.dataset.batch_size == 1, "Batch size must be 1 for test"
        predictions = []
        with torch.no_grad():
            self.network.eval()
            for i, batch in enumerate(tqdm(test_dataloader," Running tests for submission")):
                batch = batch.to(DEVICE)
                y_pred = self.network(batch).cpu().squeeze().item()

                # Denormalize. MEAN_MB, STD_MB (if noramlized)
                # y_pred = y_pred * STD_MB + MEAN_MB
                """ 
                2.Loss computation and other metrics
                """
                predictions.append(y_pred)

                ##Update all microbusiness_denisty column
                id_row=test_dataloader.dataset.test_df.iloc[i]["id"]
                self.test_df.main_df.loc[id_row,"microbusiness_density"]=predictions[-1]

        #Merge predictions
        predictions=np.array(predictions)


        #Update all microbusiness_denisty column

        pred_test_df = pd.DataFrame(
            {
                "row_id":test_dataloader.dataset.test_df["row_id"].values,
                 "cfips":test_dataloader.dataset.test_df["cfips"].values,
                "first_day_of_month":test_dataloader.dataset.test_df["first_day_of_month"].values,
                "microbusiness_density":predictions}

                                )

        pred_test_df.to_csv(os.path.join(self.experiment_dir,"submission.csv"),index=False)

        return pred_test_df

