import csv
import json
import logging
import os
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from constants import DEVICE
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
        self.summary_writer = SummaryWriter(log_dir=self.experiment_dir)
        itr = self.start_epoch * len(train_dataloader) * self.batch_size  ##Global counter for steps

        #Save model graph
        self.summary_writer.add_graph(self.network, next(iter(train_dataloader)).to(DEVICE))

        best_loss = 1e20  # infinity
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
                best_loss = best_model_info["val_loss"]


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
                y_pred = self.network(batch).squeeze()
                ## The output is the values of the density for each time step

                """
                2.Loss computation and other metrics
                """
                # The density is the last item of the batch
                y_true = batch[:,:,-1]

                # nb_futures= min(train_dataloader.dataset.seq_len-1,3)
                nb_futures=train_dataloader.dataset.seq_len-1
                loss=self.loss_fn(y_pred[:,-1-nb_futures:-1],y_true[:,-nb_futures:])

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
                self.summary_writer.add_scalar("step_Train/loss", loss.item(), itr)



            epoch_val_loss =self.eval(val_dataloader,epoch)

            infos = {
                "epoch": epoch,
                "train_loss":running_loss.value,
                "val_loss":epoch_val_loss.value,
                "lr": self.optimizer.param_groups[0]['lr']
            }

            logging.info("Epoch {} - Train loss: {:.4f} - Val loss: {:.4f}".format(epoch, running_loss.value, epoch_val_loss.value))

            if running_loss.value < best_loss:
                best_loss = running_loss.value
                best = True
            else:
                best = False
            self.network.save_state(best=best)
            self.save_model_info(infos, best=best)
            self.scheduler.step(epoch_val_loss.value)
            self.summary_writer.add_scalar("epoch_train/loss", running_loss.value, epoch)
            self.summary_writer.add_scalar("epoch_val/loss", epoch_val_loss.value, epoch)


    def eval(self, val_dataloader,epoch):
        """
        Compute loss and metrics on a validation dataloader
        @return:
        """
        with torch.no_grad():
            self.network.eval()
            running_loss=Averager()
            for _, batch in enumerate(tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}/{self.nb_epochs}")):

                """
                Training lopp
                """
                """
                1.Forward pass
                """
                batch=batch.to(DEVICE)
                y_pred = self.network(batch).squeeze()
                """ 
                2.Loss computation and other metrics
                """
                y_true = batch[:,:,-1]
                loss = self.loss_fn(y_pred[:, :-1], y_true[:, 1:])
                running_loss.send(loss.item())


        return running_loss


