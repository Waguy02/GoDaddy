import csv
import json
import logging
import os
import shutil

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from constants import DEVICE, N_COUNTY, N_DIMS_COUNTY_ENCODING
from my_utils import Averager



class TrainerFeaturesAE:
    """
    Class to manage the training of the features autoencoder.
    """
    def __init__(self, network,
                 criterion_cfips,
                 crtierion_features,
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

        self.criterion_cfips=criterion_cfips
        self.criterion_features=crtierion_features

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
            running_loss_cfips=Averager()
            running_loss_features=Averager()

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.nb_epochs}")
            for _, batch in enumerate(pbar):
                """
                Training lopp
                """
                self.optimizer.zero_grad()
                itr += self.batch_size
                """ 
                1.Forward pass and get the reconstruction loss
                """

                h,y=self.network(batch.to(DEVICE))
                y_cfips, y_features=y[:,:N_DIMS_COUNTY_ENCODING],y[:,N_DIMS_COUNTY_ENCODING:]
                """
                2.Loss computation and other metrics
                """
                loss_cfips   = self.criterion_cfips(batch[:,:N_DIMS_COUNTY_ENCODING].to(DEVICE),y_cfips)
                loss_features= self.criterion_features(batch[:,N_DIMS_COUNTY_ENCODING:].to(DEVICE),y_features)
                loss=loss_cfips+10*loss_features

                """                
                3.Optimizing
                """

                loss.backward()
                loss.cpu()
                self.optimizer.step()

                running_loss_cfips.send(loss_cfips.item())
                running_loss_features.send(loss_features.item())
                running_loss.send(loss.item())
                """
                
        
                
                4.Writing logs and tensorboard data, loss and other metrics
                """
                self.summary_writer.add_scalar("Train/loss", loss.item(), itr)
                self.summary_writer.add_scalar("Train/loss_cfips", loss_cfips.item(), itr)
                self.summary_writer.add_scalar("Train/loss_features", loss_features.item(), itr)

                pbar.set_postfix(loss=running_loss.value,loss_cfips=running_loss_cfips.value,loss_features=running_loss_features.value)


            epoch_val_loss ,epoch_val_loss_cfips,epoch_val_loss_features=self.evaluate(val_dataloader,epoch)


            infos={"epoch": epoch,
                "lr": self.optimizer.param_groups[0]['lr'],
                "val_loss": epoch_val_loss.value,
                "val_loss_cfips": epoch_val_loss_cfips.value,
                "val_loss_features": epoch_val_loss_features.value,
                "train_loss": running_loss.value,
                "train_loss_cfips": running_loss_cfips.value,
                "train_loss_features": running_loss_features.value

             }
            #Print all the metrics
            logging.info("Epoch {} : train_loss = {:.4f} ,train_loss_cfips = {:.4f} ,train_loss_features = {:.4f} ,val_loss = {:.4f} ,val_loss_cfips = {:.4f} ,val_loss_features = {:.4f}".format(epoch,running_loss.value,running_loss_cfips.value,running_loss_features.value,epoch_val_loss.value,epoch_val_loss_cfips.value,epoch_val_loss_features.value))

            if running_loss.value < best_loss:
                best_loss = running_loss.value
                best = True
            else:
                best = False
            self.network.save_state(best=best)
            self.save_model_info(infos, best=best)


            ##Displaying the metrics : epoch_train_loss, epoch_val_loss, sparsity_eval_score using summary_writer
            self.summary_writer.add_scalar("Epoch_train/loss", running_loss.value, epoch)
            self.summary_writer.add_scalar("Epoch_train/loss_cfips", running_loss_cfips.value, epoch)
            self.summary_writer.add_scalar("Epoch_train/loss_features", running_loss_features.value, epoch)
            self.summary_writer.add_scalar("Epoch_val/loss", epoch_val_loss.value, epoch)
            self.summary_writer.add_scalar("Epoch_val/loss_cfips", epoch_val_loss_cfips.value, epoch)
            self.summary_writer.add_scalar("Epoch_val/loss_features", epoch_val_loss_features.value, epoch)


             #Scheduler step
            self.scheduler.step(epoch_val_loss.value)


    def evaluate(self, val_dataloader,epoch):
        """
        Compute loss and metrics on a validation dataloader
        @return:
        """
        with torch.no_grad():

            self.network.eval()
            running_loss=Averager()
            running_loss_cfips=Averager()
            running_loss_features=Averager()

            pbar= tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}/{self.nb_epochs}")
            for _, batch in enumerate(pbar):
                """
                Training lopp
                """
                """
                1.Forward pass
                """
                h, y = self.network(batch.to(DEVICE))
                y_cfips, y_features = y[:, :N_DIMS_COUNTY_ENCODING], y[:, N_DIMS_COUNTY_ENCODING:]
                """
                2.Loss computation and other metrics
                """
                loss_cfips   = self.criterion_cfips(batch[:,:N_DIMS_COUNTY_ENCODING].to(DEVICE),y_cfips)
                loss_features= self.criterion_features(batch[:,N_DIMS_COUNTY_ENCODING:].to(DEVICE),y_features)

                loss = loss_cfips+loss_features

                running_loss_cfips.send(loss_cfips.item())
                running_loss_features.send(loss_features.item())
                running_loss.send(loss.item())

                pbar.set_postfix(loss=running_loss.value, loss_cfips=running_loss_cfips.value,
                                 loss_features=running_loss_features.value)

        return running_loss,running_loss_cfips,running_loss_features


