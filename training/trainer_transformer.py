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
from networks.transformer_predictor import TransformerPredictor


class TrainerTransformerPredictor:
    """
    Class to manage the full training pipeline
    """
    def __init__(self, network: TransformerPredictor,
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
        if self.network.use_census:
            logging.info("Using encoder census data")

        self.summary_writer = SummaryWriter(log_dir=self.experiment_dir)
        itr = self.start_epoch * len(train_dataloader) * self.batch_size  ##Global counter for steps

        #Save model graph
        # self.summary_writer.add_graph(self.network, next(iter(train_dataloader)).to(DEVICE)[:,:-1,:])

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
                self.optimizer.zero_grad(

                )
                itr += self.batch_size
                """
                1.Forward pass
                """
                batch = batch.to(DEVICE)

                y_pred = self.network(batch)
                ## The output is the values of the density for each time step

                """
                2.Loss computation and other metrics
                """
                # The density is the last item of the batch
                y_true = batch[:,:,-1].to(DEVICE)
                loss = self.loss_fn(y_pred, y_true[:, -1:])

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



            epoch_val_loss =self.eval(val_dataloader,epoch)

            #If step lr scheduler and currrent lr is not lower than 1e-5
            current_lr= self.optimizer.param_groups[0]['lr']
            if current_lr>=2e-6:
                if isinstance(self.scheduler,torch.optim.lr_scheduler.StepLR):
                    self.scheduler.step()
                else:
                    self.scheduler.step(epoch_val_loss.value)

            infos = {
                "epoch": epoch,
                "train_loss":running_loss.value,
                "val_loss":epoch_val_loss.value,
                "lr": self.optimizer.param_groups[0]['lr'],
                "batch_size": train_dataloader.batch_size,
                "stride": train_dataloader.dataset.stride,

            }
            infos.update(self.network.config)


            logging.info("Epoch {} - Train loss: {:.4f} - Val loss: {:.4f}".format(epoch, running_loss.value, epoch_val_loss.value))

            if epoch_val_loss.value < self.best_val_loss:
                self.best_val_loss = epoch_val_loss.value
                best = True
            else:
                best = False

            self.network.save_state(best=best)
            self.save_model_info(infos, best=best)


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


                loss = self.loss_fn(y_pred, y_true[:, -1:])

                running_loss.send(loss.item())

                pbar.set_postfix(current_loss=loss.item(), current_mean_loss=running_loss.value)


        return running_loss



    def run_test(self, test_dataloader):
        """
        Compute loss and metrics on a validation dataloader
        @return:
        """
        assert test_dataloader.batch_size == 1, "Batch size must be 1 for test"
        predictions = []
        pred_by_cfips={} #For each cfips, we store the predictions, we keep only the first prediction
        row_ids = []
        with torch.no_grad():
            self.network.eval()
            for i, input in enumerate(tqdm(test_dataloader," Running tests for submission")):
                input = input.to(DEVICE)
                y_pred = self.network(input.to(DEVICE)).cpu().squeeze().item()

                # Denormalize. MEAN_MB, STD_MB (if noramlized)
                # y_pred = y_pred * STD_MB + MEAN_MB
                """ 
                2.Loss computation and other metrics
                """


                ##Update all microbusiness_den isty column
                row_id=test_dataloader.dataset.test_df.loc[i,"row_id"]
                row_ids.append(row_id)

                cfips=test_dataloader.dataset.test_df.loc[i,"cfips"]
                if cfips not in pred_by_cfips:
                    pred_by_cfips[cfips]=y_pred
                else:
                    y_pred=pred_by_cfips[cfips]#We keep only the first prediction

                predictions.append(y_pred)

                test_dataloader.dataset.main_df.loc[test_dataloader.dataset.main_df["row_id"]==row_id,"microbusiness_density"]=y_pred


        #Merge predictions
        predictions=np.array(predictions)


        #Update all microbusiness_denisty column

        pred_test_df = pd.DataFrame(
            {
                "row_id":row_ids,
                 "microbusiness_density":predictions}

                                )
        pred_test_df.to_csv(os.path.join(self.experiment_dir,"submission.csv"),index=False)

        return pred_test_df

