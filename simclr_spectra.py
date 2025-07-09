import logging
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

from finetuning import finetuning_eval
import optuna

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("runs", f"{self.args.name}_{ts}")
        self.writer = SummaryWriter(log_dir=log_dir)
        # self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, trial=None):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, wl, pad_mask, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                wl = wl.repeat(self.args.n_views, 1)
                pad_mask = pad_mask.repeat(self.args.n_views, 1)

                images = images.to(self.args.device)
                wl = wl.to(self.args.device)
                pad_mask = pad_mask.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images, wl, pad_mask)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                    # evaluate the model on the test set
                    results_val = finetuning_eval(self.model, epochs=10, lr=0.01, batch_size=32, 
                                                repeats=1, k_spt=25, k_qry=25, device="cuda", 
                                                output_dir=None, emb_size=self.args.emb_size)
                    self.writer.add_scalar('val/mse', results_val.loc['mean']['mse_test'], global_step=n_iter)
                    self.writer.add_scalar('val/mae', results_val.loc['mean']['mae_test'], global_step=n_iter)
                    self.writer.add_scalar('val/r2', results_val.loc['mean']['r2_test'], global_step=n_iter)

                    if trial is not None:
                        # Report intermediate objective value for pruning
                        trial.report(results_val.loc['mean']['r2_test'], step=n_iter)
                        if trial.should_prune():
                            logging.warning("Trial was pruned.")
                            raise optuna.exceptions.TrialPruned()

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

            if epoch_counter % 25 == 0:
                logging.info(f"Saving model checkpoint at epoch {epoch_counter}.")
                # save model checkpoints
                os.makedirs(os.path.join(self.writer.log_dir, 'checkpoints'), exist_ok=True)
                checkpoint_name = 'checkpoints/checkpoint_{:04d}.pth.tar'.format(epoch_counter)
                save_checkpoint({
                    'epoch': epoch_counter,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))


        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
