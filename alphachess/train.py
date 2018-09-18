"""ChessWarrior train model"""

import logging
import os
import json
from queue import Queue

import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter

from .model import AlphaChess
from .config import Config
from .utils import ChessDataset

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.data_files = None
        self.batches = None
        self.epoch0 = 0
        self.loss = 0

    def start(self):
        if not self.model:
            try:
                # h5 file (model and weights)
                self.model = torch.load(os.path.join(self.config.resources.best_model_dir, "best_model.h5"))
                logger.info('load last trained best model.')
            except OSError:
                self.model = AlphaChess(config=self.config)
                logger.info('A new model is born.')

        

        with open(os.path.join(self.config.resources.best_model_dir, "epoch.txt"), "r") as file:
            self.epoch0 = int(file.read()) + 1
        
        self.dataset = ChessDataset(self.config)
        size = len(self.dataset)
        indices = list(range(size))
        split = int(np.floor(0.1 * size))
        np.random.shuffle(indices)
        
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train_loader = DataLoader(self.dataset, batch_size=self.config.training.batch_size, num_workers=4, sampler=train_sampler)
        
        self.valid_loader = DataLoader(self.dataset, batch_size=self.config.training.batch_size, num_workers=4, sampler=valid_sampler)

        self.training()

    def training(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self. model.to(device)
        writer = SummaryWriter()

        optimizer = optim.Adam(self.model.parameters(), 
                               lr=self.config.training.learning_rate,
                               weight_decay=self.config.training.l2_reg)

        policy_loss_func = nn.CrossEntropyLoss()
        value_loss_func = nn.MSELoss()

        for epoch in range(self.epoch0, self.epoch0 + self.config.training.epoches):
            logger.info('epoch %d start!' % epoch)
            total_loss = 0.0
            policy_loss = 0.0
            value_loss = 0.0

            for n_iter, sampled_batch in enumerate(self.train_loader):
                optimizer.zero_grad()

                s = sampled_batch['s'].to(device)
                a = sampled_batch['a'].to(device)
                r = sampled_batch['r'].to(device)

                p, v = self.model(s)

                policy_loss = policy_loss_func(p, a)
                value_loss = value_loss_func(v, r)

                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                policy_loss += policy_loss.item()
                value_loss += value_loss.item()
                writer.add_scalar('data/train/policy_loss', policy_loss, n_iter)
                writer.add_scalar('data/train/value_loss', value_loss, n_iter)
                writer.add_scalar('data/train/total_loss', total_loss, n_iter)

            total_loss = 0.0
            policy_loss = 0.0
            value_loss = 0.0
            with torch.no_grad():
                for n_iter, sampled_batch in enumerate(self.valid_loader):
                    s = sampled_batch['s'].to(device)
                    a = sampled_batch['a'].to(device)
                    r = sampled_batch['r'].to(device)

                    p, v = self.model(s)
                    policy_loss = policy_loss_func(p, a)
                    value_loss = value_loss_func(v, r)

                    loss = policy_loss + value_loss

                    total_loss += loss.item()
                    policy_loss += policy_loss.item()
                    value_loss += value_loss.item()
                    writer.add_scalar('data/val/policy_loss', policy_loss, n_iter)
                    writer.add_scalar('data/val/value_loss', value_loss, n_iter)
                    writer.add_scalar('data/val/total_loss', total_loss, n_iter)
            
            writer.export_scalars_to_json("./all_scalars.json")
        writer.close()
