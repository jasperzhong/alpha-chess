"""ChessWarrior train model"""

import logging
import os
import json
from queue import Queue
import gc

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
from .utils import ChessDataset,get_feature_plane

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
        self.model = AlphaChess(config=self.config)
        logger.info('A new model is born.')

        self.dataset = ChessDataset(self.config)
        size = len(self.dataset)
        indices = list(range(size))
        split = int(np.floor(0.01 * size))
        np.random.shuffle(indices)
        
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        self.train_size = len(train_idx)
        self.val_size = len(valid_idx)

        self.train_loader = DataLoader(self.dataset, batch_size=self.config.training.batch_size, num_workers=4, sampler=train_sampler)
        
        self.valid_loader = DataLoader(self.dataset, batch_size=self.config.training.batch_size, num_workers=4, sampler=valid_sampler)

        self.training()
    
    
    def training(self): 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            logger.info("mutil gpu %d " % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
        self. model.to(device)
        writer = SummaryWriter()

        optimizer = optim.Adam(self.model.parameters(), 
                               lr=self.config.training.learning_rate,
                               weight_decay=self.config.training.l2_reg)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        policy_loss_func = nn.CrossEntropyLoss()
        value_loss_func = nn.MSELoss()
        try:
            with open("data/model/min_val_loss","r") as f:
                min_val_loss = float(f.read())
        except:
            min_val_loss = 100.0
        
        for epoch in range(self.epoch0, self.epoch0 + self.config.training.epoches):
            logger.info('epoch %d start!' % epoch)
            total_loss = 0.0
            policy_loss = 0.0
            value_loss = 0.0

            for n_iter, batch in enumerate(self.train_loader):
                s,a,r = batch[0], batch[1], batch[2]
                
                optimizer.zero_grad()
                s = batch[0].to(device)
                a = batch[1].to(device)
                r = batch[2].to(device)
                
                p, v = self.model(s)
                policy_loss = policy_loss_func(p, a)
                value_loss = value_loss_func(v, r)
                
                loss = policy_loss + 1.5*value_loss
                loss.backward()
                optimizer.step()
                
                policy_loss = policy_loss.item()
                value_loss = value_loss.item()
                total_loss = loss.item()

  
                if n_iter % 20 == 0:
                    writer.add_scalar('data/train/policy_loss', policy_loss, n_iter + epoch*(self.train_size//self.config.training.batch_size) )
                    writer.add_scalar('data/train/value_loss', value_loss, n_iter + epoch*(self.train_size//self.config.training.batch_size) )
                    writer.add_scalar('data/train/total_loss', total_loss, n_iter + epoch*(self.train_size//self.config.training.batch_size) )
                    

                if n_iter % 500 == 0:
                    policy_loss_sum = 0.0
                    value_loss_sum = 0.0
                    with torch.no_grad():
                        cnt = 0
                        for batch in self.valid_loader:
                            s = batch[0].to(device)
                            a = batch[1].to(device)
                            r = batch[2].to(device)

                            p, v = self.model(s)
                            policy_loss = policy_loss_func(p, a)
                            value_loss = value_loss_func(v, r)

                            policy_loss_sum += policy_loss.item()
                            value_loss_sum += value_loss.item()
                            cnt += 1
                        policy_loss_sum /= cnt
                        value_loss_sum /= cnt
                        total_loss = policy_loss + 1.5*value_loss
                        writer.add_scalar('data/val/policy_loss', policy_loss_sum, n_iter + epoch*(self.train_size//self.config.training.batch_size) )
                        writer.add_scalar('data/val/value_loss', value_loss_sum, n_iter + epoch*(self.train_size//self.config.training.batch_size) )
                        writer.add_scalar('data/val/total_loss', total_loss, n_iter + epoch*(self.train_size//self.config.training.batch_size) )
                        
                        if total_loss < min_val_loss:
                            min_val_loss = total_loss.item()

                            state = {
                                'net':self.model.state_dict(),
                                'loss':min_val_loss,
                                'epoch':epoch,
                                'iter':n_iter
                            }
                            torch.save(state, 'data/model/best_model.pth')
                            logger.info("Epoch %d  Iter %d model saved!" % (epoch, n_iter))
                    
                
                if (n_iter + 1) % 1000 == 0:
                    scheduler.step()
 
        writer.close()
