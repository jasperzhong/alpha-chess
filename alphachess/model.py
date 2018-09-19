import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaChess(nn.Module):
    def __init__(self, config):
        super(AlphaChess, self).__init__()
        self.config = config
        
        self.init = nn.Sequential(
            nn.Conv2d(
                self.config.model.features,
                self.config.model.cnn_filter_num,
                self.config.model.cnn_first_filter_size,
                padding=(self.config.model.cnn_first_filter_size // 2)
            ),
            nn.BatchNorm2d(self.config.model.cnn_filter_num),
            nn.ReLU()
        )

        self.blocks = []
        for i in range(self.config.model.res_layer_num):
            self.blocks.append(ResidualBlock(
                self.config.model.cnn_filter_num,
                self.config.model.cnn_filter_num,
                self.config.model.cnn_filter_size
            ))
        
        self.res_layer = nn.Sequential(*self.blocks)

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                self.config.model.cnn_filter_num,
                12,
                1
            ),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
                

        self.value_head = nn.Sequential(
            nn.Conv2d(
                self.config.model.cnn_filter_num,
                12,
                1
            ),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )

        self.policy_linear = nn.Linear(8*8*12, 1968)

        self.value_linear1 = nn.Linear(8*8*12, 512)
        self.value_linear2 = nn.Linear(512, 1)

    def forward(self, x):
        s = self.init(x)
        s = self.res_layer(s)

        pi = self.policy_head(s)
        pi = self.policy_linear(pi.view(-1, 8*8*12))

        V = self.value_head(s)
        V = self.value_linear1(V.view(-1, 8*8*12))
        V = F.relu(V)
        V = self.value_linear2(V)
        V = torch.tanh(V)

        return pi, V 


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel,
                      padding=(kernel // 2)
                      ),
            nn.BatchNorm2d(out_channels),
            self.relu,
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel,
                      padding=(kernel // 2)
                      ),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        out = self.block(x)
        residual = x
        out += residual
        return self.relu(out)
    

if __name__=='__main__':
    x = torch.randn(32, 18, 8, 8)
    from config import Config 
    config = Config()
    model = AlphaChess(config)
    pi, V = model(x)