# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:19:58 2021

@author: jurri
"""

import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
      