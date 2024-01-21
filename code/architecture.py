import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import sklearn
from sklearn.metrics import mean_squared_error, r2_score, mean_poisson_deviance
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import flwr as fl 
from typing import Dict, List, Tuple
from collections import OrderedDict
import argparse
import run_config


# Define architecture (OPTUNA ARCHITECTURE)
class MultipleRegression(nn.Module):
    def __init__(self, num_features=39, num_units_1=10, num_units_2=15, activation=nn.Tanh, dropout_rate=0):
        super(MultipleRegression, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, num_units_1)
        self.layer_2 = nn.Linear(num_units_1, num_units_2)
        self.layer_out = nn.Linear(num_units_2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.act = activation()

        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.zeros_(self.layer_1.bias)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        torch.nn.init.zeros_(self.layer_2.bias)
        torch.nn.init.xavier_uniform_(self.layer_out.weight)
        torch.nn.init.zeros_(self.layer_out.bias)
    
    def forward(self, inputs):
        x = self.dropout(self.act(self.layer_1(inputs)))
        x = self.dropout(self.act(self.layer_2(x)))
        x = torch.exp(self.layer_out(x))

        return (x)

    def predict(self, test_inputs):
        x = self.act(self.layer_1(test_inputs))
        x = self.act(self.layer_2(x))
        x = torch.exp(self.layer_out(x))

        return (x)