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


# Define architecture
class NeuralNetworks(torch.nn.Module):
    # define model elements
    def __init__(self, n_features):
        super(NeuralNetworks, self).__init__()
        self.hid1 = torch.nn.Linear(n_features, 5)
        self.hid2 = torch.nn.Linear(5, 10)
        self.hid3 = torch.nn.Linear(10, 15)
        self.drop = torch.nn.Dropout(0.13690812525293783)
        self.output = torch.nn.Linear(15, 1)

        torch.nn.init.xavier_uniform_(self.hid1.weight)
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight)
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.hid3.weight)
        torch.nn.init.zeros_(self.hid3.bias)
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, X):
        z = torch.relu(self.hid1(X))
        z = torch.relu(self.hid2(z))
        z = torch.relu(self.hid3(z))
        z = self.drop(z)
        z = self.output(z)
        return z