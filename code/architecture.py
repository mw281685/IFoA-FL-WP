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


# Define architecture
class NeuralNetworks(torch.nn.Module):
    # define model elements
    def __init__(self, n_features):
        super(NeuralNetworks, self).__init__()
        self.hid1 = torch.nn.Linear(n_features, 5)
        self.hid2 = torch.nn.Linear(5, 10)
        self.hid3 = torch.nn.Linear(10, 15)
        self.drop = torch.nn.Dropout(run_config.model_architecture["dropout"])
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


# Define architecture (OPTUNA ARCHITECTURE ALT)
class MultipleRegressionALT(torch.nn.Module):
    # define model elements
    def __init__(self, num_features=39, num_units_1=60, num_units_2=20, num_units_3=60, activation=nn.ReLU, dropout_rate=0):
        super(MultipleRegressionALT, self).__init__()
        self.hid1 = torch.nn.Linear(num_features, num_units_1)
        self.hid2 = torch.nn.Linear(num_units_1, num_units_2)
        self.hid3 = torch.nn.Linear(num_units_2, num_units_3)
        self.drop = torch.nn.Dropout(dropout_rate)
        self.output = torch.nn.Linear(num_units_3, 1)

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


# Define architecture (OPTUNA ARCHITECTURE)
class MultipleRegression(nn.Module):
    def __init__(self, num_features=39, num_units_1=60, num_units_2=20, activation=nn.Tanh, dropout_rate=0):
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

def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
