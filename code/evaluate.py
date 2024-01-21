
from tokenize import String
import utils
import time
import os
import warnings
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
from typing import Dict, List, Tuple
from collections import OrderedDict
import argparse
import architecture as archit
import copy

NUM_FEATURES = 39


def main():

 
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--agent_id",
        type=int,
        default=-1,
        choices=range(-1, 10),
        required=False,
        help="Specifies the partition of data to be used for training. -1 means all data . \
        Picks partition 0 by default",
    )

    args = parser.parse_args()

    print(f'Federated Learning Model predictions for agent {args.agent_id}')
    train_dataset, val_dataset, test_dataset, train_column_names, X_test_sc = utils.load_individual_data(args.agent_id) 

    model = archit.NeuralNetworks(NUM_FEATURES)

    model_name = 'fl_model.pt'
    AGENT_PATH = '../ag_' + str(args.agent_id) + '/' + model_name 

    model.load_state_dict(torch.load(AGENT_PATH))
    model.eval()
    pred = model(X_test_sc)
    print(torch.exp(pred))

    print('--------------------------------------------------')
    print(f'Local Model predictions for agent {args.agent_id}')
    
    model_name = 'local_model.pt'
    AGENT_PATH = '../ag_' + str(args.agent_id) + '/' + model_name 

    model.load_state_dict(torch.load(AGENT_PATH))
    model.eval()
    pred = model(X_test_sc)
    print(torch.exp(pred))


if __name__ == "__main__":
          main()

