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
import flwr as fl 
from typing import Dict, List, Tuple
from collections import OrderedDict
import argparse
#import architecture as archit


def main():       
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--agents",
        type=int,
        default=10,
        choices=range(1, 11),
        required=False,
        help="Specifies the number of agents. \
        Picks 10 by default",
    )

    args = parser.parse_args()
    print(f'Arg.agents = {args.agents}')
    utils.prep_partitions(int(args.agents))  # args.partition
   
if __name__ == "__main__":
    main()

