#from tokenize import String
import utils
import time
import os
import warnings
import numpy as np
import pandas as pd
import torch as th
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import sklearn
from sklearn.metrics import mean_squared_error, r2_score, mean_poisson_deviance
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import flwr as fl 
from typing import Dict, List, Tuple
from collections import OrderedDict
import argparse
import architecture as archit
import copy
import run_config
import shutil

MODEL_PATH = '.'
EPOCHS = run_config.model_architecture["epochs"]
#BATCH_SIZE = run_config.model_architecture["batch_size"] #1000 # Wutrich suggestion this may be better at 6,000 or so, 488169
BATCH_SIZE = 1_000
NUM_FEATURES = run_config.model_architecture["num_features"]
LEARNING_RATE = run_config.model_architecture["learning_rate"] #6.888528294546944e-05 #0.013433393353340668 #6.888528294546944e-05
#LEARNING_RATE = 0.1 #6.888528294546944e-05 #0.013433393353340668 #6.888528294546944e-05
NUM_ROUNDS = run_config.server_config["num_rounds"] 
EPOCHS_LOCAL_GLOBAL = run_config.EPOCHS_LOCAL_GLOBAL

device = th.device("cpu" if th.cuda.is_available() else "cpu")

def training_graph(loss_stats):
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    plt.figure(figsize=(15,8))
    plot = sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    fig = plot.get_figure()
    return fig


for ag in range(0,10):
    # Set seed
    utils.seed_torch()

    # Read in data
    train_dataset, val_dataset, test_dataset, train_column_names, X_test_sc, exposure = utils.load_individual_data(ag)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)

    # Instantiate Model
    model = archit.MultipleRegression(num_features=39, num_units_1=60, num_units_2=20)
    model.to(device)

    # Set Optimiser
    optimizer = optim.NAdam(params=model.parameters(), lr=LEARNING_RATE)

    # Set criterion
    criterion = nn.PoissonNLLLoss(log_input= False, full= True) 

    # Train
    # Before we start our training, let’s define a dictionary which will store the loss/epoch for both train and validation sets.
    loss_stats = {
    'train': [],
    "val": []
    }

    print(f"Begin training agent {ag}")
    for e in range(EPOCHS):
        
        # TRAINING
        train_epoch_loss = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            
            y_train_pred = model(X_train_batch)
            
            train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
            
            train_loss.backward()
            optimizer.step()
            
            train_epoch_loss += train_loss.item()
            
            
        # VALIDATION    
        with th.no_grad():
            
            val_epoch_loss = 0
            
            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                
                y_val_pred = model(X_val_batch)
                            
                val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))
                
                val_epoch_loss += val_loss.item()

        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))                              
        
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')

    training_loss_graph = training_graph(loss_stats)
    training_loss_graph.savefig(f'../ag_{ag}/' + 'agent_' + str(ag) + '_training_loss_chart')

#if __name__ == "__main__":
          #main()