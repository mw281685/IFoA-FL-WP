# benchmark : FLuseCase_NN_secureProtocol2_25e_ClaudioModel_optuna.ipynb

from tokenize import String
import utils
import time
import os
import warnings
import numpy as np
#import pandas as pd
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
import architecture as archit
import copy

MODEL_PATH = '.'
EPOCHS = 25
BATCH_SIZE = 1000 # Wutrich suggestion this may be better at 6,000 or so, 488169
NUM_FEATURES = 39
LEARNING_RATE = 6.888528294546944e-05
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


# Flower Client
class IFoAClient(fl.client.NumPyClient):
    """Flower client """
    def __init__(
        self,
        model: archit.NeuralNetworks(NUM_FEATURES),
        optimizer, 
        criterion,
        trainset: torch.utils.data.dataset,
        valset: torch.utils.data.dataset,
        testset: torch.utils.data.dataset,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        self.num_examples = num_examples

    def get_parameters(self, config) -> List[np.ndarray]:
        state_dict_elements = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        print('[GET_PARAMETERS CALLED ]:', self.model.state_dict()['hid1.weight'][0])
        self.model.train()
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]


    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        
        self.model.train()
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        print('[SET_PARAMETERS CALLED]: ', self.model.state_dict()['hid1.weight'][0])



    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:

        # Set model parameters, train model, return updated model parameters
        # Update local model parameters
        self.set_parameters(parameters)
        trainLoader = DataLoader(self.trainset, batch_size=BATCH_SIZE, shuffle=True)
        valLoader = DataLoader(self.valset, batch_size=BATCH_SIZE)
        train(self.model, self.optimizer, self.criterion, trainLoader, valLoader, epochs=10 )
        return self.get_parameters(config), len(self.trainset), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
    
        # Evaluate global model parameters on the local test data and return results
        testloader = DataLoader(self.testset, batch_size=512)
        loss = test(self.model, self.criterion, testloader)#/len(testloader)
        return float(loss), len(self.testset), {"accuracy": loss/len(testloader)}
#        loss = test(self.model, self.criterion, testloader)/len(testloader)
#        return float(loss), len(self.testset), {"val_loss": float(loss)}



def train(model:archit.NeuralNetworks(NUM_FEATURES), optimizer, criterion,
                  train_loader: torch.utils.data.DataLoader, 
                  val_loader: torch.utils.data.DataLoader, 
                  epochs=EPOCHS):
    
    loss_stats = {
    'train': [],
    "val": []
    }

    for e in range(epochs):
        # TRAINING
        train_epoch_loss = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            
            train_loss.backward()
            optimizer.step()
            
            train_epoch_loss += train_loss.item()
            
        val_epoch_loss = test(model, criterion,  val_loader)


        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))                              
        
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')

    return model, loss_stats


def test(model, criterion, val_loader):
        # VALIDATION    
    with torch.no_grad():
        val_epoch_loss = 0   
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            y_val_pred = model(X_val_batch)
            val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))      
            val_epoch_loss += val_loss.item()

    return val_epoch_loss


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
    parser.add_argument(
        "--agents_no",
        type=int,
        default=10,
        choices=range(1, 11),
        required=False,
        help="Specifies the number of FL participants. \
        Picks partition 10 by default",
    )
    
    parser.add_argument(
        "--if_FL",
        type=int,
        default=1,
        choices=range(0,2),
        required=False,
        help="Specifies the pipeline type: centralised (0) of FL (1). \
        Picks 1 by default",
    )



    args = parser.parse_args()

    utils.seed_torch() # for FL training, we cannot set seed, as everyone will have same rng
    
    print(f'args = {args}')

    print(f'Processing client {args.agent_id}')
    NUM_AGENTS = 3
#    train_dataset, val_dataset, test_dataset, train_column_names = utils.load_partition(NUM_AGENTS, args.partition) 
    train_dataset, val_dataset, test_dataset, train_column_names, X_test_sc = utils.load_individual_data(args.agent_id)  # in folder my_data each training participant is storing their private, unique dataset 
#    train_dataset, val_dataset, test_dataset, train_column_names = utils.load_partition(args.agent_id, args.agents_no)  # args.partition
   
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

   

    model = archit.NeuralNetworks(NUM_FEATURES)
    model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # Set loss function change to true and then exp the output
    criterion = nn.PoissonNLLLoss(log_input= True, full= True)

    if args.if_FL==0:
        # Global model training args.partition =-1
        train(model, optimizer, criterion, train_loader, val_loader, 5 )

        if args.agent_id ==-1 :
            model_name = 'global_model.pt'
            AGENT_PATH = '../ag_global/' + model_name
        else:
            model_name = 'local_model.pt'      
            AGENT_PATH = '../ag_' + str(args.agent_id) + '/' + model_name 
    else:

        model_l = copy.deepcopy(model)
        optimizer_l = optim.Adam(params=model_l.parameters(), lr=LEARNING_RATE)
        train(model_l, optimizer_l, criterion, train_loader, val_loader, 5)
        model_name = 'local_model.pt'      
        AGENT_PATH = '../ag_' + str(args.agent_id) + '/' + model_name 
        torch.save(model_l.state_dict(), AGENT_PATH)  


    #Fl training     
        model_name = 'fl_model.pt'
        AGENT_PATH = '../ag_' + str(args.agent_id) + '/' + model_name 

        if args.agent_id in range(10):
            AGENT_PATH = '../ag_' + str(args.agent_id) + '/' + model_name 
            agent = IFoAClient(model, optimizer, criterion, train_dataset, val_dataset, test_dataset, {})
            fl.client.start_numpy_client(server_address="[::]:8080", client=agent,)     # when running server locally ! 
#            fl.client.start_numpy_client("193.0.96.129:8080", client) # Polish server ! Make sure Malgorzata starts it :) , otherwise it won't work

#            fl.client.start_numpy_client("193.0.96.129:6555", client) # Polish server ! Make sure Malgorzata starts it :) , otherwise it won't work
       
    torch.save(model.state_dict(), AGENT_PATH)            
    

if __name__ == "__main__":
          main()


