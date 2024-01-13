
from tokenize import String
import utils
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import flwr as fl 
from typing import Dict, List, Tuple
from collections import OrderedDict
import argparse
import architecture as archit
import copy
import csv
from utils_quantisation import quantize, dequantize, modulus,  N, M, CR
from smpc_utils import load_noise, calc_noise, calc_noise_zero
import run_config
from run_config import EPOCHS, BATCH_SIZE, NUM_FEATURES, NUM_UNITS_1, NUM_UNITS_2, EPOCHS_LOCAL_GLOBAL, QUANTISATION, SMPC_NOISE

MODEL_PATH = '.'
ROUND_NO = 0

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

class IFoAClient(fl.client.NumPyClient):
    """FL model training client """
    
    def __init__(
        self,
        model: archit.MultipleRegression(num_features=run_config.NUM_FEATURES, num_units_1=25, num_units_2=2),  #archit.NeuralNetworks(NUM_FEATURES),
        optimizer, 
        criterion,
        trainset: torch.utils.data.dataset,
        valset: torch.utils.data.dataset,
        testset: torch.utils.data.dataset,
        num_examples: Dict,
        exposure: float,
        noise: List,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        self.num_examples = num_examples
        self.exposure = round(exposure) # we round it to int as we will use it to weight the parameter updates
        self.stats = [] 
        self.noise = noise # noise added to model parameter updates to secure aggregation of parameters updates


    def get_parameters(self, config) -> List[np.ndarray]:
        """Get local model parameters. 
           QUANTISATION flag- model parameters are mapped to integers
           SMPC_NOISE flag - noise is applied to model parameters
        """

        self.model.train()
        #st_dict_np = [np.round(val.cpu().numpy(),3) for _, val in self.model.state_dict().items()]
        #st_dict_np = [val.cpu().numpy().astype(np.float64) for _, val in self.model.state_dict().items()]
        #st_dict_np = [np.round(val.cpu().numpy(),5) for _, val in self.model.state_dict().items()]
        st_dict_np = [val.cpu().numpy().astype(np.float_) for _, val in self.model.state_dict().items()]
        
        global ROUND_NO

        if QUANTISATION:
            st_dict_np = quantize(st_dict_np, CR, N)
            k = 0
            for i in range(len(st_dict_np)):
                for j in range(len(st_dict_np[i])):
                    st_dict_np[i][j] = modulus(st_dict_np[i][j] + self.noise[ROUND_NO][k])         
                    k = k + 1
        elif SMPC_NOISE:
            k = 0
            for i in range(len(st_dict_np)):
                for j in range(len(st_dict_np[i])):
                    st_dict_np[i][j] += np.float64(self.noise[ROUND_NO][k])     
                    k = k + 1

        ROUND_NO = ROUND_NO + 1

        return st_dict_np 

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """ Set model parameters from a list of NumPy ndarrays"""

        self.model.train()
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)


    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int,  Dict]:
        """Train the model locally """

        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        trainLoader = DataLoader(self.trainset, batch_size=BATCH_SIZE, shuffle=True)
        valLoader = DataLoader(self.valset, batch_size=BATCH_SIZE)
        _, loss_stat = train(self.model, self.optimizer, self.criterion, trainLoader, valLoader, epochs=EPOCHS )
        self.stats.append(loss_stat)   # after FL local training append learning curves

        # test predictions on test dataset:
        testloader = DataLoader(self.testset, batch_size=BATCH_SIZE)
        loss = test(self.model, self.criterion, testloader)
        
        return self.get_parameters(config), 1, {'exposure': self.exposure}  # ms to test now 10.09.2023 prev :self.get_parameters(config), len(self.trainset), {'exposure': self.exposure}


    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        """Evaluate global model parameters on the local test data and return results """
        
        self.set_parameters(parameters)
    
        testloader = DataLoader(self.testset, batch_size=BATCH_SIZE)
        loss = test(self.model, self.criterion, testloader)

        return float(loss), len(self.testset), {"accuracy": loss/len(testloader)}


def train(
        model, 
        optimizer, 
        criterion,
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader, 
        epochs=EPOCHS
    ) :
    """Train model and print validation results for each epoch.  
        Return train model and loss statistics
    """

    
    loss_stats = {
                    'train': [],
                    'val': []
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


def test(model, criterion, val_loader)-> float:
    """ Test model 
    """ 
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

    utils.seed_torch() 
    
    print(f'Processing client {args.agent_id}')
    train_dataset, val_dataset, test_dataset, train_column_names, X_test_sc, exposure = utils.load_individual_data(args.agent_id)  # in folder data each training participant is storing their private, unique dataset 

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    model = archit.MultipleRegression(num_features=39, num_units_1=25, num_units_2=2)

    model.to(device)
    optimizer = optim.NAdam(model.parameters()) #optim.Adam(params=model.parameters(), lr=LEARNING_RATE)  

    # Set loss function change to true and then exp the output
    criterion = nn.PoissonNLLLoss(log_input= False, full= True) 

    if args.agent_id ==-1 :
        model_name = 'global_model.pt'
        PATH = '../ag_global/'
        AGENT_PATH = PATH + model_name
    else:
        model_name = 'local_model.pt'
        PATH = '../ag_' + str(args.agent_id) + '/' 
        AGENT_PATH = PATH + model_name 


    # Delete agent specific folder and create new one 
    #try:
    #    shutil.rmtree(PATH)
    #    os.mkdir(PATH)
    #except FileNotFoundError:
    #    os.mkdir(PATH)


    if args.if_FL==0:
        # Global model training
        _, loss_stats = train(model, optimizer, criterion, train_loader, val_loader, epochs=EPOCHS_LOCAL_GLOBAL )
        torch.save(model.state_dict(), AGENT_PATH)  

        f = open('../ag_global/los_stats.csv', 'w')
        loss_data = [loss_stats]

    else: 
        #Fl training

        # calculate noise vectors
        #noise = load_noise('noise_'+ str(args.agent_id) + '.csv')

        if SMPC_NOISE:
            noise = calc_noise('../data/seeds.csv', args.agent_id) #quantized
        else:
            noise = calc_noise_zero('../data/seeds.csv', args.agent_id) #quantized
    
        model_l = copy.deepcopy(model)
        optimizer_l = optim.NAdam(model_l.parameters()) #optim.Adam(params=model_l.parameters(), lr=LEARNING_RATE) #
        _, loss_stats = train(model_l, optimizer_l, criterion, train_loader, val_loader, epochs=EPOCHS_LOCAL_GLOBAL)
        torch.save(model_l.state_dict(), AGENT_PATH)  

        model_name = 'fl_model.pt'
        AGENT_PATH = '../ag_' + str(args.agent_id) + '/' + model_name 

        if args.agent_id in range(10):
        # contact FL server to join the training
    
            AGENT_PATH = '../ag_' + str(args.agent_id) + '/' + model_name 

            agent = IFoAClient(model, optimizer, criterion, train_dataset, val_dataset, test_dataset, {}, exposure, noise)
            fl.client.start_numpy_client(server_address="[::]:8080", client=agent,)     # 

            torch.save(model.state_dict(), AGENT_PATH)      

            if args.agent_id == 0:
                torch.save(model.state_dict(), '../ag_-1/' + model_name ) 
                print(model)


            f = open('../ag_' + str(args.agent_id) + '/los_stats.csv', 'w')
            loss_data = agent.stats
            #writer = csv.writer(f)

    writer = csv.DictWriter(f, fieldnames=['train', 'val'])
    writer.writeheader()
    writer.writerows(loss_data)
    f.close()

    print("FL training completed; loss_results saved!")

    return 0

if __name__ == "__main__":
          main()
