
import argparse
import utils
import run_config
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import torch as th
import torch.nn.functional as F
import torch.nn as nn 
from torch import optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split,  RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_poisson_deviance, d2_tweedie_score, make_scorer, auc
import seaborn as sns
import matplotlib.pyplot as plt
import random
from scipy import stats
import os
from skorch import NeuralNetRegressor, NeuralNet, callbacks
from skorch.helper import predefined_split
import skorch
from skorch.dataset import Dataset, ValidSplit
import utils

DATA_PATH = run_config.dataset_config["path"]
SEED = run_config.dataset_config["seed"]
 
# Formatting options to print dataframe to terminal
pd.set_option('display.max_columns', 7)
pd.set_option('display.width', 200)


def seed_torch(seed=SEED):
    th.manual_seed(seed)
    #random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    #np.random.seed(seed)  # Numpy module.
    #random.seed(seed)  # Python random module.
    #th.manual_seed(seed)
    th.backends.cudnn.benchmark = True
    th.backends.cudnn.deterministic = True

seed_torch()

rng = np.random.RandomState(SEED) 

# Define architecture
class MultipleRegression(nn.Module):
    def __init__(self, num_features=39, num_units_1=50, num_units_2=60, activation=nn.Tanh, dropout_rate=0):
        super(MultipleRegression, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, num_units_1)
        self.layer_2 = nn.Linear(num_units_1, num_units_2)
        self.layer_out = nn.Linear(num_units_2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.act = activation()

        th.nn.init.xavier_uniform_(self.layer_1.weight)
        th.nn.init.zeros_(self.layer_1.bias)
        th.nn.init.xavier_uniform_(self.layer_2.weight)
        th.nn.init.zeros_(self.layer_2.bias)
        th.nn.init.xavier_uniform_(self.layer_out.weight)
        th.nn.init.zeros_(self.layer_out.bias)
    
    def forward(self, inputs):
        x = self.dropout(self.act(self.layer_1(inputs)))
        x = self.dropout(self.act(self.layer_2(x)))
        x = th.exp(self.layer_out(x))

        return (x)

    def predict(self, test_inputs):
        x = self.act(self.layer_1(test_inputs))
        x = self.act(self.layer_2(x))
        x = th.exp(self.layer_out(x))

        return (x)

# Create PDE score to print during training/tuning
pde_score = make_scorer(d2_tweedie_score, power=1)
# Create callback to display PDE after each epoch on validation set
pde_callback = callbacks.EpochScoring(pde_score, lower_is_better=False, name='PDE')
# Create callback to do early stopping if PDE doesn't improve for 5 epochs on validation PDE
early_stopping_callback = callbacks.EarlyStopping(monitor='PDE', lower_is_better=False, patience=5)
# Create callback to save and reload highest PDE on validation set
check_point_callback = callbacks.Checkpoint(monitor='PDE_best', load_best=True)

# Grid Search space dictionary
params = {
    'optimizer__lr': [0.0001, 0.001, 0.01], # 3
    'batch_size':[1_00, 1_000, 10_000], # 3
    'module__num_units_1': [20, 40 ,60],# 3
    'module__num_units_2': [20, 40 ,60], # 3
}

def main():

    # Create list to store results
    all_results_list = []
    
    for ag in range(-1,10):
        
        print(f'\n Tuning Agent = {ag}', end='\n')
        
        X_train, y_train, X_val, y_val, X_test, y_test, X_column_names, exposure = utils.load_individual_skorch_data(ag)  # in folder my_data each training participant is storing their private, unique dataset    
       
        X_trainval_ordered = np.concatenate((X_train, X_val))
        y_trainval_ordered = np.concatenate((y_train, y_val))

        # The indices which have the value -1 will be kept in val.
        train_indices = np.full((len(X_train),), -1, dtype=int)

        # The indices which have zero or positive values, will be kept in val
        val_indices = np.full((len(X_val),), 0, dtype=int)

        val_fold = np.append(train_indices, val_indices)

        ps = PredefinedSplit(val_fold)
    
        # Define weighted PDE scorer
        weighted_pde_score = make_scorer(d2_tweedie_score, sample_weight=X_val['Exposure'], power=1, greater_is_better=True)

        # Define skorch neural network
        net_regr = NeuralNetRegressor(
            MultipleRegression().double(),
            optimizer=optim.NAdam,
            criterion=nn.PoissonNLLLoss(log_input= False, full= True),
            max_epochs=50,
            #batch_size=10_000,
            train_split=skorch.dataset.ValidSplit(0.1, stratified=False, random_state=SEED), # use 10% to set early stopping
            callbacks=[pde_callback, early_stopping_callback, check_point_callback],
            device=None, # ignore CUDA for now
            iterator_train__shuffle=True
            )
                
        gs = RandomizedSearchCV(net_regr,
                        params,
                        refit=True,
                        cv=ps,
                        scoring=weighted_pde_score,
                        n_iter=15, # grid size
                        #n_jobs=4, # turning off mutli-threading as issues with reproducibility
                        random_state=SEED
                        )
        
        gs.fit(X_trainval_ordered.astype(np.float32), y_trainval_ordered.reshape(-1, 1).astype(np.float32))

        # Save model
        gs.best_estimator_.save_params(f_params=f'../ag_{ag}/agent_' + str(ag) + '_model.pkl', 
                                       f_optimizer=f'../ag_{ag}/agent_' + str(ag) + '_opt.pkl', 
                                       f_history=f'../ag_{ag}/agent_' + str(ag) + '_history.json')

        print(f'\n Agent={ag} Results:', end='\n')
        
        results_df = pd.DataFrame(gs.cv_results_)
        results_df = results_df.sort_values(by=["rank_test_score"])
        results_df = results_df.set_index(
            results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
        ).rename_axis("params_key")
        results_df['agent']=ag
        
        print(results_df[["params", "rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time"]])

        # Get number of epochs
        results_df['best_epochs'] = len(gs.best_estimator_.history)

        # Create model predictions
        predictions = gs.best_estimator_.predict(X_test.to_numpy().astype(np.float32))

        print(f'\n Agent={ag} Test Statistics:')

        def test_statistics(y_test, y_pred_list_exp):
            mpd = mean_poisson_deviance(y_test, y_pred_list_exp)
            weighted_mpd = mean_poisson_deviance(y_test, y_pred_list_exp, sample_weight=X_test['Exposure'])
            pde = d2_tweedie_score(y_test, y_pred_list_exp, power=1)
            weighted_pde = d2_tweedie_score(y_test, y_pred_list_exp, sample_weight=X_test['Exposure'], power=1)
            mse = mean_squared_error(y_test, y_pred_list_exp)
            r_square = r2_score(y_test, y_pred_list_exp)
            cum_exposure, cum_claims = utils.lorenz_curve(y_test, y_pred_list_exp, X_test['Exposure'])
            gini = 1 - 2 * auc(cum_exposure, cum_claims)
            print("Mean Poisson Deviance :",mpd)
            print("Weighted Mean Poisson Deviance :",weighted_mpd)
            print("Poisson Deviance Explained:",pde)
            print("Weighted Poisson Deviance Explained:",weighted_pde)
            print("Mean Squared Error :",mse)
            print("R^2 :",r_square)
            print("Gini :",gini)
            print(stats.describe(y_pred_list_exp))

        test_statistics(y_test, predictions)

        # Store test statistics to dataframe
        results_df['test_mean_poisson_deviance']=mean_poisson_deviance(y_test, predictions)
        results_df['test_weighted_mean_poisson_deviance']=mean_poisson_deviance(y_test, predictions, sample_weight=X_test['Exposure'])
        results_df['test_pde']=d2_tweedie_score(y_test, predictions, power=1)
        results_df['test_weighted_pde']=d2_tweedie_score(y_test, predictions, sample_weight=X_test['Exposure'], power=1)
        results_df['test_r^2']=r2_score(y_test, predictions)
        cum_exposure, cum_claims = utils.lorenz_curve(y_test, predictions, X_test['Exposure'])
        results_df['test_gini']=1 - 2 * auc(cum_exposure, cum_claims)
        results_df['test_min_pred']=min(predictions).item()
        results_df['test_max_pred']=max(predictions).item()
        results_df['test_mean_pred']=np.mean(predictions)
        results_df['test_var_pred']=np.var(predictions)

        # Save HPT results
        results_df.to_csv(f'../ag_{ag}/agent_' + str(ag) + '_results.csv')

        # Append dataframe to list
        all_results_list.append(results_df)

        # Save and graph training loss curves
        utils.training_loss_curve(gs.best_estimator_, ag)

    # Combine list of results into datafrane, and then save
    all_results_df = pd.concat(all_results_list)
    all_results_df.to_csv('../results/all_results.csv')
        
if __name__ == "__main__":
          main()