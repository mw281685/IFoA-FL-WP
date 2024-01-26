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
import architecture as archit

DATA_PATH = run_config.dataset_config["path"]
SEED = run_config.dataset_config["seed"]
BATCH_SIZE = 5_00
 
# Formatting options to print dataframe to terminal
pd.set_option('display.max_columns', 7)
pd.set_option('display.width', 200)

utils.seed_torch()

rng = np.random.RandomState(SEED) 

# Create PDE score to print during training/tuning
pde_score = make_scorer(d2_tweedie_score, power=1)
# Create callback to display PDE after each epoch on validation set
pde_callback = callbacks.EpochScoring(pde_score, lower_is_better=False, name='PDE')
# Create callback to do early stopping if PDE doesn't improve for 5 epochs on validation PDE
early_stopping_callback = callbacks.EarlyStopping(monitor='weighted_PDE', lower_is_better=False, patience=5)
# Create callback to save and reload highest PDE on validation set
check_point_callback = callbacks.Checkpoint(monitor='weighted_PDE_best', load_best=True)

# Grid Search space dictionary
params = {
    'optimizer__lr': [0.001, 0.01], # 3
    #'optimizer__momentum': [0.9],
    #'batch_size':[500, 5_000, 50_000], # 3
    'module__num_units_1': [5, 10, 15],# 3
    'module__num_units_2': [5, 10, 15], # 3
    #'module__num_units_3': [10, 20, 30, 40], # 3
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

        valid_ds = Dataset(X_val.to_numpy().astype(np.float32), y_val.to_numpy().astype(np.float32))

        def weighted_pde(ground_truth, predictions, sample_val_weight=X_val['Exposure'], sample_train_weight=X_train['Exposure']):

            ground_truth_len = ground_truth.shape[0]
            
            if ground_truth_len == sample_train_weight.shape[0]:
                chosen_sample_weight = sample_train_weight
            else:
                chosen_sample_weight = sample_val_weight

            d2_tweedie_score_1 = d2_tweedie_score(ground_truth, predictions, sample_weight=chosen_sample_weight, power=1)

            return d2_tweedie_score_1.item()
        
        # Define weighted PDE scorer
        #weighted_pde_score = make_scorer(d2_tweedie_score, sample_weight=X_val['Exposure'], power=1, greater_is_better=True)
        weighted_pde_score = make_scorer(weighted_pde, greater_is_better=True)

        # Create callback to display weighted PDE after each epoch on validation set
        weighted_pde_callback = callbacks.EpochScoring(weighted_pde_score, lower_is_better=False, name='weighted_PDE')

        # Define skorch neural network
        net_regr = NeuralNetRegressor(
            archit.MultipleRegression(num_features=39, num_units_1=20, num_units_2=40).double(),
            #optimizer=optim.SGD,
            optimizer=optim.NAdam,
            criterion=nn.PoissonNLLLoss(log_input= False, full= True),
            max_epochs=100,
            batch_size=BATCH_SIZE,
            #train_split=skorch.dataset.ValidSplit(0.1, stratified=False, random_state=SEED), # use 10% to set early stopping
            train_split = predefined_split(valid_ds),
            #train_split=None,
            #callbacks=[pde_callback, early_stopping_callback, check_point_callback],
            callbacks=[weighted_pde_callback],
            device=None, # ignore CUDA for now
            #device='cuda', # ignore CUDA for now
            iterator_train__shuffle=True
            )
                
        gs = RandomizedSearchCV(net_regr,
                        params,
                        refit=True,
                        cv=ps,
                        scoring=weighted_pde_score,
                        n_iter=18, # grid size
                        #n_jobs=-1, # turning off mutli-threading as issues with reproducibility
                        random_state=SEED,
                        return_train_score=True
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
        
        print(results_df[["params", "rank_test_score", "mean_train_score", "mean_test_score", "std_test_score", "mean_fit_time"]])

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

    # Select local top 5 hyperparameters
    top_5_results_df = all_results_df[(all_results_df['agent']!=-1) & (all_results_df['rank_test_score']<=5)]

    # Output learning rate distribution
    #utils.hyperparameter_counts(top_5_results_df, hyperparameter='param_optimizer__lr', x_label='Learning Rate', title='Best Local Learning Rates', name='learning_rate_chart')

    # Output layer 1 neurons distribution
    utils.hyperparameter_counts(top_5_results_df, hyperparameter='param_module__num_units_1', x_label='Layer 1 Neurons', title='Best Local Layer 1 Neurons', name='layer_1_neuron_chart')

    # Output layer 2 neurons distribution
    utils.hyperparameter_counts(top_5_results_df, hyperparameter='param_module__num_units_2', x_label='Layer 2 Neurons', title='Best Local Layer 2 Neurons', name='layer_2_neuron_chart')

    # Output best epochs
    utils.hyperparameter_counts(top_5_results_df, hyperparameter='best_epochs', x_label='Epochs', title='Best Local Epochs', name='epochs_chart')

    # Overal HPT set
    top_5_results_df.index.value_counts().plot(kind='bar',figsize=(10,12))
    plt.grid()
    plt.xlabel('HPT set')
    plt.ylabel('Count')
    plt.title('Best HPT Sets')
    plt.savefig('../results/HPT_chart', facecolor='white')

    # Randomly select 1 of the Top 5
    best_HPT_randomly_chosen_df = top_5_results_df.sample(random_state=SEED)
    best_HPT_randomly_chosen_df.to_csv('../results/best_HPT_randomly_chosen.csv')

    # Alternative HPT method
    # Filter to just local HPT results
    all_local_results_df = all_results_df[all_results_df['agent']!=-1]
    # Compute average performance by each HPT set across all agents
    local_average_results_df = all_local_results_df[['mean_test_score','best_epochs']].groupby(level=0).mean()
    # Rank and sort data
    local_average_results_df['mean_test_score_rank'] = local_average_results_df['mean_test_score'].rank(ascending=False)
    local_average_results_df=local_average_results_df.sort_values(by='mean_test_score_rank', ascending=True)
    # Save result
    local_average_results_df.to_csv('../results/local_average_HPT_performance.csv')

if __name__ == "__main__":
          main()