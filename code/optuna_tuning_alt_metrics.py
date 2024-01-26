import torch as th
import torch.nn.functional as F
import torch.nn as nn 
import optuna
import argparse
from torch import optim
from optuna.trial import TrialState
from optuna.samplers import RandomSampler, TPESampler
from torch.utils.data import TensorDataset, DataLoader
import utils
from sklearn.metrics import auc, mean_poisson_deviance, d2_tweedie_score
import numpy as np
import run_config

DATA_PATH = run_config.dataset_config["path"]
SEED = run_config.dataset_config["seed"]


NUM_FEATURES = 39
device = 'cpu'
EPOCHS = 40 #10 to test also 50 ! 
BATCH_SIZE = 10_000
LOG_INTERVAL = 10
# Set loss function change to true and then exp the output
criterion = nn.PoissonNLLLoss(log_input= True, full= True)

def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = NUM_FEATURES
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 10, 80)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        #p = trial.suggest_float("dropout_{}".format(i), 0.1, 0.2)
        #layers.append(nn.Dropout(p))

        in_features = out_features

    layers.append(nn.Linear(in_features, 1))

    return nn.Sequential(*layers)



def objective(trial, train_loader, val_loader, val_dataset):

    utils.seed_torch(seed=SEED)

    # Generate the model. Transfer to device
    model = define_model(trial).to(device)

    # Generate the optimizers.
    #optimizer_name = trial.suggest_categorical("optimizer", ["NAdam"])
    #lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    #optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    optimizer = optim.NAdam(model.parameters())

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            
            y_train_pred = model(X_train_batch)
            
            train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
            
            train_loss.backward()
            optimizer.step()
            
            # train_epoch_loss += train_loss.item()

        # Validation of the model.
        with th.no_grad():
            
            val_epoch_loss = 0
            
            model.eval()
            #correct = 0
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                
                y_val_pred = model(X_val_batch)
                            
                val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))
                
                val_epoch_loss += val_loss.item()

        y_pred_list_exp = utils.exp_model_predictions(model, val_loader)

        y_pred_null = np.full(val_dataset.tensors[1].numpy().shape, np.mean(val_dataset.tensors[1].numpy()))

        null_mpd = mean_poisson_deviance(val_dataset.tensors[1].numpy(), y_pred_null, sample_weight=val_dataset.tensors[0].numpy()[:, 0])

        cum_exposure, cum_claims = utils.lorenz_curve(val_dataset.tensors[1].numpy(), y_pred_list_exp, val_dataset.tensors[0].numpy()[:, 0])

        accuracy = d2_tweedie_score(val_dataset.tensors[1].numpy(), y_pred_list_exp, sample_weight=val_dataset.tensors[0].numpy()[:, 0], power=1)
        #accuracy = mean_poisson_deviance(val_dataset.tensors[1].numpy(), y_pred_list_exp, sample_weight=val_dataset.tensors[0].numpy()[:, 0])
        #accuracy = null_mpd
        #accuracy = 1 - 2 * auc(cum_exposure, cum_claims)
        # this works
        #accuracy = val_epoch_loss / len(val_loader.dataset)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def run_study(train_loader, val_loader, val_dataset):
    if __name__ == "__main__":
        sampler = TPESampler(seed=SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda trial: objective(trial, train_loader, val_loader, val_dataset), n_trials=20, timeout=60*60*12)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
        print("Best trial:")
        
        trial = study.best_trial
        
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        
    #store hyperparameter dictionary
    return trial.params


def main():

#    parser = argparse.ArgumentParser(description="Flower")
#    parser.add_argument(
#        "--agent_id",
#        type=int,
#        default=-1,
#        choices=range(-1, 10),
#        required=False,
#        help="Specifies the partition of data to be used for training. -1 means all data . \
#        Picks partition 0 by default",
#    )
    
#    args = parser.parse_args()

    #lr= dict()
    #dropout = dict()

    for ag in range(-1,3):
        print(f'processing agent = {ag}')
        train_dataset, val_dataset, test_dataset, train_column_names, X_test_sc, exposure = utils.load_individual_data(ag)  # in folder my_data each training participant is storing their private, unique dataset    
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1)

        global_params = run_study(train_loader, val_loader, val_dataset)
        #lr[ag] = global_params["lr"]
        #dropout[ag] = global_params["dropout_0"]

    #for el in lr:
        #print(f'learning rate[{el}] = {lr[el]}')

    #for el in dropout:
        #print(f'dropout rate[{el}] = {dropout[el]}')

if __name__ == "__main__":
          main()