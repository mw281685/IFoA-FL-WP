from sklearn.metrics import auc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import utils
import torch
import architecture as archit


train_dataset, val_dataset, test_dataset, X_train_sc, y_tr, X_val_sc, y_vl, X_test_sc, y_te = utils.load_individual_data(1) 
NUM_FEATURES = 39

# Federated Model
fl_model = archit.NeuralNetworks(NUM_FEATURES)

model_name = 'fl_model.pt'
AGENT_PATH = '../ag_1/' + model_name 

fl_model.load_state_dict(torch.load(AGENT_PATH))
fl_model.eval()


# Local model
local_model = archit.NeuralNetworks(NUM_FEATURES)

model_name = 'local_model.pt'
AGENT_PATH = '../ag_1/' + model_name 

local_model.load_state_dict(torch.load(AGENT_PATH))
local_model.eval()


# Lorenz Curves

def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_true = y_true.reshape((len(y_true), ))
    y_pred = y_pred.reshape((len(y_pred), ))
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_frequencies = y_true[ranking]
    ranked_exposure = exposure[ranking]
    cumulated_claims = np.cumsum(ranked_frequencies * ranked_exposure)
    cumulated_claims /= cumulated_claims[-1]
    cumulated_exposure = np.cumsum(ranked_exposure)
    cumulated_exposure /= cumulated_exposure[-1]
    
    return cumulated_exposure, cumulated_claims


fig, ax = plt.subplots(figsize=(8, 8))

for model in [fl_model, local_model]:
    log_y_pred = []
    with torch.no_grad():
        model.eval()
        log_y_pred = model(torch.tensor(X_test_sc.values).float())
    log_y_pred = [a.squeeze().tolist() for a in log_y_pred]
    y_pred = np.exp(log_y_pred)
    cum_exposure, cum_claims = lorenz_curve(
        y_te, y_pred, X_test_sc["Exposure"]
    )
    gini = 1 - 2 * auc(cum_exposure, cum_claims)
    label = "(Gini: {:.2f})".format(gini)
    ax.plot(cum_exposure, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
cum_exposure, cum_claims = lorenz_curve(
    y_te, y_te, X_test_sc["Exposure"]
)
gini = 1 - 2 * auc(cum_exposure, cum_claims)
label = "Oracle (Gini: {:.2f})".format(gini)
ax.plot(cum_exposure, cum_claims, linestyle="-.", color="gray", label=label)

# Random Baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz curves by model",
    xlabel="Cumulative proportion of exposure (from safest to riskiest)",
    ylabel="Cumulative proportion of claims",
)
ax.legend(loc="upper left")

print('done')