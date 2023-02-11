import utils
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
import architecture
import run_config


NUM_FEATURES = run_config.model_architecture["num_features"]
NUM_AGENTS = run_config.server_config["num_clients"]



def main():

    for ag in range(NUM_AGENTS):

        AGENT_PATH =  '../ag_global/global_model.pt'
        model_global = architecture.NeuralNetworks(NUM_FEATURES) 
        model_global.load_state_dict(torch.load(AGENT_PATH))
        model_global.eval()

        AGENT_PATH =  f'../ag_{ag}/local_model.pt'
        model_partial = architecture.NeuralNetworks(NUM_FEATURES)
        model_partial.load_state_dict(torch.load(AGENT_PATH))
        model_partial.eval()

        AGENT_PATH = f'../ag_{ag}/fl_model.pt'
        model_fl = architecture.NeuralNetworks(NUM_FEATURES)
        model_fl.load_state_dict(torch.load(AGENT_PATH))
        model_fl.eval()

        utils.predictions_check( 'Agent ' + ag + 'FL ' + str(run_config.server_config["num_rounds"]) + ' rnd'  + str(run_config.model_architecture["epochs"]) + ' epoch ' + str(run_config.server_config["num_clients"]) + ' agents.png', model_global, model_partial, model_fl, ag)


        train_dataset, val_dataset, test_dataset, X_train_sc, y_tr, X_val_sc, y_vl, X_test_sc, y_te = utils.load_individual_data_lift(ag)
        
        fig, ax = plt.subplots(figsize=(8, 8))

        for model in [model_fl, model_partial]:
            log_y_pred = []
            with torch.no_grad():
                model.eval()
                log_y_pred = model(torch.tensor(X_test_sc.values).float())
            log_y_pred = [a.squeeze().tolist() for a in log_y_pred]
            y_pred = np.exp(log_y_pred)
            cum_exposure, cum_claims = utils.lorenz_curve(
                y_te, y_pred, X_test_sc["Exposure"]
            )
            gini = 1 - 2 * auc(cum_exposure, cum_claims)
            label = "(Gini: {:.2f})".format(gini)
            ax.plot(cum_exposure, cum_claims, linestyle="-", label=label)

        # Oracle model: y_pred == y_test
        cum_exposure, cum_claims = utils.lorenz_curve(
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
        plt.savefig(f'../ag_{ag}/' + 'lift_chart')

        print('done')

    return

if __name__ == "__main__":
      main()



