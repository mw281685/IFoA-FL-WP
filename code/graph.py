import utils
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc



MODEL_PATH = '/home/malgorzata/IFoA/FL/code/federated with flower/'
NUM_FEATURES = 39

# Define architecture
class NeuralNetworks(torch.nn.Module):
    # define model elements
    def __init__(self, n_features):
        super(NeuralNetworks, self).__init__()
        self.hid1 = torch.nn.Linear(n_features, 5)
        self.hid2 = torch.nn.Linear(5, 10)
        self.hid3 = torch.nn.Linear(10, 15)
        self.drop = torch.nn.Dropout(0.12409392594394411)
        self.output = torch.nn.Linear(15, 1)

        torch.nn.init.xavier_uniform_(self.hid1.weight)
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight)
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.hid3.weight)
        torch.nn.init.zeros_(self.hid3.bias)
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, X):
        z = torch.relu(self.hid1(X))
        z = torch.relu(self.hid2(z))
        z = torch.relu(self.hid3(z))
        z = self.drop(z)
        z = self.output(z)
        return z

def main():

    for ag in range(3):

        AGENT_PATH =  '../ag_global/global_model.pt'
        model_global = NeuralNetworks(NUM_FEATURES) 
        model_global.load_state_dict(torch.load(AGENT_PATH))
        model_global.eval()

        AGENT_PATH =  f'../ag_{ag}/local_model.pt'
        model_partial = NeuralNetworks(NUM_FEATURES)
        model_partial.load_state_dict(torch.load(AGENT_PATH))
        model_partial.eval()

        AGENT_PATH = f'../ag_{ag}/fl_model.pt'
        model_fl = NeuralNetworks(NUM_FEATURES)
        model_fl.load_state_dict(torch.load(AGENT_PATH))
        model_fl.eval()

        utils.predictions_check('FL 10 rnd; 10 epoch; 3 agents.png', model_global, model_partial, model_fl, ag)


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



