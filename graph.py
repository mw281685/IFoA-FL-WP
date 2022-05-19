import utils
import torch


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

      AGENT_PATH =  './ag_global/global_model.pt'
      model_global = NeuralNetworks(NUM_FEATURES) 
      model_global.load_state_dict(torch.load(AGENT_PATH))
      model_global.eval()

      AGENT_PATH =  './ag_3/partial_model.pt'
      model_partial = NeuralNetworks(NUM_FEATURES)
      model_partial.load_state_dict(torch.load(AGENT_PATH))
      model_partial.eval()

      AGENT_PATH = './ag_3/fl_model.pt'
      model_fl = NeuralNetworks(NUM_FEATURES)
      model_fl.load_state_dict(torch.load(AGENT_PATH))
      model_fl.eval()

      utils.predictions_check('FL 10 rnd; 10 epoch; 3 agents.png', model_global, model_partial, model_fl)

      return

if __name__ == "__main__":
      main()



