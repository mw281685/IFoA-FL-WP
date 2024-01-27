
from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics
import run_config
import architecture as archit
import torch

if __name__ == "__main__":

    # Define metric aggregation function
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics] 
        examples = [num_examples for num_examples, _ in metrics]

        return {"accuracy": sum(accuracies) / sum(examples)} 

    def init_parameters():
        PATH = "./init_state_dict.pt"
        model = archit.MultipleRegression(num_features=run_config.NUM_FEATURES, num_units_1=run_config.NUM_UNITS_1, num_units_2=run_config.NUM_UNITS_2)

        # Initialize the state dictionary with zeros
        #state_dict = model.state_dict()
        #for key in state_dict:
        #    state_dict[key].zero_()

        #model.load_state_dict(torch.load(PATH))
        params = archit.get_parameters(model)  # Xavier initialization
        
        return fl.common.ndarrays_to_parameters(params)

    strategy = run_config.LocalUpdatesStrategy(
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_fit_clients = run_config.server_config["num_clients"],
        min_evaluate_clients = 1,
        min_available_clients = run_config.server_config["num_clients"],
        evaluate_metrics_aggregation_fn=weighted_average,   # used to define stoping criterion when training a FL model.
        initial_parameters = init_parameters(),
    )


    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=run_config.server_config["num_rounds"]), strategy=strategy
      )

exit(0)
