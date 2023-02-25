
from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics
import run_config


if __name__ == "__main__":

    # Define metric aggregation function
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:

        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics] 
        examples = [num_examples for num_examples, _ in metrics]

        return {"accuracy": sum(accuracies) / sum(examples)} 

    strategy = run_config.LocalUpdatesStrategy(
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_fit_clients = run_config.server_config["num_clients"],
        min_evaluate_clients = 1,
        min_available_clients = run_config.server_config["num_clients"],
        evaluate_metrics_aggregation_fn=weighted_average   # used to define stoping criterion when training a FL model.
    )


    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=run_config.server_config["num_rounds"]), strategy=strategy
      )

exit(0)