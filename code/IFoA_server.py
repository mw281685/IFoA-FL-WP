
from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics
import run_config
import utils


if __name__ == "__main__":


    # Define metric aggregation function
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:

        # debug weighted average:
        for num, m in metrics:
            print('num: ', num, 'm: ', m)

        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics] # Multiply accuracy of each client by number of examples used
        examples = [num_examples for num_examples, _ in metrics]

        return {"accuracy": sum(accuracies) / sum(examples)} # Aggregate and return custom metric (weighted average)


    strategy = run_config.LocalUpdatesStrategy(
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_fit_clients = run_config.server_config["num_clients"],
        min_evaluate_clients = 1,
        min_available_clients = run_config.server_config["num_clients"],
        evaluate_metrics_aggregation_fn=weighted_average   # might be useful to define stoping criterion when training a FL model. Not used and analysed for now.
    )


    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=run_config.server_config["num_rounds"]), strategy=strategy
      )
