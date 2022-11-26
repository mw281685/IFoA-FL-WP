"""Flower server example."""

from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics


if __name__ == "__main__":

    # Define metric aggregation function
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}



    strategy = fl.server.strategy.FedAvg(
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_fit_clients = 3,
        min_evaluate_clients = 1,
        min_available_clients = 3,
        evaluate_metrics_aggregation_fn=weighted_average)

    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=10), strategy=strategy
      )
