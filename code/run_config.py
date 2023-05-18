
from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from flwr.server.strategy.strategy import Strategy
from flwr.common.typing import Status
from flwr.common import NDArray, NDArrays

EPOCHS_LOCAL_GLOBAL = 75


# OPTUNA ARCHITECTURE
model_architecture = {
    "dropout" : 0, #0.12409392594394411, # remove
    "learning_rate": 0.001, #6.888528294546944e-05, #  not needed for Adam optimizer
    "epochs" : 30,
    "batch_size": 100,
    "num_features": 39,
}

server_config = {
    "num_clients": 10,
    "num_rounds": 10
}

run_name = "uniform partitions, " + str(server_config["num_clients"]) + " agents," + str(server_config["num_rounds"]) + " rounds, " + str(model_architecture["epochs"]) + " epochs " + str(EPOCHS_LOCAL_GLOBAL) + " epochs for local and global tr"

# used in prepare_dataset.py 
dataset_config = {
    "path" : '../data/freMTPL2freq.csv',
    "seed" : 300,
    "num_features": 39,
    "num_agents" : server_config["num_clients"],
}

class LocalUpdatesStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        """Aggregate fit results using weighted average of claims exposure."""
        if not results:
            return None, {}
    # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

    # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics['exposure']) for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


def get_clients_no():
    print(server_config["num_clients"])

if __name__ == "__main__":
    get_clients_no()
    
