
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
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy
from flwr.common.typing import Status
from flwr.common import NDArray, NDArrays

if __name__ == "__main__":

    # Define metric aggregation function
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        

        # debug weighted average:
        for num, m in metrics:
                  print('num: ', num, 'm: ', m)

        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics] # Multiply accuracy of each client by number of examples used
        examples = [num_examples for num_examples, _ in metrics]

        return {"accuracy": sum(accuracies) / sum(examples)} # Aggregate and return custom metric (weighted average)



    class FedAvgIFoA(fl.server.strategy.FedAvg):

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

            """Aggregate fit results using weighted average."""
            if not results:
                return None, {}
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}


            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics['exposure'])
                for _, fit_res in results
            ]


            parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")

            return parameters_aggregated, metrics_aggregated


    strategy = FedAvgIFoA(
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_fit_clients = 10,
        min_evaluate_clients = 1,
        min_available_clients = 10,
        evaluate_metrics_aggregation_fn=weighted_average
    )

    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=10), strategy=strategy
      )
