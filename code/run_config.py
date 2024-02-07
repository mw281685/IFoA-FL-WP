
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
import numpy as np
from utils_quantisation import quantize, dequantize, dequantize_mean, modulus, add_mod, N, M, CR
from functools import reduce
import pandas as pd



tuning_results_df = pd.read_csv('../results/all_results.csv')
top_results_df = tuning_results_df.loc[tuning_results_df['rank_test_score']==1]
top_results_dict = top_results_df[['agent', 'param_module__num_units_1', 'param_module__num_units_2']].set_index('agent').to_dict('index')

IF_TRAIN_VAL = 1 # 1: validation dataset included in training
QUANTISATION = 1
SMPC_NOISE = 0  
EPOCHS_LOCAL_GLOBAL = 10 
EPOCHS = 10
BATCH_SIZE = 500
NUM_FEATURES = 39
NUM_UNITS_1 = 10 #list(top_results_dict[-1].items())[0][1]
NUM_UNITS_2 = 15 #list(top_results_dict[-1].items())[1][1]





server_config = {
    "num_clients": 10,
    "num_rounds": 250
}


#run_name = "uniform partitions, " + str(server_config["num_clients"]) + " agents," + str(server_config["num_rounds"]) + " rounds, " + str(EPOCHS) + " epochs " + str(EPOCHS_LOCAL_GLOBAL) + " epochs for local and global tr"
run_name = 'UP_'+ str(server_config["num_clients"])+'ag_'+ str(server_config["num_rounds"])+'rnd_'+ str(EPOCHS)+'ep_'+ str(QUANTISATION) +'qt_'+ str(SMPC_NOISE) +'SMPCn'
# used in prepare_dataset.py 
dataset_config = {
    "path" : '../data/freMTPL2freq.csv',
    "seed" : 300,
    "num_agents" : server_config["num_clients"],
    "num_features": 39.
}


def aggregate2(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])
    print('num examples total:')
    print(num_examples_total)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer.astype(np.float64) * np.float64(num_examples) for layer in weights] for weights, num_examples in results
    ]

    print('agent0: ')
    print(weighted_weights[0][0])
    print('agent1:')
    print(weighted_weights[1][0])

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        (reduce(np.add, layer_updates) / np.float64(num_examples_total)).astype(np.float64)
        for layer_updates in zip(*weighted_weights)
    ]

    print('weights_prime')
    print(weights_prime)

    return weights_prime


def aggregate_qt(results: List[Tuple[NDArrays, np.int64]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])
    print('num examples total:')
    print(num_examples_total)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    print('agent0: ')
    print(weighted_weights[0][0])
    print('agent1:')
    print(weighted_weights[1][0])
     
    # Compute average weights of each layer
    weights_prime: NDArrays = [
        (reduce(add_mod, layer_updates))
        for layer_updates in zip(*weighted_weights)
    ]

    #dequantize:
    print('num_examples_total', num_examples_total)
    weights_deq = dequantize(weights_prime, CR, N, num_examples_total)
    weights_deq_weigh = np.divide(weights_deq, num_examples_total)

    print('weights_deq_weigh', weights_deq_weigh)

    return weights_deq_weigh

def aggregate_qt2(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])
    print('num examples total:')
    print(num_examples_total)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
    print('type(weighted_weitgts)', type(weighted_weights))
    print('type(weighted_weitgts[0])', type(weighted_weights[0]))
    print('type(weighted_weitgts[0][0])', type(weighted_weights[0][0]))

    print('agent0: ')
    print(weighted_weights[0][0])
    print('agent1:')
    print(weighted_weights[1][0])
     
    # Compute average weights of each layer
    weights_prime: NDArrays = [
        (reduce(add_mod, layer_updates))
        for layer_updates in zip(*weighted_weights)
    ]

    #dequantize:
    print('num_examples_total', num_examples_total)
    weights_deq_weigh = dequantize_mean(weights_prime, CR, N, num_examples_total)
   
    print('weights_deq_weigh', weights_deq_weigh)

    return weights_deq_weigh


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
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results # ms test 10.09.2023 prev: (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics['exposure']) for _, fit_res in results
        ]

        if QUANTISATION:
            parameters_aggregated = ndarrays_to_parameters(aggregate_qt(weights_results))
        else:
            parameters_aggregated = ndarrays_to_parameters(aggregate2(weights_results))
            
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
    
