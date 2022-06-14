"""Flower server example."""


import flwr as fl

if __name__ == "__main__":

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_eval=0.0,
        min_fit_clients=3,
        min_eval_clients=1,
        min_available_clients=3,
 #       eval_fn=get_eval_fn(model, args.toy),
 #       on_fit_config_fn=fit_config,
 #       on_evaluate_config_fn=evaluate_config,
 #       initial_parameters=fl.common.weights_to_parameters(model_weights),
    )



    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": 10}, strategy=strategy
      )
