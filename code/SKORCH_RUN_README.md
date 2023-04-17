# IFoA-FL-WP

Run Instructions:

1) Create 10 uniform data splits

RUN: python uniform_prepare_dataset.py --agents=10

This creates 10 uniformly split training and validation files (including 1 Test set)

Optional step: Check if split has been executed correctly:

python check_dataset.py --agents=10

2) Tune 1 global model and 10 local models

RUN: python skorch_tuning.py

This should output and save all models in .pkl as well as training loss curves, and results of hyperparameter tuning

!!NOTE!! You will need folders for ag_-1, ag_0,...,ag_9 in your repo!!!

3) Get best hyperparamter results (for federated model)

Take hyperparameters from 1st row of 'local_average_HPT_performance.csv' in `/results` (which should be automatically output from step above)

Note the format of the `params_key` is `learning_rate_layer1neurons_layer2neurons_bath_size` i.e. 3 tuned hyperparameters. Also note on my machine this is currently `0.001_60_20_100` (could change if run on different machine with different seed, OS etc. but should be reproducible on each machine)

4) Train federated model

RUN: ???

TODO: Remember to use average of 10 local validation losses for early stopping i.e. tune how long to train the federated model for

!!NOTE!! The skorch_tuning.py script has the neural net architecture e.g. activation function etc. you should use for the federated model

5) Output model performance

RUN: federated_learing_notebook_skorch n_jobs_ps_v8.4_graph.ipynb

Note the above notebook only currently reads in global and local models i.e. the federated model will need to be added



