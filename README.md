# IFoA-FL-WP

---------------------------------------------------------
DEPENDENCY MANAGEMENT - FLOWER INSTALLATION:
-----------------------------------------------------------

We recommend Poetry to install those dependencies and manage your virtual environment (Poetry installation: https://python-poetry.org/docs/#installing-with-the-official-installer ), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

poetry install
poetry shell
Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

python3 -c "import flwr"
If you don't see any errors you're good to go!


-----------------------------------------------------------
PROJECT STRUCTURE - TO DO 
-----------------------------------------------------------
IFoA Use case [Privacy preserving ML collaboration on claims modelling]. We use Flower.dev framework to federate workflow. 

1. pip install flwr
2. start server: python3 IFoA_server.py     [please see the configuration of FL run in IFoAserver.py ; later on we will separate configuration from server code]
3. start FL training participants, --partition=i i in range(3) identifies participant's data chunk eg. to start client0 we call: python3 'IFoA_client.py' --agent_id=0
4. Individual data are stored in data folder
5. results are stored in dedicated folders ag_0 .... ag_3 . Resulting models:
    fl_model.pt -> model trained using FL pipeline
    local_model.pl - > model trained on single participant's dataset


-----------------------------------------------------------
STARTING FL TRAINING :
-------------------------------------------------------------

1. Setup config in run_config.py

Mac users: 
2. run ./START_TRAINING.sh
if this doesn't work you can start clients manually using
python3 'IFoA_client.py' --agent_id=3 >../terminal_output/out_3.txt 2>../terminal_output/err_3.txt


(>../terminal_output/out_3.txt 2>../terminal_output/err_3.txt) --> this bit of code redirects results to .txt files .   You need to create a folder named terminal_output on the same level as code folder 



Windows users: follow 2-4 or test if .sh script works on Windows too :) 
2. Prepare datasets for each individual agent with  
    1. python3 prepare_dataset.py
3. Train FL model by:
    1. Starting FL server with python3 IFoA_server.py
    2. Adding server_config[‘num_clients’] ( specified in run_config.py) training agents with python3 IFoA_client.py  —agent_id=0  
4. Train Global model by running python3 IFoA_client.py —agent_id=-1


All operating systems : 

Generating report:
5. Run python3 report.py



# IFoA-FL-WP

# Run Instructions:

1) **Create 10 uniform data splits**

**RUN**: `python uniform_prepare_dataset.py --agents=10`

This creates 10 uniformly split training and validation files (including 1 Test set)

Optional step: Check if split has been executed correctly:

**RUN**: `python check_dataset.py --agents=10`

2) **Tune 1 global model and 10 local models**

**RUN**: `python skorch_tuning.py`

This should output and save all models in .pkl as well as training loss curves, and results of hyperparameter tuning

**NOTE** You will need folders for ag_-1, ag_0,...,ag_9 in your repo!!!

3) **Get best hyperparamter results (for federated model)**

Take hyperparameters from 1st row of 'local_average_HPT_performance.csv' in `/results` (which should be automatically output from step above)

Note the format of the `params_key` is `learning_rate_layer1neurons_layer2neurons_bath_size` i.e. 3 tuned hyperparameters. Also note on my machine this is currently `0.001_60_20_100` (could change if run on different machine with different seed, OS etc. but should be reproducible on each machine)

4) **Train federated model**

**RUN**: ???

*TODO*: Remember to use average of 10 local validation losses for early stopping i.e. tune how long to train the federated model for

**NOTE** The `skorch_tuning.py` script has the neural net architecture e.g. activation function etc. you should use for the federated model

5) **Output model performance**

**RUN**: `federated_learing_notebook_skorch n_jobs_ps_v8.4_graph.ipynb`

Note the above notebook only currently reads in global and local models i.e. the federated model will need to be added





-------------------------------------------------------------
DEPRECATED: 
-------------------------------------------------------------



TEST RUN 3 PARTICIPANTS:
1. Dylan: python3 'IFoA client  [ Multilayer ] [freMTPL2freq].py' --agent_id=0
2. Ben: python3 'IFoA client  [ Multilayer ] [freMTPL2freq].py' --agent_id=1
3. Malgorzata python3 'IFoA client  [ Multilayer ] [freMTPL2freq].py' --agent_id=2


EDA: 

1. Dylan: python3 EDA.py --agent=0
2. Ben: python3 EDA.py --agent=1
3. Malgorzata: python3 EDA.py --agent=2

PREDICTIONS:
1. Dylan : python3 evaluate.py --agent=0
2. Ben : python3 evaluate.py --agent=1
3. Malgorzata :  python3 evaluate.py --agent=2





