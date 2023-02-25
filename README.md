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





