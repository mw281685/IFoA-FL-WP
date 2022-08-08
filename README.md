# IFoA-FL-WP

IFoA Use case [Privacy preserving ML collaboration on claims modelling]. We use Flower.dev framework to federate workflow. 

1. pip install flwr
2. start server: python3 IFoAserver.py     [please see the configuration of FL run in IFoAserver.py ; later on we will separate configuration from server code]
3. start FL training participants, --partition=i i in range(3) identifies participant's data chunk eg. to start client0 we call: python3 'IFoA client  [ Multilayer ] [freMTPL2freq].py' --agent_id=0
4. Individual data are stored in data folder
5. results are stored in dedicated folders ag_0 .... ag_3 . Resulting models:
    fl_model.pt -> model trained using FL pipeline
    local_model.pl - > model trained on single participant's dataset


Execution:
1. Global model training ( no FL loop ):
python3 'IFoA client  [ Multilayer ] [freMTPL2freq].py' --agent_id=0 --if_FL=0

2. FL training ( assuming 10 participants). 


a) Start FL server (make sure the ip adress used in the code is correct):


b) Start participant == 0 with a call (make sure you use right ip adress and port when connecting to the server):
python3 'IFoA client  [ Multilayer ] [freMTPL2freq].py' --agent_id=0

( you need to start 3 such processes representing participant==i i in range(3) ) 



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


SCRIPTS:
./code/prepare_dataset.py - prepares dataset for 10 agents. Split function defined in prep_partitions function in utils.py 
./code/optuna_tuning.py - runs a study to define lr and dropout rate . 

SEED = 300
learning rate= 0.013433393353340668
dropout rate= 0.13690812525293783
