# IFoA-FL-WP

IFoA Use case [Privacy preserving ML collaboration on claims modelling]. We use Flower.dev framework to federate workflow. 

1. pip install flwr
2. start server: python3 IFoAserver.py     [please see the configuration of FL run in IFoAserver.py ; later on we will separate configuration from server code]
3. start FL training participants, --partition=i i in range(9) identifies participant's data chunk eg. to start client0 we call: python3 'IFoA client  [ Multilayer ] [freMTPL2freq].py' --partition=0
4. results are stored in dedicated folders ag_0 .... ag_9


TO DO:
1. separate global model run
2. Separate partion model run



