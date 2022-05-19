# IFoA-FL-WP

IFoA Use case [Privacy preserving ML collaboration on claims modelling]. We use Flower.dev framework to federate workflow. 

1. pip install flwr
2. start server: python3 IFoAserver.py     [please see the configuration of FL run in IFoAserver.py ; later on we will separate configuration from server code]
3. start FL training participants, --partition=i i in range(9) identifies participant's data chunk eg. to start client0 we call: python3 'IFoA client  [ Multilayer ] [freMTPL2freq].py' --partition=0
4. results are stored in dedicated folders ag_0 .... ag_9


Execution:
1. Global model training ( no FL loop ):
python3 'IFoA client  [ Multilayer ] [freMTPL2freq].py' --partition=-1 --if_FL=0
2. Partial model training ( assuming 10 participants) . Model training for participant = i  i in range(0,9):


3. FL training ( assuming 10 participants). 


a) Start FL server (make sure the ip adress used in the code is correct):


b) Start participant == 0 with a call (make sure you use right ip adress and port when connecting to the server):


( you need to start 10 such processes representing participant==i i in range(10) ) 






