#!/bin/bash

# Open a new Terminal window and activate environment
osascript -e 'tell application "Terminal" to do script "cd /Users/ms/Documents/IFoA/flw_projects/branch_testing/local_branch/IFoA-FL-WP/code && source /Users/ms/Library/Caches/pypoetry/virtualenvs/flw-3agentsdemo-O3axsXRA-py3.8/bin/activate && python3 IFoA_server.py" '

no_clients=10

for (( i=0; i<no_clients; ++i)) do 
    osascript -e 'tell application "Terminal" to do script "cd /Users/ms/Documents/IFoA/flw_projects/branch_testing/local_branch/IFoA-FL-WP/code && source /Users/ms/Library/Caches/pypoetry/virtualenvs/flw-3agentsdemo-O3axsXRA-py3.8/bin/activate && python3 IFoA_client.py --agent_id='$i' " '

done

exit 0

