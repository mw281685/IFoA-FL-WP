cd /Users/ms/Documents/IFoA/flw_projects/IFoA-FL-WP

for (( i=0; i < 10; ++i)) do 
    echo "$i"
    python3 'IFoA client  [ Multilayer ] [freMTPL2freq].py' --agent_id=$i --agents_no=10 --if_FL=0
done