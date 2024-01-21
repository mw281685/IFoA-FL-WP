no_clients=$(python3 ./run_config.py)
echo '-----------------------------------------------'
echo "Starting FL training for $no_clients  clients. "
echo '-----------------------------------------------'

# delete individual folders
# delete content of terminal_output folder
echo "Folders cleanup"
python3 'folders_cleanup.py'


echo "START: Dataset preparation "
python3 'prepare_dataset.py'
echo "DONE: Dataset preparation"

pids=()

echo "Starting FL server "
python3 'IFoA_server.py' >../terminal_output/out_server.txt  2>../terminal_output/err_server.txt &
#pids+=($!)
#echo $!

echo "Starting global_model training"
python3 'IFoA_client.py' --agent_id=-1 --if_FL=0 >../terminal_output/out_global.txt 2>../terminal_output/err_global.txt &
#global_process=$! # $! - PID of the last command launched in the background
pids+=($!)
echo $!


for (( i=0; i < $no_clients; ++i)) do 
    echo "Starting client $i"
    python3 'IFoA_client.py' --agent_id=$i > ../terminal_output/out$i.txt 2>../terminal_output/err$i.txt &
    pids+=($!)
    echo $!
done


# wait for all processes in the background to finish
for i in "${pids[@]}"
do
	echo "$i"
    wait $i
done

#wait $! # wait for the last process to end , will work most of the time but I'm aware it needs improvement

#echo "Waiting for all clients processes to finish"
# wait
# echo "DONE: FL TRAINING"

# run the report once all the training is finished:
python3 'report.py' 

exit 0