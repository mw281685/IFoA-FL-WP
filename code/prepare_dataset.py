import utils
import argparse
import run_config


#def main():       
#    parser = argparse.ArgumentParser(description="Flower")
#    parser.add_argument(
#        "--agents",
#        type=int,
#        default=10,
#        choices=range(1, 11),
#        required=False,
#        help="Specifies the number of agents. \
#        Picks 10 by default",
#    )

#    args = parser.parse_args()
#    print(f'Arg.agents = {args.agents}')
#    utils.prep_partitions(int(args.agents))  # args.partition



if __name__ == "__main__":
    utils.prep_partitions(int(run_config.dataset_config["num_agents"]))

