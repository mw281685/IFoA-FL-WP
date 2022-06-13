import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--agent",
    type=int,
    default=10,
    choices=range(0, 11),
    required=False,
    help="Specifies the number of agents. \
    Picks 10 by default",
)

args = parser.parse_args()

df = pd.read_csv('./data/y_tr_' + str(args.agent) +'.csv')

no_obj_cols = [var for var in df.columns if df[var].dtype!='object']

print(df[no_obj_cols].describe().T)