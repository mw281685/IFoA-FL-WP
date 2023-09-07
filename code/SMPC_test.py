import csv
import numpy as np
import architecture as archit
import torch

noise_file = '/Users/ms/Documents/IFoA/flw_projects/IFoA-FL-WP/data/seeds.csv'


def calc_noise(file, ag_no):
    with open(file, mode='r') as infile:
        reader = csv.reader(infile)
        seeds = {int(rows[0]): rows[1:] for rows in reader}
        print(seeds)
        print(type(seeds))

    noises ={}
    for k,v in seeds.items():
        vect = np.zeros(2)
        for el in v:
            np.random.seed(int(v[ag_no]))
            vect = vect - np.random.random(2)
        np.random.seed(int(seeds[k][ag_no]))
        vect = vect + 2*np.random.random(2)
        noises[k] = vect
    print(noises)


for el in ['layer_1.weight', 'layer_1.bias', 'layer_2.weight', 'layer_2.bias', 'layer_out.weight', 'layer_out.bias']:
    PATH = "/Users/ms/Documents/IFoA/flw_projects/IFoA-FL-WP/_tmp/fl_model_10_smpc.pt"
    model = archit.MultipleRegression(num_features=39, num_units_1=60, num_units_2=20)
    model.load_state_dict(torch.load(PATH))
    #params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    params = model.state_dict()[el]

    print('-------------------------------------------------------------------')
    PATH = "/Users/ms/Documents/IFoA/flw_projects/IFoA-FL-WP/_tmp/fl_model_10_vanilla.pt"
    model.load_state_dict(torch.load(PATH))
    #params2 = [val.cpu().numpy() for _, val in model.state_dict().items()]
    params2 = model.state_dict()[el]


    comparison = params == params2
    equal_arrays = comparison.all()

    print(equal_arrays)




#calc_noise(noise_file, 0)