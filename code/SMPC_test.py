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
    PATH = "/Users/ms/Documents/IFoA/flw_projects/IFoA-FL-WP/_tmp/fl_model_excel_no_noise.pt"
    model = archit.MultipleRegression(num_features=39, num_units_1=60, num_units_2=20)
    model.load_state_dict(torch.load(PATH))
    #params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    #params = torch.round(model.state_dict()[el],decimals=4)
    #print(params.size())
    params = model.state_dict()[el]

    print('-------------------------------------------------------------------')
    PATH_2 = "/Users/ms/Documents/IFoA/flw_projects/IFoA-FL-WP/_tmp/fl_model_seed_smpc.pt"
    model_2 = archit.MultipleRegression(num_features=39, num_units_1=60, num_units_2=20)
    model_2.load_state_dict(torch.load(PATH_2))
    #params2 = [val.cpu().numpy() for _, val in model.state_dict().items()]
    #params2 = torch.round(model_2.state_dict()[el],decimals=4)
    params2 = model_2.state_dict()[el]

#    for x, y in zip(params, params2):
#        if x==y:
#            print(x, y)


    comparison = params == params2
    comparison = torch.isclose(params, params2)
    equal_arrays = comparison.all()

    diff = params - params2
    print(diff)

    print(equal_arrays)




#calc_noise(noise_file, 0)