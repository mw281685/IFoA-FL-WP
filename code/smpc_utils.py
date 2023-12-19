import csv
import run_config
from utils_quantisation import modulus, M
import numpy as np

def load_noise(file):
    with open(file, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        return list(csv_reader)


def calc_noise(file, ag_no):
    with open(file, mode='r', encoding='utf-8-sig') as infile:
        reader = csv.reader(infile)
        seeds = {int(rows[0]): rows[1:] for rows in reader}   # round_no : seeds for each collaborators
 
    noises ={}
    for rnd_no,v in seeds.items(): # round no, collaborators' seeds
        vect = np.zeros(162, dtype=np.int64)
        for i in range(len(v)):
            np.random.seed(int(v[i]))
            if run_config.QUANTISATION:
                if i == ag_no:
                    #vect = vect + np.around(np.random.random(162)*100,2)
                    vect = modulus(vect + modulus((run_config.server_config["num_clients"]-1)*np.random.randint(M + 1,size=162)))
                else:
                    vect = modulus(vect - np.random.randint(M + 1,size=162))  
            else:
                if i == ag_no:
                    vect = vect + np.round(np.random.uniform(-2,2,162),5)*(run_config.server_config["num_clients"]-1)
                else:
                    vect = vect - np.round(np.random.uniform(-2,2,162),5)

        noises[rnd_no] = vect
    print('noise:', noises)

    return noises

def calc_noise_zero(file, ag_no):
    with open(file, mode='r', encoding='utf-8-sig') as infile:
        reader = csv.reader(infile)
        seeds = {int(rows[0]): rows[1:] for rows in reader}   # round_no : seeds for each collaborators
 
    noises ={}
    for rnd_no,v in seeds.items(): # round no, collaborators' seeds
        vect = np.zeros(162)
        noises[rnd_no] = vect

    print('noise:', noises)
    return noises
    
