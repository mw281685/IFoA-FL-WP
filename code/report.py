from cProfile import label
from calendar import EPOCH
from fpdf import FPDF
from run_config import EPOCHS_LOCAL_GLOBAL
import utils
import run_config
import architecture
import shutil
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import json
import run_config
import os
import calendar
from datetime import datetime
import torch
from torch import nn, optim
from sklearn.metrics import auc

# Run config:
NUM_AGENTS = utils.run_config.dataset_config["num_agents"]
NUM_ROUNDS = utils.run_config.server_config["num_rounds"]
NUM_FEATURES = run_config.model_architecture["num_features"]
EPOCHS = run_config.model_architecture["epochs"]
EPOCHS_LOCAL_GLOBAL = run_config.EPOCHS_LOCAL_GLOBAL

MARGIN = 10 # Margin
pw = 210 - 2*MARGIN # Page width: Width of A4 is 210mm
ch = 10 # Cell height

PLOT_DIR = '../plots'

# delete elements from folder ../plots
try:
    shutil.rmtree(PLOT_DIR)
    os.mkdir(PLOT_DIR)
except FileNotFoundError:
    os.mkdir(PLOT_DIR)


# Custom class to overwrite the header and footer methods
class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.WIDTH = 150
        self.HEIGHT = 350
    
    def page_body(self, images):
        # Determine how many plots there are per page and set positions
        # and margins accordingly
        if len(images) == 3:
            self.image(images[0], 15, 25, self.WIDTH - 30)
            self.image(images[1], 15, 90, self.WIDTH - 30)
            self.image(images[2], 15, 160, self.WIDTH - 30)
        elif len(images) == 2:
            self.image(images[0], 15, 25, self.WIDTH - 30)
            self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
        else:
            self.image(images[0], 15, 25, self.WIDTH - 30)
            
    def print_page(self, images):
        # Generates the report
        self.add_page()
        self.page_body(images)

pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', '', 12)

text = "Run Results: num_agents: " + str(NUM_AGENTS) + "; num_rounds: " + str(NUM_ROUNDS) + "; epochs: " + str(EPOCHS) + "; epochs local and global: " + str(EPOCHS_LOCAL_GLOBAL)
pdf.cell(w=0, h=ch, txt=text, border=1, ln=1)

#------------------------------------------------------------------------------------------------
# 1. Data characteristics: 
#------------------------------------------------------------------------------------------------
(X_train, X_val, X_test, y_train, y_val, y_test, X_column_names, scaler) = utils.upload_dataset()

pdf.ln(ch)

def claims_count_analysis(y_train):
    unique, counts  = np.unique(y_train, return_counts=True)
    df = pd.DataFrame(zip(unique, counts/sum(counts)*100), columns=['ClaimsCount', 'Prc'])

    # Table Header
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=35, h=ch, txt='Count', border=1, ln=0, align='C')
    pdf.cell(w=35, h=ch, txt='Prc', border=1, ln=1, align='C')
    # Table contents
    pdf.set_font('Arial', '', 12)
    for i in range(0, len(df)):
        pdf.cell(w=35, h=ch, 
                txt=df['ClaimsCount'].iloc[i].astype(str)[:1], 
                border=1, ln=0, align='C')
        pdf.cell(w=35, h=ch, 
                txt=df['Prc'].iloc[i].astype(str)[0:4], 
                border=1, ln=1, align='C')


def generate_counts(agents):
    MY_DATA_PATH = '../data'
    for agent_id in range(agents):
        y_tr = pd.read_csv(MY_DATA_PATH + '/y_tr_' + str(agent_id) +  '.csv')
        claims_count_analysis(y_tr)

#generate_counts(3)

def claims_counts(agents):
    MY_DATA_PATH = '../data'
    agents_counts= dict()

    for agent_id in range(agents):
        y_train = pd.read_csv(MY_DATA_PATH + '/y_tr_' + str(agent_id) +  '.csv')
        unique, counts  = np.unique(y_train, return_counts=True)
        df = pd.DataFrame(zip(unique, counts/sum(counts)*100), columns=['ClaimsCount', 'Prc'])
        agents_counts[agent_id] =  df['Prc']
    return agents_counts

#------------------------- grouped bar plot: - TO DO : automate for 4 +  agents!!!!!!!!!

labels = ['0', '1', '2', '3', '4']
agents_claims_counts = claims_counts(NUM_AGENTS)

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

plt.figure(0)
fig, ax = plt.subplots()

rects1 = ax.bar(x - (width*1.5), agents_claims_counts[0], width, label='agent 0')
rects2 = ax.bar(x - width*0.5 , agents_claims_counts[1], width, label='agent 1')
rects3 = ax.bar(x + width*0.5, agents_claims_counts[2], width, label='agent 2')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Prc of data')
ax.set_title('Claims counts distribution by agent_id')
ax.set_xticks(x, labels)
ax.legend()

fig.tight_layout()

plt.savefig('../plots/claim_counts.png', 
           transparent=False,  
           facecolor='white', 
           bbox_inches="tight")
plt.close()

pdf.cell(w=0, h=ch, txt="Input Data: Distribution of numbers of observed claims by FL participating parties.", border=0, ln=1)

pdf.ln(12)
pdf.image('../plots/claim_counts.png', x = 50, y = None, w = 100, h = 0, type = 'PNG')

pdf.cell(w=0, h=ch, txt="Model Training: Learning curves; Train dataset", border=0, ln=1)

#------------------------------------------------------------------------------------------------
# 2. Learning curves: 
#------------------------------------------------------------------------------------------------

def read_csv(filename):
    loss_stats_dict = dict()
    with open(filename) as f:
        file_data=csv.reader(f)
        headers=next(file_data)
        round = 0
        for i in file_data:
            loss_stats_dict[round] = dict(zip(headers,i))
            round +=1
    return loss_stats_dict

loss_stats = dict()
for ag_id in range(-1, NUM_AGENTS):
    if ag_id == -1:
        MY_DATA_PATH = '../ag_global'
    else:    
        MY_DATA_PATH = '../ag_' + str(ag_id)
    
    loss_stats[ag_id] = read_csv(MY_DATA_PATH + '/los_stats.csv')


# for each agent plot learning curves
def plot_learning_curves(loss_stats, train_or_val):
    for ag_id in range(-1, NUM_AGENTS):
        # plot lines
        plt.figure()
        fig, ax = plt.subplots()

        ax.set_ylabel('loss')
        if ag_id == -1:    
            ax.set_title('Learning curves global_model : ' + train_or_val + ' dataset')
        else:
            ax.set_title('Learning curves ' +  str(ag_id) + ' : ' + train_or_val + ' dataset')

        if ag_id == -1:
            #x = range(1, 10 + 1)
            #y = json.loads(loss_stats[ag_id][1][train_or_val])
            #plt.plot(x, y, label = "global model" , linestyle="-")
            pass
        else:
            x = range(1, EPOCHS + 1)   
            for rnd_no in range(0, NUM_ROUNDS*2 +1):
                if rnd_no % 2 != 0: # FIXED : TO_FIX !!!!!!!!    MS: -1 added temporarly!!!! I need to find out why some clients join from 2nd round ( ask Malgorzata if that is not clear :) 
                    print('ag_id ', ag_id, 'rnd_no ', rnd_no)
                    y = json.loads(loss_stats[ag_id][rnd_no][train_or_val])
                    plt.plot(x, y, label = "round_" + str(rnd_no + 1) , linestyle="-")

        plt.legend()
        plot_name = '../plots/Learning curves agent_id ' + str(ag_id) + train_or_val + '.png'
        plt.savefig(plot_name, 
            transparent=False,  
            facecolor='white', 
            bbox_inches="tight")
        plt.close()

        pdf.image(plot_name, x = 50, y = None, w = 100, h = 0, type = 'PNG')
        pdf.cell(w=0, h=2, txt='', border=0, ln=1)

plot_learning_curves(loss_stats, "train")
pdf.cell(w=0, h=ch, txt="Model Training: Learning curves; Validation dataset", border=0, ln=1)
plot_learning_curves(loss_stats, "val")

def construct():
    # Delete folder if exists and create it again
    try:
        shutil.rmtree(PLOT_DIR)
        os.mkdir(PLOT_DIR)
    except FileNotFoundError:
        os.mkdir(PLOT_DIR)
        
    # Iterate over all agent's folders and create plots
    for ag in range(NUM_AGENTS):

        AGENT_PATH =  '../ag_global/global_model.pt'
        model_global = architecture.MultipleRegression(num_features=39, num_units_1=60, num_units_2=20)  #architecture.NeuralNetworks(NUM_FEATURES) 
        model_global.load_state_dict(torch.load(AGENT_PATH))
        model_global.eval()

        AGENT_PATH =  f'../ag_{ag}/local_model.pt'
        model_partial = architecture.MultipleRegression(num_features=39, num_units_1=20, num_units_2=100)  #architecture.NeuralNetworks(NUM_FEATURES)
        model_partial.load_state_dict(torch.load(AGENT_PATH))
        model_partial.eval()

        AGENT_PATH = f'../ag_{ag}/fl_model.pt'
        model_fl = architecture.MultipleRegression(num_features=39, num_units_1=20, num_units_2=100) #architecture.NeuralNetworks(NUM_FEATURES)
        model_fl.load_state_dict(torch.load(AGENT_PATH))
        model_fl.eval()

        # Save one_way-graph visualization
        utils.predictions_check( 'One way graph Company ' + str(ag + 1) + ' ' + str(run_config.server_config["num_rounds"]) + ' FL training rounds '  + str(run_config.model_architecture["epochs"]) + ' epochs ' + str(run_config.server_config["num_clients"]) + ' training participants', model_global, model_partial, model_fl, ag)

        train_dataset, val_dataset, test_dataset, X_train_sc, y_tr, X_val_sc, y_vl, X_test_sc, y_te = utils.load_individual_data_lift(ag)        
        fig, ax = plt.subplots(figsize=(15, 8))

        fl = 1
        partial = 0

        for model in [model_fl, model_partial, model_global]:
            log_y_pred = []
            with torch.no_grad():
                model.eval()
                log_y_pred = model(torch.tensor(X_test_sc.values).float())
            log_y_pred = [a.squeeze().tolist() for a in log_y_pred]
            y_pred = np.exp(log_y_pred)
            cum_exposure, cum_claims = utils.lorenz_curve(
                y_te, y_pred, X_test_sc["Exposure"]
            )
            gini = 1 - 2 * auc(cum_exposure, cum_claims)
            print('Gini ', gini)

            if fl:
                label = "(FL model Gini: {:.3f})".format(gini)
                fl = 0
                partial = 1
                color = "red"
            elif partial:
                label = "(Partial model Gini: {:.3f})".format(gini)
                fl = 0
                partial = 0
                color = "green"
            else: 
                label = "(Global model Gini: {:.3f})".format(gini)
                fl = 1
                color = "orange"

            print('agent ' + str(ag) + ' ' + label )
            ax.plot(cum_exposure, cum_claims, linestyle="-", linewidth=2.0, color=color, label=label)

        # Oracle model: y_pred == y_test
        cum_exposure, cum_claims = utils.lorenz_curve(
            y_te, y_te, X_test_sc["Exposure"]
        )
        gini = 1 - 2 * auc(cum_exposure, cum_claims)
        label = "Oracle (Gini: {:.3f})".format(gini)
        ax.plot(cum_exposure, cum_claims, linestyle="-.", linewidth=2.0, color="gray", label=label)

        # Random Baseline
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0, color="black", label="Random baseline")
        ax.set(
            title='Company ' + str(ag) + ': Lorenz curves by model',
            xlabel="Cumulative proportion of exposure (from safest to riskiest)",
            ylabel="Cumulative proportion of claims",
        )
        ax.legend(loc="upper left")

        # lift charts
        plt.savefig(f'../plots/' + 'Lift chart Company ' + str(ag) + 'Training config ' + str(run_config.server_config["num_rounds"]) + ' FL training rounds '  + str(run_config.model_architecture["epochs"]) + ' epochs ' + str(run_config.server_config["num_clients"]) + ' training participants' )

    # Construct data shown in document
    counter = 0
    pages_data = []
    temp_one_way = []
    # Get all plots
    files = os.listdir(PLOT_DIR)
    # Sort them by month - a bit tricky because the file names are strings
    files = sorted(os.listdir(PLOT_DIR), key=lambda x: x.split(';')[0])
    # Iterate over all created visualization
    for fname in files:
        # We want 3 per page
        if counter == 3:
            pages_data.append(temp_one_way)
            temp_one_way = []
            counter = 0

        temp_one_way.append(f'{PLOT_DIR}/{fname}')
        counter += 1

    return [*pages_data, temp_one_way]


X_test_sc = pd.read_csv( '../data/X_test.csv')
y_te = pd.read_csv('../data/y_test.csv')

def model_predictions(model):
    log_y_pred = []
    with torch.no_grad():
        model.eval()
        log_y_pred = model(torch.tensor(X_test_sc.values).float())
    log_y_pred = [a.squeeze().tolist() for a in log_y_pred]
    #y_pred_list_exp = np.exp(log_y_pred)
    return log_y_pred

from sklearn.metrics import auc, mean_squared_error, r2_score, mean_poisson_deviance, explained_variance_score, d2_tweedie_score
from scipy import stats

def test_statistics(model, y_test, ag_no):
    y_pred_list = model_predictions(model) # it doesn't exp actually 
    mpd = mean_poisson_deviance(y_test, y_pred_list)
    ppde = d2_tweedie_score(y_test, y_pred_list, sample_weight=X_test_sc['Exposure'], power=1) # percentage of poisson deviance explained
    mse = mean_squared_error(y_test, y_pred_list)
    r_square = r2_score(y_test, y_pred_list)
    explained_variance = explained_variance_score(y_test, y_pred_list)

    pdf.set_font('Arial', '', 12)
    if ag_no ==-1:
        pdf.cell(w=0, h=ch, txt="Test statistics for global model " , border=1, ln=1)
    elif ag_no ==-2:
        pdf.cell(w=0, h=ch, txt="Test statistics for FL model " , border=1, ln=1)
    else:
        pdf.cell(w=0, h=ch, txt="Test statistics for agent_no " + str(ag_no), border=1, ln=1)
    
    pdf.set_font('Arial', '', 11)
    pdf.cell(w=0, h=ch, txt="Mean Poisson Deviance : {:.3f} ".format(mpd), border=0, ln=1)
    pdf.cell(w=0, h=ch, txt="Prc Poisson Deviance Explained : {:.3f} ".format(ppde), border=0, ln=1)
    pdf.cell(w=0, h=ch, txt="Mean Squared Error : {:.3f} ".format(mse), border=0, ln=1)
    pdf.cell(w=0, h=ch, txt="R^2 : {:.3f} ".format(r_square), border=0, ln=1)
    pdf.cell(w=0, h=ch, txt="EV  : {:.3f} ".format(explained_variance), border=0, ln=1)

    #pdf.cell(w=0, h=ch, txt="Stats: " + str(stats.describe(y_pred_list)), border=0, ln=1)

    pdf.set_font('Arial', '', 12)


for ag_no in range(-2, NUM_AGENTS):

    if ag_no ==-1:
        #AGENT_PATH = '../ag_global/global_model.pt' 
        pass
    elif ag_no ==-2:
        #model_name = 'fl_model.pt'
        #AGENT_PATH = '../ag_0/fl_model.pt'
        pass
    else:
        model_name = 'local_model.pt'
        AGENT_PATH = '../ag_' + str(ag_no) + '/' + model_name 
    

    model = architecture.MultipleRegression(num_features=39, num_units_1=20, num_units_2=100) # 2 layer NN, new architecture used in Optuna tuning


    #model.load_state_dict(torch.load(AGENT_PATH))
    #model.eval()
    #test_statistics(model, y_te, ag_no)   


plots_on_page = construct()
plots_on_page


for elem in plots_on_page:
    pdf.print_page(elem)

run_name = utils.run_config.run_name

pdf.output(f'../results/' + run_name + '.pdf', 'F')