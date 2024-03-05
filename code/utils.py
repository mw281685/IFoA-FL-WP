import torch as th
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import run_config
import matplotlib.ticker as mtick
from run_config import NUM_FEATURES
import architecture
from skorch import NeuralNetRegressor, NeuralNet, callbacks

DATA_PATH = run_config.dataset_config["path"]
SEED = run_config.dataset_config["seed"]

def row_check(agents:int = 10):
      try:
            # Create empty dictionaries for validation and training data
            dataframe_X_train_dictionary = {}
            dataframe_X_val_dictionary = {}
            dataframe_y_train_dictionary = {}
            dataframe_y_val_dictionary = {}

            # Import training and validation data
            for i in range(agents):
                  dataframe_X_train_dictionary["X_train_{0}".format(i)] = pd.read_csv(f'../data/X_train_{i}.csv')
                  dataframe_X_val_dictionary["X_val_{0}".format(i)] = pd.read_csv(f'../data/X_val_{i}.csv')
                  dataframe_y_train_dictionary["y_tr_{0}".format(i)] = pd.read_csv(f'../data/y_tr_{i}.csv')
                  dataframe_y_val_dictionary["y_vl_{0}".format(i)] = pd.read_csv(f'../data/y_vl_{i}.csv')
      
            # Import test data
            X_test = pd.read_csv('../data/X_test.csv')  
            y_test = pd.read_csv('../data/y_test.csv')

            # Set row_count variable to sum
            row_count = 0

            # Set exposure_sum variable to sum
            exposure_sum = 0

            # Set claim_sum variable to sum
            claim_sum = 0

            # Add together row count of training and validation datasets
            for i in range(agents):
                  row_count += len(list(dataframe_X_train_dictionary.values())[i].index)
                  row_count += len(list(dataframe_X_val_dictionary.values())[i].index)

            total_row_count = row_count + len(X_test)

            assert total_row_count == 678_013, "Total number of rows should be 678 013"
            print('Total row count OK')

            # Add together exposure of training and validation datasets
            for i in range(agents):
                  exposure_sum  += list(dataframe_X_train_dictionary.values())[i]['Exposure'].sum()
                  exposure_sum  += list(dataframe_X_val_dictionary.values())[i]['Exposure'].sum()

            total_exposure_sum = exposure_sum + X_test['Exposure'].sum()

            assert round(total_exposure_sum, 2) == round(358_360.10546277853,2), "Exposure test failed"
            print('Total exposure OK')

            # Add together claims of training and validation datasets
            for i in range(agents):
                  claim_sum  += sum(list(dataframe_y_train_dictionary.values())[i]['0'])
                  claim_sum  += sum(list(dataframe_y_val_dictionary.values())[i]['0'])

            total_claim_sum = claim_sum + sum(y_test['0'])

            # Note the underscores are just for readability they don't affect the calculation  
            assert total_claim_sum == 36_056, "Claims check failed"
            print('Total claims sum OK')

      except Exception as e:
            print('Handling run-time error:', e)


def training_loss_curve(estimator, ag):
      # Save and graph training loss curves

      train_val_loss_df = pd.DataFrame(estimator.history[:, ['train_loss', 'valid_loss']], columns=['train_loss', 'valid_loss'])

      #plt.style.use('default')

      fig, ax = plt.subplots(figsize=(15,8))
      plt.plot(train_val_loss_df ['train_loss'],  label='Training Loss')
      plt.plot(train_val_loss_df ['valid_loss'],  label='Validation Loss')
      plt.legend(
            loc='upper right', 
            fontsize=15
            )
      plt.xlabel('Epochs', fontsize=15)
      plt.xticks(fontsize=15)
      plt.ylabel('Loss', fontsize=15)
      plt.yticks(fontsize=15)
      plt.grid()
      plt.title(f"Agent {ag}'s Best Model's Training Loss Curve", fontsize=15)

      plt.savefig(f'../ag_{ag}/' + 'agent_' + str(ag) + '_training_loss_chart', facecolor='white')


def load_individual_skorch_data(agent_id):
      #global model training:
      if agent_id == -1:
      # Created tensordataset
            MY_DATA_PATH = '../data'
            X_train_sc = pd.read_csv(MY_DATA_PATH + '/X_train.csv')
            X_column_names = X_train_sc.columns.tolist()

            y_tr = pd.read_csv(MY_DATA_PATH + '/y_tr.csv')

            X_val_sc = pd.read_csv(MY_DATA_PATH + '/X_val.csv')
            y_vl = pd.read_csv(MY_DATA_PATH + '/y_vl.csv')

            X_test_sc = pd.read_csv(MY_DATA_PATH + '/X_test.csv')
            y_te = pd.read_csv(MY_DATA_PATH + '/y_test.csv')

      else:

            MY_DATA_PATH = '../data'
            X_train_sc = pd.read_csv(MY_DATA_PATH + '/X_train_' + str(agent_id) + '.csv')
            X_column_names = X_train_sc.columns.tolist()

            y_tr = pd.read_csv(MY_DATA_PATH + '/y_tr_' + str(agent_id) +  '.csv')

            X_val_sc = pd.read_csv(MY_DATA_PATH + '/X_val_' + str(agent_id) + '.csv')
            y_vl = pd.read_csv(MY_DATA_PATH + '/y_vl_' + str(agent_id) + '.csv')

            X_test_sc = pd.read_csv(MY_DATA_PATH + '/X_test.csv')
            y_te = pd.read_csv(MY_DATA_PATH + '/y_test.csv')

      exposure = sum(X_train_sc['Exposure'])
      
      return (X_train_sc, y_tr, X_val_sc, y_vl, X_test_sc, y_te, X_column_names, exposure)



def seed_torch(seed=SEED):
      th.manual_seed(seed)
      random.seed(seed)
      th.cuda.manual_seed(seed)
      th.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
      np.random.seed(seed)  # Numpy module.
      random.seed(seed)  # Python random module.
      th.manual_seed(seed)
      th.backends.cudnn.benchmark = True
      th.backends.cudnn.deterministic = True

#seed_torch() # for FL training, we cannot set seed, as everyone will have same rng

def upload_dataset():
      seed_torch()
      
      df = pd.read_csv(DATA_PATH)
      df = df.sample(frac=1) # shuffle dataset to make sure claims are not sorted e.g by Year
      df.reset_index()
#      df = df0.sort_values('ClaimNb', ignore_index=True)

      #transformations and corrections
      df['VehPower'] = df['VehPower'].astype(object) # categorical ordinal
      df['ClaimNb'] = pd.to_numeric(df['ClaimNb'])
      df['ClaimNb'].values[df['ClaimNb']>4] = 4 # corrected for unreasonable observations (see M.V. Wuthrich)
      df['VehAge'].values[df['VehAge']>20] = 20 # capped for NN training (see M.V. Wuthrich)
      df['DrivAge'].values[df['DrivAge']>90] = 90 # capped for NN training (see M.V. Wuthrich)
      df['BonusMalus'].values[df['BonusMalus']>150] = 150 # capped for NN training (see M.V. Wuthrich)
      df['Density']=np.log(df['Density']) # logged for NN training     (see M.V. Wuthrich)
      df['Exposure'].values[df['Exposure']>1] = 1 # corrected for unreasonable observations (see M.V. Wuthrich)
      df_new = df.drop(['IDpol'], axis=1) # variable not used

      #Encode the data as per Wuthrich
      df_new_encoded = pd.get_dummies(df_new, columns=['VehBrand', 'Region'], drop_first=True)

      cleanup_nums = {"Area":     {"A": 1, "B": 2, "C": 3, "D": 4, "E":5, "F": 6},
                  "VehGas":   {"Regular": 1, "Diesel": 2} }

      #Apply label encoding - NOT ONE-HOT/DUMMY
      df_new_encoded = df_new_encoded.replace(cleanup_nums)

      #Apply MinMaxScaler as per Wuthrich
      #TODO this should really be done for each train, val, test data
      
      scaler = MinMaxScaler()
      df_new_encoded[['Area', 'VehPower', 'VehAge','DrivAge','BonusMalus','Density']] = scaler.fit_transform(df_new_encoded[['Area', 'VehPower', 'VehAge','DrivAge','BonusMalus','Density']])

      #Convert to numpy array 
      df_array=df_new_encoded.to_numpy()

      # Create Input and Output Data
      X = df_array[:, 1:]
      y = df_array[:, 0]

      #Data Splitting
      #Split data into train and final test
      X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2,  random_state=SEED)
      
      #Split train into train and validation
      X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1,  random_state=SEED)

      train_array = np.insert(X_train, 39, y_train, axis=1)
      val_array = np.insert(X_val, 39, y_val, axis=1)


      return (X_train, X_val, X_test, y_train, y_val, y_test, df_new_encoded.columns.tolist()[1:], scaler)



def prep_partitions(agents:int = 10):
      (X_train_sc, X_val_sc, X_test_sc, y_tr, y_vl, y_te, X_column_names, _) = upload_dataset()

      #whole dataset (agent_id = -1) 
      pd.DataFrame(X_train_sc, columns=X_column_names).to_csv(f'../data/X_train.csv' , index=False)
      pd.DataFrame(y_tr).to_csv(f'../data/y_tr.csv', index=False)

      pd.DataFrame(X_test_sc[:,0:NUM_FEATURES], columns=X_column_names).to_csv('../data/X_test.csv', index=False)
      pd.DataFrame(y_te).to_csv('../data/y_test.csv', index=False)

      pd.DataFrame(X_val_sc[:,0:NUM_FEATURES], columns=X_column_names).to_csv('../data/X_val.csv', index=False)
      pd.DataFrame(y_vl).to_csv('../data/y_vl.csv', index=False)


      train_array = np.insert(X_train_sc, NUM_FEATURES, y_tr, axis=1)
      val_array = np.insert(X_val_sc, NUM_FEATURES, y_vl, axis=1)

      print(f'train_array shape = {train_array.shape}, val_array shape = {val_array.shape}')

      # datasets for agents (0 .... agents-1)
      val_array_split = np.array_split(val_array, agents)
      train_array_len = train_array.shape[0]

      idx=[]

      if agents == 1:
            idx = [0, train_array_len]
      elif agents == 3:
            idx = [0, 1 * train_array_len // 6, 2 * train_array_len // 6, train_array_len]
      elif agents == 10:
            print('train_array_len: ', train_array_len)
            part = train_array_len//40

            parts=[0,1,3,5,5,5,5,5,5,5,1]
            idx.append(parts[0]*part)
            for no in range(1, agents):
                  idx.append(idx[no-1] + parts[no]*part)
            idx.append(train_array_len)


      for ag_no in range(1, agents + 1):
            X_train_sc = train_array[idx[ag_no-1]:idx[ag_no]][:,0:NUM_FEATURES]
            print(f' Agent no = {ag_no} has ranges : {idx[ag_no-1]} to {idx[ag_no]}')
            pd.DataFrame(X_train_sc, columns=X_column_names).to_csv(f'../data/X_train_{ag_no - 1}.csv' , index=False)
            y_tr = train_array[idx[ag_no - 1]:idx[ag_no]][:, NUM_FEATURES]
            pd.DataFrame(y_tr).to_csv(f'../data/y_tr_{ag_no -1}.csv', index=False)

      #truncate datas
      for idx in range(agents):
            print(f'Processing idx = {idx}')
            X_val_sc = val_array_split[idx][:, 0:NUM_FEATURES]
            pd.DataFrame(X_val_sc, columns=X_column_names).to_csv('../data/X_val_' + str(idx) + '.csv', index=False)
            y_vl = val_array_split[idx][:,NUM_FEATURES]
            pd.DataFrame(y_vl).to_csv('../data/y_vl_' + str(idx) + '.csv', index=False)



def load_partition(idx: int = -1, num_agents: int = 10):
      """Load 1/(num_agents) of the training and test data to simulate a partition."""

      (X_train_sc, X_val_sc, X_test_sc, y_tr, y_vl, y_te, X_column_names, _) = upload_dataset()


      train_array = np.insert(X_train_sc, NUM_FEATURES, y_tr, axis=1)
      val_array = np.insert(X_val_sc, NUM_FEATURES, y_vl, axis=1)

      #truncate data
      if idx in range(num_agents):
            train_array_split = np.array_split(train_array, num_agents)
            val_array_split = np.array_split(val_array, num_agents)
            X_train_sc = train_array_split[idx][:,0:NUM_FEATURES]
            y_tr = train_array_split[idx][:, NUM_FEATURES]
            X_val_sc = val_array_split[idx][:, 0:NUM_FEATURES]
            y_vl = val_array_split[idx][:,NUM_FEATURES]


      # Created tensordataset
      train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train_sc).float(), torch.from_numpy(y_tr).float())
      val_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val_sc).float(), torch.from_numpy(y_vl).float())
      test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test_sc).float(), torch.from_numpy(y_te).float())
      
      return (train_dataset, val_dataset, test_dataset, X_column_names)


def load_individual_data(agent_id, if_train_val=0): #if_train_val: True then validation data included in training set, else not included
      #global model training:
      if agent_id == -1:
      # Created tensordataset
            MY_DATA_PATH = '../data'
            X_train_sc = pd.read_csv(MY_DATA_PATH + '/X_train.csv')
            X_column_names = X_train_sc.columns.tolist()

            y_tr = pd.read_csv(MY_DATA_PATH + '/y_tr.csv')

            X_val_sc = pd.read_csv(MY_DATA_PATH + '/X_val.csv')
            y_vl = pd.read_csv(MY_DATA_PATH + '/y_vl.csv')

            X_test_sc = pd.read_csv(MY_DATA_PATH + '/X_test.csv')
            y_te = pd.read_csv(MY_DATA_PATH + '/y_test.csv')

      else:

            MY_DATA_PATH = '../data'
            X_train_sc = pd.read_csv(MY_DATA_PATH + '/X_train_' + str(agent_id) + '.csv')
            X_column_names = X_train_sc.columns.tolist()

            y_tr = pd.read_csv(MY_DATA_PATH + '/y_tr_' + str(agent_id) +  '.csv')

            X_val_sc = pd.read_csv(MY_DATA_PATH + '/X_val_' + str(agent_id) + '.csv')
            y_vl = pd.read_csv(MY_DATA_PATH + '/y_vl_' + str(agent_id) + '.csv')

            X_test_sc = pd.read_csv(MY_DATA_PATH + '/X_test.csv')
            y_te = pd.read_csv(MY_DATA_PATH + '/y_test.csv')

      exposure = sum(X_train_sc['Exposure'])

      print(f"Shape of the X_train_sc:  {X_train_sc.shape} y_tr {y_tr.shape} X_val_sc: {X_val_sc.shape} y_vl: {y_vl.shape} exposure {exposure}")

      #to align the code with scorch tuning where validation set is included in training set
      if if_train_val:
            X_train_sc = pd.concat([X_train_sc, X_val_sc], axis=0)
            y_tr = pd.concat([y_tr, y_vl], axis=0)
            exposure = sum(X_train_sc['Exposure']) + sum(X_val_sc['Exposure'])
            print(f"After concat: Shape of the X_train_sc:  {X_train_sc.shape} y_tr {y_tr.shape} exposure {exposure}")


      # Created tensordataset
      train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train_sc.values).float(), torch.tensor(y_tr.values).float())
      val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val_sc.values).float(), torch.tensor(y_vl.values).float())
      test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test_sc.values).float(), torch.tensor(y_te.values).float())
      
      return (train_dataset, val_dataset, test_dataset, X_column_names, torch.tensor(X_test_sc.values).float(), exposure)

def load_individual_data_lift(agent_id):
      MY_DATA_PATH = '../data'
      X_train_sc = pd.read_csv(MY_DATA_PATH + '/X_train_' + str(agent_id) + '.csv')
      X_column_names = X_train_sc.columns.tolist()

      y_tr = pd.read_csv(MY_DATA_PATH + '/y_tr_' + str(agent_id) +  '.csv')

      X_val_sc = pd.read_csv(MY_DATA_PATH + '/X_val_' + str(agent_id) + '.csv')
      y_vl = pd.read_csv(MY_DATA_PATH + '/y_vl_' + str(agent_id) + '.csv')

      X_test_sc = pd.read_csv(MY_DATA_PATH + '/X_test.csv')
      y_te = pd.read_csv(MY_DATA_PATH + '/y_test.csv')

      # Created tensordataset
      train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train_sc.values).float(), torch.tensor(y_tr.values).float())
      val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val_sc.values).float(), torch.tensor(y_vl.values).float())
      test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test_sc.values).float(), torch.tensor(y_te.values).float())
      
      return (train_dataset, val_dataset, test_dataset, X_train_sc, y_tr, X_val_sc, y_vl, X_test_sc, y_te)

#---------------------- Model predictions testing

def exp_model_predictions(model, test_loader):
      y_pred_list = []
      with th.no_grad():
            model.eval()
            for X_batch, _ in test_loader:
                  y_test_pred = model(X_batch)
                  y_pred_list.append(y_test_pred.cpu().numpy())
      y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
      return np.exp(y_pred_list)


def frequency_conversion(FACTOR, df, freq_dictionary):
      for key in freq_dictionary:
            df[freq_dictionary[key]]=df[key]/df['Exposure']

      df.insert(1,FACTOR+'_binned_midpoint',[round((a.left + a.right)/2,0) for a in df[FACTOR+'_binned']])


def one_way_graph(FACTOR, df, plot_name, ag,  *freq):
      data_preproc = df[[FACTOR+'_binned_midpoint', *freq]]
      plt.figure(figsize=(15,8))
      sns.set(style='dark',)

      sns.lineplot(data=pd.melt(data_preproc, [FACTOR+'_binned_midpoint']), x=FACTOR+'_binned_midpoint', y='value', hue='variable', linewidth=2.0).set(title= plot_name)
      #sns.set_style("ticks",{'axes.grid' : True})

      #plt.show()
      plt.savefig(f'../plots/' + plot_name)

      #plt.savefig(f'../ag_{ag}/' + plot_name)


def predictions_check(run_name, model_global, model_partial, model_fl, ag):
      
      (X_train, X_val, X_test, y_train, y_val, y_test, X_column_names, scaler) = upload_dataset()
      
      MY_DATA_PATH = '../data'
        
      X_test_sc = pd.read_csv(MY_DATA_PATH + '/X_test.csv')
      y_te = pd.read_csv(MY_DATA_PATH + '/y_test.csv')
      X_column_names = X_test_sc.columns.tolist()
      
      test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test_sc.values).float(), torch.tensor(y_te.values).float())
            
      
      test_loader = DataLoader(dataset=test_dataset, batch_size=1)


      test_complete_data=np.column_stack((X_test_sc, y_te))

      X_column_names.append('ClaimNb')

      #Convert dataset of test data, actuals, and prediction back into dataframe

      df_test=pd.DataFrame(data=test_complete_data,    # values
                     columns=X_column_names)  # 1st row as the column names
      

      df_test[['Area', 'VehPower', 'VehAge','DrivAge','BonusMalus','Density']]=scaler.inverse_transform(df_test[['Area', 'VehPower', 'VehAge','DrivAge','BonusMalus','Density']] )
      
      #Bin certain factors
      factor_list = ['Area', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehGas', 'Density']
      BINSIZE = 15

      for i in factor_list:
          df_test[i+'_binned'] = pd.cut(df_test[i], bins=BINSIZE, duplicates='drop')


      y_pred_list_exp = exp_model_predictions(model_global, test_loader)
      df_test['ClaimNb_pred']=pd.Series(y_pred_list_exp)

      y_partial_pred_list_exp = exp_model_predictions(model_partial, test_loader)
      df_test['ClaimNb_partial_pred']=pd.Series(y_partial_pred_list_exp)
      
      y_fl_pred_list_exp = exp_model_predictions(model_fl, test_loader)
      df_test['ClaimNb_fl_pred']=pd.Series(y_fl_pred_list_exp)

      # One way analysis
      FACTOR = 'VehAge'

      df_sum=df_test.groupby([FACTOR+'_binned'])['Exposure','ClaimNb', 'ClaimNb_pred', 'ClaimNb_partial_pred', 'ClaimNb_fl_pred'].sum().reset_index()

      frequency_conversion(FACTOR, df_sum, {'ClaimNb':'Actual freq', 'ClaimNb_pred':'Freq pred global model', 'ClaimNb_partial_pred':'Freq pred local model', 'ClaimNb_fl_pred':'Freq pred FL model'})

      one_way_graph(FACTOR, df_sum, run_name, ag,  'Actual freq', 'Freq pred global model', 'Freq pred local model','Freq pred FL model')


# Lorenz Curves

def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_true = y_true.reshape((len(y_true), ))
    y_pred = y_pred.reshape((len(y_pred), ))
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_frequencies = y_true[ranking]
    ranked_exposure = exposure[ranking]
    cumulated_claims = np.cumsum(ranked_frequencies * ranked_exposure)
    cumulated_claims /= cumulated_claims[-1]
    cumulated_exposure = np.cumsum(ranked_exposure)
    cumulated_exposure /= cumulated_exposure[-1]
    
    return cumulated_exposure, cumulated_claims
    

def uniform_partitions(agents:int = 10):
      (X_train_sc, X_val_sc, X_test_sc, y_tr, y_vl, y_te, X_column_names, _) = upload_dataset()
      
      # Training dataset 
      pd.DataFrame(X_train_sc, columns=X_column_names).to_csv(f'../data/X_train.csv' , index=False)
      pd.DataFrame(y_tr).to_csv(f'../data/y_tr.csv', index=False)

      # Test dataset
      pd.DataFrame(X_test_sc[:,0:NUM_FEATURES], columns=X_column_names).to_csv('../data/X_test.csv', index=False)
      pd.DataFrame(y_te).to_csv('../data/y_test.csv', index=False)

      # Validation dataset 
      pd.DataFrame(X_val_sc, columns=X_column_names).to_csv(f'../data/X_val.csv' , index=False)
      pd.DataFrame(y_vl).to_csv(f'../data/y_vl.csv', index=False)

      # Create empty dictionaries for validation and training data
      train_array_dictionary = {}
      val_array_dictionary = {}

      train_array = np.insert(X_train_sc, NUM_FEATURES, y_tr, axis=1)
      val_array = np.insert(X_val_sc, NUM_FEATURES, y_vl, axis=1)

      # Seed numpy etc. for shuffling
      seed_torch()

      # Shuffle arrays
      np.random.shuffle(train_array)
      np.random.shuffle(val_array)

      val_array_split = np.array_split(val_array, agents)
      train_array_split = np.array_split(train_array, agents)

      for i in range(agents):
            train_array_dictionary["X_train_{0}".format(i)] = train_array_split[i]
            pd.DataFrame(train_array_split[i][:, 0:NUM_FEATURES], columns=X_column_names).to_csv(f'../data/X_train_{i}.csv' , index=False)
            pd.DataFrame(train_array_split[i][:, NUM_FEATURES]).to_csv(f'../data/y_tr_{i}.csv' , index=False)
            val_array_dictionary["X_val_{0}".format(i)] = val_array_split[i]
            pd.DataFrame(val_array_split[i][:, 0:NUM_FEATURES], columns=X_column_names).to_csv(f'../data/X_val_{i}.csv' , index=False)
            pd.DataFrame(val_array_split[i][:, NUM_FEATURES]).to_csv(f'../data/y_vl_{i}.csv' , index=False)

def load_model(agent=-1, num_features=NUM_FEATURES):
    
    all_results_df = pd.read_csv('../results/all_results.csv')
    top_results_df = all_results_df.loc[all_results_df['rank_test_score']==1]
    top_results_dict = top_results_df[['agent', 'param_module__num_units_1', 'param_module__num_units_2']].set_index('agent').to_dict('index')

    num_units_1 = list(top_results_dict[agent].items())[0][1]
    num_units_2 = list(top_results_dict[agent].items())[1][1]

    loaded_agent_model = NeuralNetRegressor(architecture.MultipleRegression(num_features, num_units_1, num_units_2).double())
    loaded_agent_model.initialize()  # This is important!
    loaded_agent_model.load_params(f_params=f'../ag_'+str(agent)+'/agent_'+str(agent)+'_model.pkl', 
                                       f_optimizer=f'../ag_'+str(agent)+'/agent_'+str(agent)+'_opt.pkl', 
                                       f_history=f'../ag_'+str(agent)+'/agent_'+str(agent)+'_history.json')
    
    return loaded_agent_model

def frequency_conversion(FACTOR, df, freq_dictionary):
      for key in freq_dictionary:
            df[freq_dictionary[key]]=df[key]/df['Exposure']

      df.insert(1,FACTOR+'_binned_midpoint',[round((a.left + a.right)/2,0) for a in df[FACTOR+'_binned']])

def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

def create_test_data(): 
    (X_train, X_val, X_test, y_train, y_val, y_test, X_column_names, scaler) = upload_dataset()
    
    MY_DATA_PATH = '../data'
    
    X_test_sc = pd.read_csv(MY_DATA_PATH + '/X_test.csv')
    y_te = pd.read_csv(MY_DATA_PATH + '/y_test.csv')
    X_column_names = X_test_sc.columns.tolist()

    test_complete_data=np.column_stack((X_test_sc, y_te))

    X_column_names.append('ClaimNb')

    #Convert dataset of test data, actuals, and prediction back into dataframe

    df_test=pd.DataFrame(data=test_complete_data,    # values
                    columns=X_column_names)  # 1st row as the column names
    
    # Un one-hot encode Region and VehBrand
    df_test = undummify(df_test)
    df_test['VehBrand_number'] = df_test['VehBrand'].str[1:].astype(int)
    df_test['Region_number'] = df_test['Region'].str[1:].astype(int)


    df_test[['Area', 'VehPower', 'VehAge','DrivAge','BonusMalus','Density']]=scaler.inverse_transform(df_test[['Area', 'VehPower', 'VehAge','DrivAge','BonusMalus','Density']] )
    
    #Bin factors
    factor_list = ['Area', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehGas', 'Density', 'VehBrand_number', 'Region_number']
    
    # If fewer levels than MAX_BINSIZE no effective binning happens via the min() function, not used currently
    MAX_BINSIZE = 20

    # Automatic binning not used, could use qcut as well 
    for i in factor_list:
        df_test[i+'_binned'] = pd.cut(df_test[i], bins=min(len(df_test[i].unique()), MAX_BINSIZE), duplicates='drop')
        #df_test[i+'_binned'] = pd.cut(df_test[i], bins=np.linspace(0, max(df_test[i]), min(21, len(df_test[i].unique()))), duplicates='drop')
    
    # Custom binning used instead to define more sensible bins
    df_test['Area'+'_binned'] = pd.cut(df_test['Area'], bins=np.linspace(0, 7, 8), duplicates='drop')
    df_test['VehPower'+'_binned'] = pd.cut(df_test['VehPower'], bins=np.linspace(0, 15, 16), duplicates='drop')
    df_test['VehAge'+'_binned'] = pd.cut(df_test['VehAge'], bins=np.linspace(0, 20, 21), duplicates='drop')
    df_test['DrivAge'+'_binned'] = pd.cut(df_test['DrivAge'], bins=np.linspace(0, 100, 21), duplicates='drop')
    df_test['BonusMalus'+'_binned'] = pd.qcut(df_test['BonusMalus'], q=15, duplicates='drop') # note usin qcut here
    df_test['Density'+'_binned'] = pd.cut(df_test['Density'], bins=np.linspace(0, 11, 12), duplicates='drop')
    df_test['VehBrand_number'+'_binned'] = pd.cut(df_test['VehBrand_number'], bins=np.linspace(0, 15, 16), duplicates='drop')
    df_test['Region_number'+'_binned'] = pd.cut(df_test['Region_number'], bins=np.linspace(0, 100, 21), duplicates='drop')
    df_test['VehGas'+'_binned'] = pd.cut(df_test['VehGas'], bins=np.linspace(0, 2, 3), duplicates='drop')
    
    return X_test, y_test, df_test

def create_df_test_pred(df_test, X_test, NUM_AGENTS, global_model, fl_model, agent_model_dictionary):
    
    # Global Model Predictions
    y_pred = global_model.predict(X_test.astype(np.float64))
    df_test['ClaimNb_pred']=pd.Series(y_pred.flatten())

    # FL Model Predictions
    y_fl_pred = fl_model.predict(th.tensor(X_test).float())
    df_test['ClaimNb_fl_pred']=pd.Series(y_fl_pred.flatten().detach().numpy())

    # Local Model Predictions

    agent_prediction_dictionary = {}

    for agents in range(NUM_AGENTS):
        agent_prediction_dictionary["y_agent_{0}_pred".format(agents)] = agent_model_dictionary['loaded_agent_'+str(agents)+'_model'].predict(X_test.astype(np.float64))
        df_test['ClaimNb_agent_'+str(agents)+'_pred']=pd.Series(agent_prediction_dictionary['y_agent_'+str(agents)+'_pred'].flatten())

    return df_test

def create_df_sum(df_test_pred, factor, NUM_AGENTS):

    sum_list = ['Exposure',  'ClaimNb', 'ClaimNb_pred', 'ClaimNb_fl_pred']
    sum_dictionary = {'ClaimNb':'Actual freq', 'ClaimNb_pred':'Freq pred global model', 'ClaimNb_fl_pred':'Freq pred FL model'}

    for agents in range(NUM_AGENTS):
        sum_list.append('ClaimNb_agent_'+str(agents)+'_pred')
        sum_dictionary['ClaimNb_agent_'+str(agents)+'_pred']='Freq pred agent '+str(agents)+' model'

    df_sum=df_test_pred.groupby([factor+'_binned'])[sum_list].sum().reset_index()

    frequency_conversion(factor, df_sum, sum_dictionary)

    # Remove rows with 0 exposure
    df_sum = df_sum.loc[df_sum['Exposure']!=0]
    df_sum = df_sum.reset_index(drop=True)

    return df_sum

def one_way_graph_comparison(factor, df_test_pred, agents_to_graph_list, NUM_AGENTS):

        #NUM_AGENTS = len(agents_to_graph_list)

        df_sum = create_df_sum(df_test_pred, factor, NUM_AGENTS)
        
        fig, ax = plt.subplots(figsize=(12, 8))

        plt.plot(df_sum['Actual freq'],  
                label='Actual freq',
                marker='s',
                markersize=10,
                )
        
        for agents in agents_to_graph_list:
                plt.plot(df_sum['Freq pred agent '+str(agents)+' model'],  
                label='Freq pred agent '+str(agents)+' model',
                marker='o',
                markersize=5,
                #linestyle=(0, (1, 10)),
                linestyle='dotted',
                )


        plt.plot(df_sum['Freq pred FL model'],  
                label='Freq pred FL model',
                marker='o',
                markersize=10,
                #linestyle='dotted',
                )
        
        plt.plot(df_sum['Freq pred global model'],  
                label='Freq pred global model',
                marker='s',
                markersize=10,
                #linestyle='dotted',
                )

        plt.legend(bbox_to_anchor=(1.08, 1), loc='upper left', borderaxespad=0)

        plt.xlabel(factor+' binned')
        plt.xticks(rotation = 75)
        plt.ylabel('Frequency')
        plt.title('Actual vs. Expected by '+factor)
        plt.grid()


        # Get second axis
        ax2 = ax.twinx()

        plt.bar(df_sum[factor+'_binned'].astype(str), 
                df_sum['Exposure'], 
                label='Exposure', 
                color='y',
                alpha=0.35
                )

        plt.ylabel('Exposure', color='y')
        plt.xticks(rotation = 90)

        vals = ax2.get_yticks()
        ax2.set_yticklabels(['{:,.0f}'.format(x) for x in vals])

        plt.legend(bbox_to_anchor=(1.08, 0), loc='upper left', borderaxespad=0)

        plt.show()

def double_lift_rebase(df_test_pred, model1, model2):

    # Rename 
    df_test_pred['ClaimNb_global_pred'] = df_test_pred['ClaimNb_pred']
    # Compare models
    df_test_pred[model1+'_vs_'+model2] = df_test_pred['ClaimNb_'+model1+'_pred']/df_test_pred['ClaimNb_'+model2+'_pred']
    # Bin comparison
    df_test_pred[model1+'_vs_'+model2+'_binned'] = pd.qcut(df_test_pred[model1+'_vs_'+model2], 10)
    # Sum up predictions
    sum_list = ['Exposure',  
                'ClaimNb', 
                'ClaimNb_pred', 
                'ClaimNb_'+model1+'_pred',
                'ClaimNb_'+model2+'_pred',
                ]
    
    df_sum=df_test_pred.groupby([model1+'_vs_'+model2+'_binned'])[sum_list].sum().reset_index()
    # Rebase
    df_sum['ClaimNb_rebased'] = df_sum['ClaimNb']/df_sum['ClaimNb']
    df_sum['ClaimNb_'+model1+'_pred'+'_rebased'] = df_sum['ClaimNb_'+model1+'_pred']/df_sum['ClaimNb']
    df_sum['ClaimNb_'+model2+'_pred'+'_rebased'] = df_sum['ClaimNb_'+model2+'_pred']/df_sum['ClaimNb']
    # Graph
    fig, ax = plt.subplots(figsize=(12, 8))

    plt.plot(df_sum['ClaimNb_rebased'],  
            label='Actual freq',
            marker='s',
            markersize=10,
            )

    plt.plot(df_sum['ClaimNb_'+model2+'_pred'+'_rebased'],  
            label='Freq pred '+model2+' model',
            marker='o',
            markersize=10,
            #linestyle=(0, (1, 10)),
            #linestyle='dotted',
            )

    plt.plot(df_sum['ClaimNb_'+model1+'_pred'+'_rebased'],  
            label='Freq pred '+model1+' model',
            marker='o',
            markersize=10,
            #linestyle='dotted',
            )

    plt.legend(bbox_to_anchor=(1.08, 1), loc='upper left', borderaxespad=0)

    plt.xlabel('Model 1 Prediction Over Model 2 Prediction')
    plt.xticks(rotation = 75)
    plt.ylabel('Actual Over Expected')
    plt.title('Double Lift Chart Comparing Model Performance')
    plt.grid()


    # Get second axis
    ax2 = ax.twinx()

    plt.bar(df_sum[model1+'_vs_'+model2+'_binned'].astype(str), 
            df_sum['Exposure'], 
            label='Exposure', 
            color='y',
            alpha=0.35
            )

    plt.ylabel('Exposure', color='y')
    plt.xticks(rotation = 90)

    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.0f}'.format(x) for x in vals])

    plt.legend(bbox_to_anchor=(1.08, 0), loc='upper left', borderaxespad=0)

    plt.show()